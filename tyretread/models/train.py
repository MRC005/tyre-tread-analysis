"""Model selection, calibration, threshold choice and artifact production.

Everything a model needs before it can be served is decided here, from
out-of-fold predictions only. The distinction matters: the original project chose
its three-class boundaries from predictions the model had made on its own training
data, which is why those boundaries were never a measurement of anything
(docs/AUDIT.md 3.4).

Order of operations, and why
----------------------------
1. **Candidates are pipelines, not bare estimators.** Scaling must be fitted inside
   each fold. Fitting a scaler on the whole dataset before cross-validation leaks
   the test folds' distribution into training.
2. **Probabilities are calibrated.** An abstention rule is only meaningful if the
   probability it thresholds means what it says. Platt scaling is used rather than
   isotonic regression because isotonic needs more data than this project has and
   overfits badly at these sample sizes.
3. **Model choice is a paired comparison on identical folds**, so a difference can
   be tested rather than eyeballed.
4. **The threshold and the abstention band are chosen from out-of-fold
   probabilities**, then the final estimator is refitted on everything.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import EvaluationResult, compare_models, cross_validate

__all__ = ["CandidateSpec", "candidate_models", "ThresholdChoice",
           "out_of_fold_probabilities", "choose_threshold", "calibration_metrics",
           "select_model"]


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    build: Any
    rationale: str


def _calibrated(estimator: Any, *, seed: int, folds: int = 5) -> Any:
    """Wrap ``estimator`` in Platt scaling fitted by internal cross-validation."""
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=folds, ensemble=True)


def candidate_models(seed: int = 42) -> list[CandidateSpec]:
    """The models worth comparing on a few dozen engineered features.

    Deliberately excluded: gradient-boosted trees such as XGBoost or CatBoost. The
    audit measured tree ensembles on this feature set and they were the weakest
    family tried (balanced accuracy 0.677-0.692 against 0.759 for logistic
    regression), which is the expected outcome for a few hundred samples described
    by smooth, correlated, continuous features. Adding a heavier library to lose
    accuracy would be decoration.
    """
    def pipe(clf: Any) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return [
        CandidateSpec(
            "logistic_l2",
            lambda: _calibrated(pipe(LogisticRegression(
                max_iter=5000, class_weight="balanced", C=1.0, random_state=seed)), seed=seed),
            "Linear, few parameters, naturally well-calibrated. Beat the RBF-SVM "
            "significantly on the legacy features (audit 3.9), which is what a small "
            "sample of smooth features usually rewards.",
        ),
        CandidateSpec(
            "logistic_l2_strong",
            lambda: _calibrated(pipe(LogisticRegression(
                max_iter=5000, class_weight="balanced", C=0.1, random_state=seed)), seed=seed),
            "More regularisation, in case the wider feature set needs it.",
        ),
        CandidateSpec(
            "svm_rbf_c10",
            lambda: _calibrated(pipe(SVC(
                kernel="rbf", C=10, gamma="scale", class_weight="balanced",
                random_state=seed)), seed=seed),
            "The originally published model, kept as the incumbent so any change is "
            "measured against it rather than assumed.",
        ),
        CandidateSpec(
            "svm_rbf_c1",
            lambda: _calibrated(pipe(SVC(
                kernel="rbf", C=1, gamma="scale", class_weight="balanced",
                random_state=seed)), seed=seed),
            "A less aggressive RBF, since C=10 showed signs of overfitting.",
        ),
        CandidateSpec(
            "svm_linear",
            lambda: _calibrated(pipe(SVC(
                kernel="linear", C=1, class_weight="balanced", random_state=seed)), seed=seed),
            "Separates the effect of the kernel from the effect of the loss function.",
        ),
        CandidateSpec(
            "random_forest",
            lambda: _calibrated(RandomForestClassifier(
                n_estimators=500, class_weight="balanced", min_samples_leaf=2,
                random_state=seed, n_jobs=-1), seed=seed),
            "A non-linear, scale-free baseline. Included so the claim that trees do "
            "not help here stays a measurement rather than a memory.",
        ),
    ]


def out_of_fold_probabilities(
    build: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Average out-of-fold probability per sample, across repeats.

    Returns ``(probabilities, coverage)`` where coverage counts how many repeats
    contributed to each sample. Averaging over repeats makes the threshold choice
    far less sensitive to one unlucky partition.
    """
    total = np.zeros(len(y), dtype=np.float64)
    count = np.zeros(len(y), dtype=np.int64)

    for repeat in range(n_repeats):
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                            random_state=seed + repeat)
            split = splitter.split(X, y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=seed + repeat)
            split = splitter.split(X, y)

        for train_idx, test_idx in split:
            estimator = build()
            estimator.fit(X[train_idx], y[train_idx])
            total[test_idx] += estimator.predict_proba(X[test_idx])[:, 1]
            count[test_idx] += 1

    probabilities = np.divide(total, count, out=np.full(len(y), np.nan), where=count > 0)
    return probabilities, count


@dataclass
class ThresholdChoice:
    threshold: float
    abstain_band: float
    #: Performance among the samples the system does NOT abstain on.
    balanced_accuracy_on_decided: float
    recall_worn_on_decided: float
    #: Fraction of samples sent to INCONCLUSIVE.
    abstention_rate: float
    target_recall_worn: float
    sweep: list[dict[str, float]] = field(default_factory=list)


def choose_threshold(
    probabilities: np.ndarray,
    y: np.ndarray,
    *,
    target_recall_worn: float = 0.80,
    abstain_bands: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20),
    max_abstention_rate: float = 0.30,
    min_band_gain: float = 0.02,
) -> ThresholdChoice:
    """Pick a decision threshold and abstention band from out-of-fold probabilities.

    Threshold
        Among the thresholds that reach ``target_recall_worn``, the one with the best
        balanced accuracy. The recall target encodes the asymmetry in the task -
        telling someone their worn tyre looks fine is a worse failure than sending
        them for an unnecessary check - but it is a *constraint*, not the objective.
        Simply taking the lowest qualifying threshold maximises recall by calling
        almost everything worn, which reaches the target while making the system
        useless; measured on the legacy subset that approach gave 0.98 recall at 0.566
        balanced accuracy. If no threshold reaches the target, the best available
        worn-recall is used and the shortfall is visible in the returned metrics.

    Abstention band
        The smallest band that improves balanced accuracy on the still-decided cases
        by at least ``min_band_gain``, without abstaining on more than
        ``max_abstention_rate`` of images. Abstention has to earn its keep: every
        abstention is a user who got no answer, so a band that merely hides
        borderline cases without improving the answers that remain is not worth
        having.
    """
    from sklearn.metrics import balanced_accuracy_score, recall_score

    valid = ~np.isnan(probabilities)
    p, truth = probabilities[valid], y[valid]

    sweep: list[dict[str, float]] = []
    for candidate in np.round(np.arange(0.10, 0.91, 0.01), 2):
        predicted = (p >= candidate).astype(int)
        sweep.append({
            "threshold": float(candidate),
            "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
            "recall_worn": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
            "recall_serviceable": float(recall_score(truth, predicted, pos_label=0, zero_division=0)),
        })

    reaching = [s for s in sweep if s["recall_worn"] >= target_recall_worn]
    if reaching:
        chosen = max(reaching, key=lambda s: (s["balanced_accuracy"], s["threshold"]))
    else:
        chosen = max(sweep, key=lambda s: (s["recall_worn"], s["balanced_accuracy"]))
    threshold = chosen["threshold"]

    baseline = float(balanced_accuracy_score(truth, (p >= threshold).astype(int)))
    baseline_recall = chosen["recall_worn"]

    best_band = 0.0
    best_stats = {
        "balanced_accuracy": baseline,
        "recall_worn": baseline_recall,
        "abstention_rate": 0.0,
    }
    for band in sorted(b for b in abstain_bands if b > 0):
        decided = (p < threshold - band) | (p > threshold + band)
        abstention_rate = float(1.0 - decided.mean())
        if abstention_rate > max_abstention_rate:
            continue
        if decided.sum() < 10 or len(np.unique(truth[decided])) < 2:
            continue

        predicted = (p[decided] >= threshold).astype(int)
        accuracy = float(balanced_accuracy_score(truth[decided], predicted))
        if accuracy >= baseline + min_band_gain and accuracy > best_stats["balanced_accuracy"]:
            best_band = float(band)
            best_stats = {
                "balanced_accuracy": accuracy,
                "recall_worn": float(recall_score(truth[decided], predicted,
                                                  pos_label=1, zero_division=0)),
                "abstention_rate": abstention_rate,
            }

    return ThresholdChoice(
        threshold=threshold,
        abstain_band=best_band,
        balanced_accuracy_on_decided=best_stats["balanced_accuracy"],
        recall_worn_on_decided=best_stats["recall_worn"],
        abstention_rate=best_stats["abstention_rate"],
        target_recall_worn=target_recall_worn,
        sweep=sweep,
    )


def calibration_metrics(probabilities: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Brier score and expected calibration error.

    A model can rank well and still be badly calibrated, and an abstention band is
    a statement about probability values rather than about ranking, so calibration
    has to be measured separately from accuracy.
    """
    valid = ~np.isnan(probabilities)
    p, truth = probabilities[valid], y[valid].astype(float)

    brier = float(np.mean((p - truth) ** 2))

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not in_bin.any():
            continue
        ece += in_bin.mean() * abs(truth[in_bin].mean() - p[in_bin].mean())

    return {"brier_score": brier, "expected_calibration_error": float(ece),
            "n_scored": int(valid.sum())}


def select_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 10,
    candidates: list[CandidateSpec] | None = None,
) -> tuple[list[EvaluationResult], list[dict[str, Any]]]:
    """Evaluate every candidate on identical folds and rank them."""
    candidates = candidates or candidate_models(seed)
    results: list[EvaluationResult] = []
    for spec in candidates:
        result = cross_validate(
            spec.build(), X, y, groups=groups, n_splits=n_splits,
            n_repeats=n_repeats, seed=seed, model_name=spec.name,
        )
        results.append(result)
        print("  " + result.headline(), flush=True)
    return results, compare_models(results)
