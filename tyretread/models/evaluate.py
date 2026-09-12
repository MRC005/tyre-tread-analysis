"""Model evaluation.

Design decisions, all of them consequences of what the audit found.

**Balanced accuracy is the headline metric.** The legacy dataset is 63/37, so plain
accuracy gives 63.4% for predicting the majority class and rewards a model that
ignores the minority. Worse, the audit showed that a classifier reading *only* the
JPEG header reached 68.6% accuracy but just 61.0% balanced accuracy - plain accuracy
made a confound look like skill.

**Repeated cross-validation, never a single split.** The originally published 74.80%
came from one seed; repeated folds put the same model at 73.7% +/- 4.6%. With 369
samples, a difference under about four points is not measurable.

**Group-aware folds by default.** Near-duplicate photographs must not straddle a
fold boundary. Measured as making no difference on the legacy dataset, which is a
reason to keep verifying it, not a reason to stop doing it.

**Paired comparison against identical folds.** Two models are only comparable when
scored on the same partitions, so model selection uses a paired t-test rather than
two independent means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

__all__ = ["FoldScores", "EvaluationResult", "cross_validate", "compare_models"]


@dataclass
class FoldScores:
    accuracy: list[float] = field(default_factory=list)
    balanced_accuracy: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    roc_auc: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, values in self.__dict__.items():
            clean = [v for v in values if v is not None and not np.isnan(v)]
            if clean:
                out[name] = {
                    "mean": float(np.mean(clean)),
                    "std": float(np.std(clean)),
                    "n_folds": len(clean),
                }
        return out


@dataclass
class EvaluationResult:
    model_name: str
    scores: FoldScores
    confusion: np.ndarray
    per_fold_balanced_accuracy: np.ndarray
    n_samples: int
    n_features: int
    validation: str

    def summary(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "validation": self.validation,
            "metrics": self.scores.summary(),
            "confusion_matrix": self.confusion.tolist(),
        }

    def headline(self) -> str:
        s = self.scores.summary()
        ba = s["balanced_accuracy"]
        ac = s["accuracy"]
        return (f"{self.model_name:28} bal_acc={ba['mean']:.3f}+/-{ba['std']:.3f}  "
                f"acc={ac['mean']:.3f}+/-{ac['std']:.3f}")


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    model_name: str = "model",
) -> EvaluationResult:
    """Repeated, optionally group-aware stratified cross-validation.

    ``y`` must be 0/1 with 1 meaning the positive (worn) class. Recall on the worn
    class is the metric that matters operationally: missing a worn tyre is the
    expensive error, not flagging a serviceable one.
    """
    scores = FoldScores()
    confusion = np.zeros((2, 2), dtype=int)
    per_fold: list[float] = []

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
            estimator = clone(model)
            estimator.fit(X[train_idx], y[train_idx])
            pred = estimator.predict(X[test_idx])
            truth = y[test_idx]

            scores.accuracy.append(float(accuracy_score(truth, pred)))
            ba = float(balanced_accuracy_score(truth, pred))
            scores.balanced_accuracy.append(ba)
            per_fold.append(ba)
            scores.precision.append(float(precision_score(truth, pred, zero_division=0)))
            scores.recall.append(float(recall_score(truth, pred, zero_division=0)))
            scores.f1.append(float(f1_score(truth, pred, zero_division=0)))

            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X[test_idx])[:, 1]
                if len(np.unique(truth)) > 1:
                    scores.roc_auc.append(float(roc_auc_score(truth, proba)))

            confusion += confusion_matrix(truth, pred, labels=[0, 1])

    validation = (
        f"{n_splits}-fold x {n_repeats} repeats, "
        f"{'group-aware (StratifiedGroupKFold)' if groups is not None else 'StratifiedKFold'}, "
        f"seed {seed}"
    )
    return EvaluationResult(
        model_name=model_name, scores=scores, confusion=confusion,
        per_fold_balanced_accuracy=np.array(per_fold),
        n_samples=len(y), n_features=X.shape[1], validation=validation,
    )


def compare_models(results: Sequence[EvaluationResult]) -> list[dict[str, object]]:
    """Paired comparison of every model against the best one.

    Requires that all results came from identical folds, which is why every caller
    passes the same seed, splitter and group vector.
    """
    ranked = sorted(results, key=lambda r: -float(np.mean(r.per_fold_balanced_accuracy)))
    best = ranked[0]
    rows: list[dict[str, object]] = []
    for result in ranked:
        row: dict[str, object] = {
            "model": result.model_name,
            "balanced_accuracy": float(np.mean(result.per_fold_balanced_accuracy)),
            "balanced_accuracy_std": float(np.std(result.per_fold_balanced_accuracy)),
            "accuracy": result.scores.summary()["accuracy"]["mean"],
            "recall_worn": result.scores.summary()["recall"]["mean"],
            "f1_worn": result.scores.summary()["f1"]["mean"],
        }
        auc = result.scores.summary().get("roc_auc")
        row["roc_auc"] = auc["mean"] if auc else None

        if result is best:
            row.update({"delta_vs_best": 0.0, "p_value": None, "significant": None})
        else:
            a = result.per_fold_balanced_accuracy
            b = best.per_fold_balanced_accuracy
            if len(a) == len(b):
                test = stats.ttest_rel(a, b)
                row.update({
                    "delta_vs_best": float(np.mean(a) - np.mean(b)),
                    "p_value": float(test.pvalue),
                    "significant": bool(test.pvalue < 0.05),
                })
            else:
                row.update({"delta_vs_best": float(np.mean(a) - np.mean(b)),
                            "p_value": None, "significant": None})
        rows.append(row)
    return rows
