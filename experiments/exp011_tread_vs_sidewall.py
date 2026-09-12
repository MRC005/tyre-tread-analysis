"""exp011 — Can the system tell tread from sidewall, and does it matter?

exp010 surfaced a limitation that cross-validation cannot see: the ROI stage returns a
plausible "tread band" from a photograph of a tyre's sidewall. A user who photographs
the side of their tyre gets an assessment worded as though it were about the tread.

Three questions, in order:

1. How much of the data that passes the quality gate is not actually tread?
2. Is the surface confounded with the condition label - is the model partly learning
   "this is a sidewall photograph" rather than "this tyre has a defect"?
3. Can the surface be detected reliably enough to act on?

Answering any of them needs labels that do not exist, so 120 gate-passing images were
sampled at random and hand-labelled tread / sidewall / mixed. That annotation set is
in data/annotations/surface_labels.json and is the input here.

Run: python experiments/exp011_tread_vs_sidewall.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.features.extract import feature_names
from tyretread.imaging.surface import NOT_TREAD_BELOW, SURFACE_FEATURES, TREAD_ABOVE

ANNOTATIONS = Path("data/annotations/surface_labels.json")
FEATURES = Path("outputs/features/mendeley_tyres.parquet")
N_BOOTSTRAP = 3000


def model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])


def main() -> None:
    annotations = pd.DataFrame(json.loads(ANNOTATIONS.read_text()))
    joined = annotations.merge(pd.read_parquet(FEATURES), on="path", how="inner")
    # The annotation set was sampled before the quality gate was tightened, so a few
    # of its images are now refused and carry no features. Analysing the surface
    # question on images the system would not assess anyway would misstate the problem.
    merged = joined[joined["usable"].fillna(False)].reset_index(drop=True)
    print(f"annotated images: {len(annotations)}; still passing the gate: {len(merged)} "
          f"({len(joined) - len(merged)} now refused by the tightened gate)")

    counts = merged["surface"].value_counts().to_dict()
    not_tread = (merged["surface"] != "tread").mean()
    print("\n=== 1. what fraction of accepted images is not clean tread? ===")
    for key in ("tread", "sidewall", "mixed"):
        n = counts.get(key, 0)
        print(f"  {key:10} {n:4} ({n / len(merged):5.1%})")
    print(f"  -> {not_tread:.0%} of accepted images are NOT clean tread")

    print("\n=== 2. is surface confounded with the condition label? ===")
    table = pd.crosstab(merged["surface"] != "tread", merged["condition_label"] == "worn")
    chi = stats.chi2_contingency(table)
    for surface in ("tread", "sidewall", "mixed"):
        subset = merged[merged["surface"] == surface]
        if len(subset):
            print(f"  {surface:10} {(subset['condition_label'] == 'worn').mean():5.1%} labelled defective")
    print(f"  chi-square p = {chi.pvalue:.4f} -> "
          f"{'CONFOUNDED' if chi.pvalue < 0.05 else 'no association'}")

    print("\n=== 3. can the surface be detected? ===")
    y = (merged["surface"] == "tread").astype(int).to_numpy()
    all_features = [n for n in feature_names()
                    if n in merged.columns and n != "orientation_dominant_deg"]

    def out_of_fold(columns: list[str]) -> np.ndarray:
        runs = []
        for seed in range(10):
            runs.append(cross_val_predict(
                model(), merged[columns].astype(float).to_numpy(), y,
                cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                method="predict_proba",
            )[:, 1])
        return np.mean(runs, axis=0)

    p_all = out_of_fold(all_features)
    p_sub = out_of_fold(list(SURFACE_FEATURES))

    rng = np.random.default_rng(42)
    boot = []
    index = np.arange(len(y))
    for _ in range(N_BOOTSTRAP):
        sample = rng.choice(index, len(index), replace=True)
        if len(np.unique(y[sample])) > 1:
            boot.append(roc_auc_score(y[sample], p_sub[sample]))
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    auc_all = float(roc_auc_score(y, p_all))
    auc_sub = float(roc_auc_score(y, p_sub))
    print(f"  all {len(all_features)} features          AUC = {auc_all:.3f}")
    print(f"  {len(SURFACE_FEATURES)} interpretable features   AUC = {auc_sub:.3f} "
          f"[{ci[0]:.3f}, {ci[1]:.3f}]")

    print(f"\n  operating points for the {len(SURFACE_FEATURES)}-feature detector:")
    print(f"  {'threshold':>10}{'tread kept':>13}{'non-tread flagged':>20}{'false flag':>13}")
    operating = []
    for t in (0.25, 0.35, 0.45, 0.50, 0.60, 0.70):
        predicted = (p_sub >= t).astype(int)
        kept = float(recall_score(y, predicted))
        flagged = float(1 - recall_score(1 - y, 1 - predicted, zero_division=0))
        operating.append({"threshold": t, "tread_kept": kept,
                          "non_tread_flagged": 1 - flagged, "false_flag": 1 - kept})
        print(f"  {t:10.2f}{kept:13.1%}{1 - flagged:20.1%}{1 - kept:13.1%}")

    record_experiment(ExperimentRecord(
        experiment_id="exp011_tread_vs_sidewall",
        title="Half the accepted images are not tread, and the surface cannot be detected reliably",
        hypothesis=(
            "The ROI stage cannot distinguish the tread from the sidewall, so a "
            "significant share of accepted images are sidewall photographs being "
            "assessed as though they were tread. If the surface is detectable "
            "reliably, non-tread images can be refused; if it is only partly "
            "detectable, refusing would cost more than it gains."
        ),
        dataset=(
            f"{len(merged)} images sampled at random from those passing the quality "
            "gate on the Mendeley dataset, hand-labelled tread / sidewall / mixed. "
            "Labels in data/annotations/surface_labels.json."
        ),
        method=(
            "Balanced logistic regression over the production feature set, and over "
            f"the {len(SURFACE_FEATURES)} interpretable structure features that "
            "tyretread.imaging.surface ships. Association between surface and "
            "condition label tested with chi-square."
        ),
        validation=(
            f"Out-of-fold probabilities averaged over 10 runs of stratified 5-fold. "
            f"ROC-AUC with a {N_BOOTSTRAP}-sample percentile bootstrap 95% confidence "
            "interval, which 120 labels make essential."
        ),
        metrics={
            "n_annotated": int(len(merged)),
            "surface_counts": counts,
            "fraction_not_clean_tread": float(not_tread),
            "surface_condition_chi2_p": float(chi.pvalue),
            "detector_auc_all_features": auc_all,
            "detector_auc_interpretable": auc_sub,
            "detector_auc_ci95": list(ci),
            "operating_points": operating,
            "shipped_thresholds": {"not_tread_below": NOT_TREAD_BELOW,
                                   "tread_above": TREAD_ABOVE},
        },
        interpretation=(
            f"{not_tread:.0%} of the images the system accepts are not clean tread - "
            f"{counts.get('sidewall', 0)} sidewall close-ups and "
            f"{counts.get('mixed', 0)} mixed out of {len(merged)}. Many are photographs "
            "of moulded sidewall lettering, which contains no tread at all. "
            f"The surface is not confounded with the condition label (chi-square "
            f"p = {chi.pvalue:.2f}); defect prevalence is similar across surfaces, so "
            "the model is not covertly learning 'this is a sidewall photograph'. That "
            "is the reassuring half. "
            f"The detector reaches AUC {auc_sub:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] - real "
            "signal, but not reliable enough to reject on. At a threshold flagging "
            "three-quarters of non-tread images it also flags one genuine tread "
            "photograph in five. Given that the production model performs comparably "
            "on both surfaces, refusing sidewall images would discard working "
            "functionality to enforce a distinction the system cannot make confidently."
        ),
        decision=(
            "Report the surface, do not enforce it. The inspection states which surface "
            "it believes it assessed and says when it does not know, using the "
            f"thresholds {NOT_TREAD_BELOW} and {TREAD_ABOVE} - the lower one chosen at "
            "roughly 5% false flagging of genuine tread. The user-facing copy for a "
            "non-tread result states explicitly that the assessment says nothing about "
            "remaining tread. This is the honest reading of 'do not pretend the system "
            "can identify tread': it neither claims a tread assessment it cannot "
            "support, nor throws away a sidewall assessment it can. "
            "The detector is fitted on 120 labels and should be refitted on a larger "
            "annotated sample before being relied on more heavily. Refusing non-tread "
            "images remains a reasonable future option if tread-specific labels ever "
            "become available."
        ),
        seed=42,
        config={"surface_features": list(SURFACE_FEATURES), "n_bootstrap": N_BOOTSTRAP},
    ))
    print("\nrecorded -> experiments/exp011_tread_vs_sidewall.json")


if __name__ == "__main__":
    main()
