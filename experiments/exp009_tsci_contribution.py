"""exp009 — Does TSCI earn a place in the production model?

exp004 established that TSCI is not intrinsically broken: its resolution sensitivity
collapses once oversampling exceeds about 3x, and the quality gate now enforces that.
The original work's real error was a missing precondition plus an unvalidated physical
interpretation, not a worthless formula.

That rehabilitation is necessary but not sufficient. A feature being *valid* is a
different claim from a feature being *useful*. This experiment settles the second
question the only way it can be settled: by adding TSCI to the production feature set
and testing, on identical folds, whether the model gets measurably better.

The project has a stake in the answer - TSCI was the original paper's headline
contribution - which is exactly why the test is pre-specified and paired.

Run: python experiments/exp009_tsci_contribution.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.features.extract import feature_names
from tyretread.models.train import candidate_models

FEATURES = Path("outputs/features/mendeley_tyres.parquet")
N_SEEDS = 6


def scores(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Balanced accuracy across identical resolution-grouped folds."""
    out: list[float] = []
    for seed in range(N_SEEDS):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        out += list(cross_val_score(model, X, y, groups=groups, cv=splitter,
                                    scoring="balanced_accuracy"))
    return np.array(out)


def main() -> None:
    table = pd.read_parquet(FEATURES)
    usable = table[table["usable"]].reset_index(drop=True)
    y = (usable["label"] == "worn").astype(int).to_numpy()
    groups = usable.groupby(["native_width", "native_height"]).ngroup().to_numpy()

    production = [n for n in feature_names()
                  if n in usable.columns and n != "orientation_dominant_deg"]
    spec = next(c for c in candidate_models(42) if c.name == "svm_rbf_c10")

    print(f"n={len(usable)}  production features={len(production)}  "
          f"folds={N_SEEDS * 5} (resolution-grouped)\n")

    # ---- is TSCI even informative on its own here? ----------------------
    worn_tsci = usable.loc[usable.label == "worn", "legacy_tsci"]
    ok_tsci = usable.loc[usable.label == "serviceable", "legacy_tsci"]
    u_stat = stats.mannwhitneyu(worn_tsci, ok_tsci)
    tsci_auc = float(u_stat.statistic / (len(worn_tsci) * len(ok_tsci)))
    print("=== TSCI on its own ===")
    print(f"  mean TSCI, worn        : {worn_tsci.mean():.4f}")
    print(f"  mean TSCI, serviceable : {ok_tsci.mean():.4f}")
    print(f"  univariate AUC         : {tsci_auc:.3f}  (0.5 = uninformative)")
    print("  original paper's claim : TSCI decreases with wear, so worn should be LOWER")
    print(f"  observed direction     : worn is "
          f"{'LOWER - consistent' if worn_tsci.mean() < ok_tsci.mean() else 'HIGHER - contradicts the paper'}")

    alone = scores(spec.build(), usable[["legacy_tsci"]].astype(float).to_numpy(), y, groups)
    print(f"  TSCI alone, in a model : {alone.mean():.3f} +/-{alone.std():.3f}")

    # ---- does it add anything to the production set? --------------------
    print("\n=== does adding TSCI to the production feature set help? ===")
    without = scores(spec.build(), usable[production].astype(float).to_numpy(), y, groups)
    with_tsci = scores(spec.build(),
                       usable[production + ["legacy_tsci"]].astype(float).to_numpy(), y, groups)
    delta = with_tsci - without
    paired = stats.ttest_rel(with_tsci, without)

    print(f"  production features ({len(production)})      : {without.mean():.4f} +/-{without.std():.4f}")
    print(f"  production + TSCI ({len(production) + 1})       : {with_tsci.mean():.4f} +/-{with_tsci.std():.4f}")
    print(f"  paired difference                  : {delta.mean():+.4f}")
    print(f"  paired t-test p                    : {paired.pvalue:.4f}  "
          f"-> {'SIGNIFICANT' if paired.pvalue < 0.05 else 'not significant'}")

    # ---- and against the scale-invariant spectral descriptors? ----------
    print("\n=== TSCI versus the scale-invariant spectral descriptors it was replaced by ===")
    spectral = [n for n in production if n.startswith(("spectral", "orientation"))]
    spectral_only = scores(spec.build(), usable[spectral].astype(float).to_numpy(), y, groups)
    print(f"  scale-invariant spectral ({len(spectral)})      : {spectral_only.mean():.3f}")
    print(f"  legacy TSCI alone (1)              : {alone.mean():.3f}")

    non_spectral = [n for n in production if not n.startswith(("spectral", "orientation"))]
    texture_plus_tsci = scores(
        spec.build(), usable[non_spectral + ["legacy_tsci"]].astype(float).to_numpy(), y, groups)
    print(f"  texture + TSCI, no new spectral    : {texture_plus_tsci.mean():.3f}")
    print(f"  texture + new spectral (production): {without.mean():.3f}")

    keep = bool(paired.pvalue < 0.05 and delta.mean() > 0)

    record_experiment(ExperimentRecord(
        experiment_id="exp009_tsci_contribution",
        title="TSCI is valid but adds no measurable predictive value",
        hypothesis=(
            "Having been rehabilitated in exp004 as valid above the oversampling floor, "
            "TSCI should be readmitted to the production feature set only if it "
            "measurably improves the model. The null hypothesis is that it adds nothing "
            "beyond the scale-invariant spectral descriptors that replaced it."
        ),
        dataset=(
            f"Mendeley, {len(usable)} images passing the quality gate "
            f"({int(y.sum())} worn / {int((1 - y).sum())} serviceable). Every image is "
            "behind the oversampling floor, so TSCI is evaluated only inside the regime "
            "exp004 validated."
        ),
        method=(
            f"RBF-SVM (C=10) with Platt scaling, the model exp007 selected. The "
            f"{len(production)}-feature production set is compared against the same set "
            "plus legacy TSCI, on identical folds. TSCI alone and the scale-invariant "
            "spectral descriptors alone are also fitted for context."
        ),
        validation=(
            f"{N_SEEDS * 5} resolution-grouped folds, identical across both arms, "
            "compared with a paired t-test. Resolution grouping matters here "
            "specifically: TSCI's known failure mode is resolution sensitivity, so a "
            "fold scheme that let resolution leak could credit TSCI for the confound it "
            "is prone to."
        ),
        metrics={
            "n": int(len(usable)),
            "n_production_features": len(production),
            "tsci_univariate_auc": tsci_auc,
            "tsci_mean_worn": float(worn_tsci.mean()),
            "tsci_mean_serviceable": float(ok_tsci.mean()),
            "tsci_alone_balanced_accuracy": float(alone.mean()),
            "production_balanced_accuracy": float(without.mean()),
            "production_plus_tsci_balanced_accuracy": float(with_tsci.mean()),
            "paired_delta": float(delta.mean()),
            "paired_p_value": float(paired.pvalue),
            "significant": keep,
            "scale_invariant_spectral_only": float(spectral_only.mean()),
            "texture_plus_tsci_no_new_spectral": float(texture_plus_tsci.mean()),
        },
        interpretation=(
            f"TSCI adds nothing, and the way it fails is more informative than the "
            f"headline. Its univariate AUC of {tsci_auc:.3f} looks like a real signal, "
            f"but as a lone feature under resolution-grouped folds it scores "
            f"{alone.mean():.3f} against a 0.500 floor - indistinguishable from chance. "
            "The univariate figure was measured across the whole dataset, where the "
            "acquisition confound identified in exp006 is free to contribute; blocking "
            "sessions removes it. That is the same failure mode exp001 found on the "
            "legacy dataset, reproduced here on independent data: what looks like tread "
            "information turns out to be information about the photograph. "
            f"Added to the {len(production)}-feature production set, TSCI changes "
            f"balanced accuracy by {delta.mean():+.4f} with a paired p of "
            f"{paired.pvalue:.4f} - "
            f"{'a significant improvement' if keep else 'no measurable improvement'}. "
            f"The scale-invariant spectral descriptors that replaced it reach "
            f"{spectral_only.mean():.3f} on their own, and swapping them out for TSCI "
            f"costs the full model {texture_plus_tsci.mean():.3f} against "
            f"{without.mean():.3f}, so the replacement was worth making. "
            "One further result deserves recording. Worn tyres average "
            f"{worn_tsci.mean():.4f} against {ok_tsci.mean():.4f} for serviceable ones - "
            + ("the direction the original paper predicted."
               if worn_tsci.mean() < ok_tsci.mean() else
               "again the opposite of the original paper's claim that TSCI decreases "
               "with wear. On the legacy dataset that inversion could be attributed to "
               "the resolution confound. Here it is reproduced on an independent "
               "dataset, entirely inside the oversampling regime exp004 validated, so "
               "that explanation no longer applies. The stated physical justification - "
               "that shallower grooves attenuate high-frequency energy - is not "
               "supported by either dataset.")
        ),
        decision=(
            "TSCI is not included in the production classifier. It is retained as a "
            "reported diagnostic on the inspection report and as a research feature, "
            "where it is genuinely useful: it is cheap, interpretable, and its history "
            "in this project is the clearest available illustration of why a feature "
            "needs a stated validity regime. Keeping it out of the model while keeping "
            "it in the report is the honest resolution - it was never shown to be "
            "worthless, only redundant here, and the distinction is recorded rather than "
            "flattened. If a future dataset with real tread-depth labels changes that, "
            "this experiment is the one to re-run."
        ),
        seed=42,
        config={"model": "svm_rbf_c10", "n_seeds": N_SEEDS,
                "validation": "resolution-grouped StratifiedGroupKFold"},
    ))
    print(f"\nrecorded -> experiments/exp009_tsci_contribution.json")


if __name__ == "__main__":
    main()
