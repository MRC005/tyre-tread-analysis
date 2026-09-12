"""exp005 — Model selection, calibration and threshold choice on a given dataset.

This is the script that produces a deployable artifact. It is written to be run
against any dataset whose feature table has been built, so the same procedure applies
to the legacy data, the Mendeley data, or a combination, and the results are
comparable because the procedure does not change.

Everything is decided from out-of-fold predictions:

* which estimator, by paired comparison on identical folds;
* whether probabilities are trustworthy, by Brier score and calibration error;
* where the decision threshold sits, from a worn-recall target;
* how wide the abstention band is, by whether it actually improves accuracy on the
  cases the system still answers.

Usage:
    python experiments/exp005_model_selection.py --features outputs/features/<name>.parquet
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tyretread.config import CONFIG
from tyretread.data.hygiene import assign_groups, find_duplicates
from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.features.extract import feature_names
from tyretread.imaging.quality import QualityThresholds
from tyretread.models.artifact import ArtifactMetadata, ModelArtifact, save_artifact
from tyretread.models.evaluate import cross_validate
from tyretread.imaging.surface import SURFACE_FEATURES, fit_surface_detector
from tyretread.models.evidence_features import REPORTED_FEATURES, build_reference
from tyretread.models.train import (
    calibration_metrics, candidate_models, choose_threshold,
    out_of_fold_probabilities, select_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="parquet feature table from tyretread.data.build")
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--model-id", default=None,
                        help="persist the winning model under this artifact id")
    parser.add_argument("--target-recall", type=float, default=0.80,
                        help="minimum worn-tyre recall the threshold must achieve")
    parser.add_argument("--include-legacy-tsci", action="store_true",
                        help="add legacy TSCI as a candidate feature (see exp004)")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-groups", action="store_true",
                        help="disable group-aware folds, to measure their effect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pd.read_parquet(args.features)
    dataset_name = str(table["dataset"].iloc[0]) if "dataset" in table else args.features.stem

    usable = table[table["usable"]].reset_index(drop=True)
    print(f"dataset            : {dataset_name}")
    print(f"rows in table      : {len(table)}")
    print(f"passed quality gate: {len(usable)} ({len(usable)/max(1,len(table)):.0%})")
    print(f"label counts       : {usable['label'].value_counts().to_dict()}")

    if len(usable) < 60:
        raise SystemExit(
            f"only {len(usable)} usable images; too few for a meaningful comparison"
        )

    names = [n for n in feature_names() if n in usable.columns]
    # An absolute groove angle describes how the phone was held, not the tyre.
    names = [n for n in names if n != "orientation_dominant_deg"]
    if args.include_legacy_tsci and "legacy_tsci" in usable.columns:
        names.append("legacy_tsci")

    X = usable[names].astype(np.float64).to_numpy()
    y = (usable["label"] == "worn").astype(int).to_numpy()

    if np.isnan(X).any():
        bad = [names[i] for i in np.where(np.isnan(X).any(axis=0))[0]]
        raise SystemExit(f"non-finite values in features: {bad}")

    print(f"features           : {len(names)}")
    print(f"positive class     : worn (n={int(y.sum())}), negative: serviceable (n={int((1-y).sum())})")

    # ---- data hygiene ----------------------------------------------------
    paths = usable["path"].tolist()
    duplicates = find_duplicates(paths, usable["label"].tolist())
    print(f"\nexact duplicate groups        : {len(duplicates.exact_groups)}")
    print(f"contradictory label groups    : {len(duplicates.contradictory_groups)}")
    print(f"near-duplicate pairs          : {len(duplicates.near_pairs)}")

    groups = None if args.no_groups else assign_groups(paths)
    if groups is not None:
        print(f"distinct visual groups        : {len(set(groups.tolist()))} of {len(paths)}")

    # ---- model comparison ------------------------------------------------
    print("\nmodel comparison (identical folds, paired):")
    results, ranking = select_model(
        X, y, groups=groups, seed=CONFIG.random_seed, n_repeats=args.repeats,
    )

    print(f"\n{'model':22}{'bal_acc':>10}{'acc':>9}{'recall_worn':>13}{'roc_auc':>10}{'p vs best':>11}")
    for row in ranking:
        p = "-" if row["p_value"] is None else f"{row['p_value']:.4f}"
        auc = "-" if row["roc_auc"] is None else f"{row['roc_auc']:.3f}"
        print(f"{row['model']:22}{row['balanced_accuracy']:10.3f}{row['accuracy']:9.3f}"
              f"{row['recall_worn']:13.3f}{auc:>10}{p:>11}")

    winner_name = ranking[0]["model"]
    winner_spec = next(c for c in candidate_models(CONFIG.random_seed) if c.name == winner_name)
    print(f"\nwinner: {winner_name}")
    print(f"  rationale: {winner_spec.rationale}")

    # ---- majority-class floor -------------------------------------------
    from sklearn.dummy import DummyClassifier
    dummy = cross_validate(
        DummyClassifier(strategy="most_frequent"), X, y, groups=groups,
        n_repeats=2, seed=CONFIG.random_seed, model_name="dummy_majority",
    )
    dummy_acc = dummy.scores.summary()["accuracy"]["mean"]
    print(f"\nmajority-class floor: accuracy {dummy_acc:.3f}, balanced accuracy 0.500")

    # ---- calibration and thresholds -------------------------------------
    print("\nout-of-fold probabilities for the winner ...")
    probabilities, coverage = out_of_fold_probabilities(
        winner_spec.build, X, y, groups=groups, n_repeats=5, seed=CONFIG.random_seed,
    )
    calibration = calibration_metrics(probabilities, y)
    print(f"  Brier score               : {calibration['brier_score']:.4f}")
    print(f"  expected calibration error: {calibration['expected_calibration_error']:.4f}")

    choice = choose_threshold(probabilities, y, target_recall_worn=args.target_recall)
    print(f"\n  decision threshold        : {choice.threshold:.2f}")
    print(f"  abstention band           : +/-{choice.abstain_band:.2f}")
    print(f"  abstention rate           : {choice.abstention_rate:.1%}")
    print(f"  balanced accuracy (decided): {choice.balanced_accuracy_on_decided:.3f}")
    print(f"  worn recall (decided)      : {choice.recall_worn_on_decided:.3f}")
    if choice.recall_worn_on_decided < args.target_recall:
        print(f"  NOTE: the {args.target_recall:.0%} worn-recall target was NOT reached")

    # ---- final fit and artifact -----------------------------------------
    best = next(r for r in results if r.model_name == winner_name)
    metrics = {
        "ranking": ranking,
        "winner": best.summary(),
        "majority_class_accuracy": dummy_acc,
        "calibration": calibration,
        "threshold_choice": {k: v for k, v in asdict(choice).items() if k != "sweep"},
        "quality_gate_pass_rate": len(usable) / max(1, len(table)),
        "data_hygiene": duplicates.as_dict(),
    }

    model_id = args.model_id
    if model_id:
        estimator = winner_spec.build()
        estimator.fit(X, y)
        thresholds = QualityThresholds()

        # The reference distribution lets a served prediction place a measurement in
        # context. It must come from the training set the model actually saw.
        reference = build_reference(
            {n: usable[n].to_numpy() for n in REPORTED_FEATURES if n in usable.columns},
            y,
        )
        # Computed and reported, but excluded from the model: valid above the
        # oversampling floor (exp004), no measurable contribution (exp009).
        diagnostics = [n for n in ("legacy_tsci",)
                       if n in usable.columns and n not in names]

        # Tread-versus-sidewall detector, fitted on the hand-labelled annotation set
        # if one is present. Reported, never used to reject (see imaging/surface.py).
        surface_detector: dict = {}
        annotation_path = Path("data/annotations/surface_labels.json")
        if annotation_path.is_file() and all(f in usable.columns for f in SURFACE_FEATURES):
            import json as _json
            annotations = pd.DataFrame(_json.loads(annotation_path.read_text()))
            merged = annotations.merge(usable, on="path", how="inner")
            if len(merged) >= 50:
                surface_detector = fit_surface_detector(
                    merged[list(SURFACE_FEATURES)].astype(float).to_numpy(),
                    (merged["surface"] == "tread").astype(int).to_numpy(),
                )
                print(f"  surface detector fitted on {len(merged)} hand-labelled images")
        artifact = ModelArtifact(
            estimator=estimator,
            metadata=ArtifactMetadata(
                model_id=model_id,
                model_name=winner_name,
                feature_names=names,
                class_labels={"0": "serviceable", "1": "worn"},
                decision_threshold=choice.threshold,
                abstain_band=choice.abstain_band,
                training_dataset=f"{dataset_name} ({len(usable)} images passing the quality gate)",
                n_training_samples=len(usable),
                validation=best.validation,
                metrics=metrics,
                quality_thresholds=asdict(thresholds),
                feature_reference=reference,
                diagnostic_features=diagnostics,
                surface_detector=surface_detector,
                config_snapshot={
                    "scale": str(CONFIG.scale), "spectral": str(CONFIG.spectral),
                    "texture": str(CONFIG.texture), "roi": str(CONFIG.roi),
                },
                experiment_id=args.experiment_id,
                notes=(
                    "Binary visible-condition model: defect present versus tyre in good "
                    "condition. The user-facing verdict adds an abstention band around "
                    "the threshold; it is not a validated three-class model. No "
                    "tread-depth ground truth exists in any available dataset, so no "
                    "depth claim is made. exp008 shows this model does not transfer to "
                    "a dataset labelled for tread wear, so its validity is limited to "
                    "the domain it was trained on."
                ),
            ),
        )
        model_path, meta_path = save_artifact(artifact)
        print(f"\nartifact written: {model_path.name}, {meta_path.name}")

    experiment_id = args.experiment_id or f"exp005_model_selection_{dataset_name}"
    record_experiment(ExperimentRecord(
        experiment_id=experiment_id,
        title=f"Model selection on {dataset_name}",
        hypothesis=(
            "Among lightweight classifiers on engineered texture and frequency "
            "features, a regularised linear model will be competitive with or better "
            "than an RBF-SVM and tree ensembles, and its probabilities will calibrate "
            "well enough to support a principled abstention band."
        ),
        dataset=(
            f"{dataset_name}: {len(table)} images, {len(usable)} passing the quality "
            f"gate ({len(usable)/max(1,len(table)):.0%}). "
            f"worn={int(y.sum())}, serviceable={int((1-y).sum())}."
        ),
        method=(
            f"{len(names)} features from tyretread.features (scale-normalised ROI). "
            "Each candidate is a scaler-plus-estimator pipeline wrapped in Platt "
            "scaling, so scaling and calibration are fitted inside every fold. "
            "orientation_dominant_deg is excluded as an artefact of camera pose."
            + (" legacy TSCI included as a candidate feature." if args.include_legacy_tsci else "")
        ),
        validation=(
            f"{best.validation}. Model choice by paired t-test on identical folds. "
            "Threshold and abstention band chosen from averaged out-of-fold "
            "probabilities, never from in-sample predictions."
        ),
        metrics=metrics,
        tables={"threshold_sweep": choice.sweep},
        interpretation="",
        decision="",
        seed=CONFIG.random_seed,
        config={"features": names, "target_recall_worn": args.target_recall,
                "group_aware": groups is not None},
    ))
    print(f"recorded -> experiments/{experiment_id}.json")


if __name__ == "__main__":
    main()
