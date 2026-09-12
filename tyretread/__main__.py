"""Command-line entry point.

``python -m tyretread inspect photo.jpg``

Exists so the pipeline can be exercised without starting a server, and so a
developer's local check goes through exactly the same ``inspect_image`` call the API
makes. A CLI that reimplemented the pipeline would defeat the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ARTIFACT_DIR
from .imaging.io import load_image
from .models.artifact import list_artifacts, load_artifact


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        artifact = load_artifact(args.model, ARTIFACT_DIR)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        available = list_artifacts(ARTIFACT_DIR)
        print(f"available models: {available or 'none'}", file=sys.stderr)
        return 2

    from .inspect import inspect_image

    try:
        loaded = load_image(args.image)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    inspection = inspect_image(loaded.bgr, artifact)

    if args.json:
        print(json.dumps(inspection.as_dict(include_features=args.features), indent=2))
        return 0

    result = inspection.decision
    print()
    print(f"  {result.copy.headline.upper()}")
    print(f"  {'-' * len(result.copy.headline)}")
    print(f"  {result.copy.detail}")
    print()
    if result.probability_defect is not None:
        print(f"  probability of a visible defect : {result.probability_defect:.1%}")
        print(f"  confidence                      : {result.confidence:.1%}")
    print(f"  image quality                   : {inspection.quality.summary}")
    if inspection.surface is not None:
        print(f"  surface assessed                : {inspection.surface.surface.value} "
              f"(tread evidence {inspection.surface.tread_probability:.0%})")
        import textwrap
        for line in textwrap.wrap(inspection.surface.note, 74):
            print(f"    {line}")

    if inspection.quality.issues:
        print()
        print("  This image could not be assessed:")
        for issue, advice in zip(inspection.quality.issues, inspection.quality.advice):
            print(f"    - {issue.value}: {advice}")

    if inspection.quality.warnings:
        print()
        print("  Caveats:")
        for warning in inspection.quality.warnings:
            print(f"    - {warning.value}")

    if inspection.explanation and inspection.explanation.exact and inspection.explanation.reasons:
        print()
        print("  Why this result (model interpretation):")
        for reason in inspection.explanation.reasons:
            print(f"    - {reason}")
    elif inspection.explanation is not None:
        print()
        print("  Model interpretation: not available.")
        print("    The model in use is non-linear, so its decision cannot be broken")
        print("    down per measurement. No breakdown is guessed at. What was actually")
        print("    measured is shown below.")

    if inspection.evidence is not None and inspection.evidence.measurements:
        print()
        print("  Measured evidence (this photograph, against the training range):")
        for item in inspection.evidence.measurements:
            resembles = f"  ~ {item.resembles}" if item.resembles else ""
            print(f"    {item.description[:46]:48} {item.value:10.4f}  "
                  f"p{item.percentile:5.1f}  {item.band:9}{resembles}")
        if inspection.evidence.agreement:
            print()
            import textwrap
            for line in textwrap.wrap(inspection.evidence.agreement, 74):
                print(f"    {line}")

        if inspection.evidence.diagnostics:
            print()
            print("  Diagnostics (computed and reported, not used by the model):")
            for name, value in inspection.evidence.diagnostics.items():
                print(f"    {name:28} {value:10.4f}")
    elif inspection.measurements:
        print()
        print("  Measurements:")
        for name, value in inspection.measurements.items():
            print(f"    {name:28} {value:10.4f}")

    print()
    print(f"  Recommendation: {result.copy.recommendation}")
    print()
    print(f"  model: {inspection.model_id}")
    print()
    return 0


def _cmd_models(_: argparse.Namespace) -> int:
    available = list_artifacts(ARTIFACT_DIR)
    if not available:
        print(f"no model artifacts in {ARTIFACT_DIR}")
        return 1
    for model_id in available:
        artifact = load_artifact(model_id, ARTIFACT_DIR)
        meta = artifact.metadata
        print(f"{model_id}")
        print(f"  estimator      {meta.model_name}")
        print(f"  trained on     {meta.training_dataset} (n={meta.n_training_samples})")
        print(f"  validation     {meta.validation}")
        print(f"  threshold      {meta.decision_threshold} +/- {meta.abstain_band}")
        print(f"  features       {len(meta.feature_names)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tyretread",
        description="Assistive tyre tread condition screening from photographs.",
        epilog=(
            "This is a screening aid. It cannot measure tread depth and is not a "
            "certified inspection."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="assess one photograph")
    inspect_parser.add_argument("image", type=Path)
    inspect_parser.add_argument("--model", default="current", help="artifact id to use")
    inspect_parser.add_argument("--json", action="store_true", help="machine-readable output")
    inspect_parser.add_argument("--features", action="store_true",
                                help="include every raw feature value (implies --json)")
    inspect_parser.set_defaults(func=_cmd_inspect)

    models_parser = subparsers.add_parser("models", help="list available model artifacts")
    models_parser.set_defaults(func=_cmd_models)

    args = parser.parse_args(argv)
    if getattr(args, "features", False):
        args.json = True
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
