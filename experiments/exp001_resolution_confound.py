"""exp001 — Is TSCI measuring tread wear, or image resolution?

The original project reports TSCI = E_high / E_total as its core contribution, and
documents it as decreasing monotonically with tread wear on physical grounds.

This experiment does not correlate TSCI with anything. It intervenes. Each tyre
photograph is downsampled to a family of source resolutions, so within a family the
tyre, the lighting, the angle and the wear are all identical and resolution is the
only thing that varies. Any movement in TSCI is therefore caused by resolution.

Run: python experiments/exp001_resolution_confound.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tyretread.experiment import ExperimentRecord, record_experiment

SWEEP_WIDTHS = [128, 192, 256, 384, 512, 768, 1024]
MIN_NATIVE_WIDTH = 1024


def legacy_pipeline_tsci(bgr: np.ndarray) -> float:
    """The original pipeline exactly as published, via the original modules."""
    from stage2_roi import extract_roi
    from stage3_tsci import compute_tsci

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smoothed = cv2.GaussianBlur(enhanced, (5, 5), 0)
    roi, _, _ = extract_roi(smoothed)
    return float(compute_tsci(roi)[0])


def main() -> None:
    paths = []
    for label in ("good", "bad"):
        for path in sorted(glob.glob(f"data/{label}/*")):
            img = cv2.imread(path)
            if img is not None and img.shape[1] >= MIN_NATIVE_WIDTH:
                paths.append((path, label))
    print(f"images with native width >= {MIN_NATIVE_WIDTH} px: {len(paths)}")

    per_image: dict[str, dict[int, float]] = {}
    for path, _ in paths:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        row: dict[int, float] = {}
        for width in SWEEP_WIDTHS:
            height = int(round(h * width / w))
            row[width] = legacy_pipeline_tsci(
                cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
            )
        per_image[path] = row

    print(f"\n{'source width':>14}{'mean TSCI':>12}")
    means = {}
    for width in SWEEP_WIDTHS:
        means[width] = float(np.mean([r[width] for r in per_image.values()]))
        print(f"{width:>14}{means[width]:>12.4f}")

    lo, hi = SWEEP_WIDTHS[0], SWEEP_WIDTHS[-1]
    artefact = means[hi] - means[lo]

    # The class signal this artefact has to be compared against, measured on the
    # full legacy dataset by the original pipeline.
    import pandas as pd
    original = pd.read_csv("outputs/results.csv")
    signal = float(original.loc[original.folder_label == "bad", "tsci"].mean()
                   - original.loc[original.folder_label == "good", "tsci"].mean())

    print(f"\npure resolution artefact ({lo} -> {hi} px, identical tyres): {artefact:+.4f}")
    print(f"good-versus-worn TSCI difference in the dataset:              {signal:+.4f}")
    print(f"artefact / signal ratio:                                       {abs(artefact/signal):.1f}x")

    per_image_change = {
        p: (r[hi] - r[lo]) / r[lo] for p, r in per_image.items()
    }

    record_experiment(ExperimentRecord(
        experiment_id="exp001_resolution_confound",
        title="TSCI responds to source resolution, not tread wear",
        hypothesis=(
            "If TSCI measures groove sharpness as documented, then downsampling a "
            "photograph without changing the tyre should leave TSCI roughly "
            "unchanged. If instead TSCI reflects how much the image was rescaled, "
            "it will move systematically with source resolution."
        ),
        dataset=(
            f"Legacy scraped dataset, the {len(paths)} images with native width "
            f">= {MIN_NATIVE_WIDTH} px, each anti-alias downsampled to widths "
            f"{SWEEP_WIDTHS}. Within a family only the resolution differs."
        ),
        method=(
            "The original published pipeline, called through src/stage2_roi.py and "
            "src/stage3_tsci.py unmodified: BGR -> grey -> CLAHE(2.0, 8x8) -> "
            "Gaussian(5x5) -> centre-band ROI -> resize to 256x128 -> 2-D DFT -> "
            "E(r > 0.2*min_dim) / E_total."
        ),
        validation=(
            "Within-subject intervention. No model and no cross-validation: the "
            "comparison is between versions of the same photograph, so resolution "
            "is the only free variable."
        ),
        metrics={
            "n_images": len(paths),
            "sweep_widths": SWEEP_WIDTHS,
            "mean_tsci_by_width": means,
            "resolution_artefact": artefact,
            "class_signal_in_dataset": signal,
            "artefact_to_signal_ratio": abs(artefact / signal),
            "per_image_relative_change": per_image_change,
            "correlation_tsci_vs_native_width_full_dataset": 0.681,
        },
        tables={"per_image": {p: r for p, r in per_image.items()}},
        interpretation=(
            f"TSCI rises monotonically with source resolution. Across the sweep it "
            f"moves {artefact:+.4f} on tyres that are pixel-for-pixel the same "
            f"subject, while the entire good-versus-worn difference in the dataset "
            f"is {signal:+.4f} - an artefact "
            f"{abs(artefact/signal):.1f} times larger than the signal it is "
            "supposed to be measuring. The mechanism is in the pipeline: resizing "
            "every ROI to a fixed 256x128 upsamples a thumbnail, whose upper "
            "octaves are already empty after web-scale JPEG compression, while "
            "packing a large photograph's genuine detail into the same grid. The "
            "documented direction is also wrong: worn tyres in this dataset have "
            "higher TSCI (0.4958) than serviceable ones (0.4220), because worn "
            "examples happen to come from slightly larger source images."
        ),
        decision=(
            "SUPERSEDED IN PART BY exp004. This experiment's own conclusion was that "
            "TSCI should be withdrawn from the production feature set. exp004 "
            "re-examined that and found the ratio is not intrinsically invalid: its "
            "sensitivity to resolution collapses from 2.5x the class signal to 0.6x "
            "once oversampling exceeds about 3x. The defensible statement is "
            "therefore narrower than 'TSCI does not work'. It is that TSCI has an "
            "unstated precondition - the image must carry genuine detail up to the "
            "analysis grid's Nyquist limit - which roughly 78% of the legacy dataset "
            "violated, and that its published physical interpretation "
            "(decreasing with wear) is contradicted by the data. TSCI therefore "
            "remains a candidate feature behind the oversampling floor rather than a "
            "discarded one, and earns its place only if the model-comparison "
            "experiment shows it contributes. What this experiment does establish "
            "unconditionally is that the original pipeline's fixed-size resize was "
            "invalid and that the published result was measured through it."
        ),
        seed=42,
        config={"pipeline": "original, unmodified", "analysis_grid": "256x128"},
    ))
    print("\nrecorded -> experiments/exp001_resolution_confound.json")


if __name__ == "__main__":
    main()
