"""exp003 — Choosing the analysis grid and the oversampling floor.

exp001 showed that a fixed-size resize makes frequency features a function of
source resolution. exp002 showed the artefact largely disappears once every image is
resampled *downwards* onto a common grid from at least about 3x oversampling.

Those two results imply a production rule: refuse any image that cannot fill the
analysis grid with genuine detail. This experiment asks what that rule costs. For
each candidate grid and each candidate oversampling floor it measures how much of a
dataset survives, and what happens to the class balance.

Run: python experiments/exp003_oversampling_floor.py
"""

from __future__ import annotations

import collections
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tyretread.config import CONFIG, ScaleConfig
from tyretread.data.build import build_feature_table
from tyretread.data.datasets import LEGACY_SCRAPED
from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.imaging.quality import QualityThresholds

GRIDS = [(96, 48), (128, 64), (192, 96), (256, 128)]
FLOORS = [1.0, 2.0, 3.0]


def main() -> None:
    rows = []
    print(f"{'grid':>10}{'floor':>7}{'kept':>7}{'serviceable':>13}{'worn':>7}{'% serviceable':>15}")
    print("-" * 60)
    for width, height in GRIDS:
        for floor in FLOORS:
            config = dataclasses.replace(
                CONFIG,
                scale=ScaleConfig(
                    analysis_width=width, analysis_height=height,
                    min_native_width=width, min_native_height=height,
                    allow_upsampling=False,
                ),
            )
            thresholds = QualityThresholds(min_oversampling=floor, warn_oversampling=3.0)
            table = build_feature_table(
                LEGACY_SCRAPED, config=config, thresholds=thresholds,
                include_legacy_tsci=False, progress_every=0,
            )
            usable = table[table.usable]
            counts = collections.Counter(usable.label)
            frac = counts["serviceable"] / max(1, len(usable))
            rows.append({
                "grid": f"{width}x{height}", "oversampling_floor": floor,
                "kept": int(len(usable)), "serviceable": int(counts["serviceable"]),
                "worn": int(counts["worn"]), "fraction_serviceable": float(frac),
            })
            print(f"{f'{width}x{height}':>10}{floor:>7.0f}{len(usable):>7}"
                  f"{counts['serviceable']:>13}{counts['worn']:>7}{frac:>14.0%}")

    at_floor_3 = [r for r in rows if r["oversampling_floor"] == 3.0]
    best_yield = max(at_floor_3, key=lambda r: r["kept"])

    print(f"\nBaseline: the published experiment used all 369 images "
          f"(234 serviceable / 135 worn, 63% serviceable).")
    print(f"Best yield at the evidence-based 3x floor: {best_yield['grid']} keeps "
          f"{best_yield['kept']} images at {best_yield['fraction_serviceable']:.0%} serviceable.")

    record_experiment(ExperimentRecord(
        experiment_id="exp003_oversampling_floor",
        title="Cost of enforcing resolution correctness on the legacy dataset",
        hypothesis=(
            "Enforcing downsample-only scale normalisation requires rejecting images "
            "that cannot fill the analysis grid. If the legacy dataset's resolution "
            "distribution is adequate, a scientifically defensible floor should "
            "retain most of it. If not, the dataset cannot support the corrected "
            "pipeline at any grid size."
        ),
        dataset="Legacy scraped dataset, all 369 images (234 serviceable / 135 worn).",
        method=(
            "Full tyretread pipeline at each candidate analysis grid, with the "
            "quality gate's oversampling floor set to each candidate value. An image "
            "is kept only if its ROI can be downsampled onto the grid without "
            "upsampling and clears every other quality check."
        ),
        validation=(
            "Descriptive. Counts surviving images and class balance per "
            "configuration; no model is fitted, because the question is whether "
            "enough usable data exists to fit one."
        ),
        metrics={
            "grids": [f"{w}x{h}" for w, h in GRIDS],
            "floors": FLOORS,
            "baseline_n": 369,
            "baseline_fraction_serviceable": 234 / 369,
            "best_at_3x": best_yield,
        },
        tables={"sweep": rows},
        interpretation=(
            "The legacy dataset cannot support a resolution-correct pipeline. At the "
            "3x floor that exp002 shows is needed for descriptor stability, every "
            "grid retains at most 80 of 369 images, and the class balance inverts "
            "from 63% serviceable to 34-40% - because the worn examples were scraped "
            "from systematically larger source images, which is the same confound "
            "exp001 identified, now visible as a survival bias. Relaxing the floor to "
            "1x keeps 289 images but reinstates the artefact the correction exists to "
            "remove. There is no configuration that is both honest and adequately "
            "powered on this data."
        ),
        decision=(
            "The legacy dataset is retired from training. Its remaining roles are to "
            "reproduce the original published result and to serve as an "
            "out-of-domain robustness check where its resolution permits. Training "
            "requires a dataset whose images are large enough that the oversampling "
            "floor is never the binding constraint - which is the scientific "
            "argument for the Mendeley phone-camera dataset, independent of its size. "
            "The production grid is deferred until it can be chosen on data that can "
            "actually support the measurement."
        ),
        seed=CONFIG.random_seed,
        config={"grids": str(GRIDS), "floors": str(FLOORS)},
    ))
    print("\nrecorded -> experiments/exp003_oversampling_floor.json")


if __name__ == "__main__":
    main()
