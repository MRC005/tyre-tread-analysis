"""exp016 — Can the gate detect "the tyre is too small in this photograph"?

exp015 rejected texture search as a *localiser*: it helps wide shots, hurts
well-framed ones, and no confidence threshold separates the two because the score
saturates. But separability by rank was high (AUC 0.927), which says the signal knows
the difference even though it cannot be thresholded in that form.

This tests the reformulation. Instead of asking "where is the tread", ask "how much of
this frame is tyre" - a proportion, which uses the whole range instead of piling up
against a ceiling - and use it as a quality-gate check rather than a crop.

The product question is what matters: can a wide shot be refused with "move closer"
without refusing photographs that are perfectly fine?

Run: python experiments/exp016_coverage_check.py
"""

from __future__ import annotations

import random
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.imaging.localise import subject_coverage
from tyretread.imaging.preprocess import preprocess

N_IMAGES = 60
SEED = 41
#: Several degrees of "standing back", so the check is characterised across a range
#: rather than tuned to one synthetic distance.
ZOOMS = (1.6, 2.0, 2.5, 3.2)


def ground_texture(shape, rng):
    h, w = shape[:2]
    coarse = rng.integers(105, 165, (max(2, h // 24), max(2, w // 24), 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR), (21, 21), 7)


def wide_of(image, rng, zoom):
    h, w = image.shape[:2]
    sh, sw = int(h / zoom), int(w / zoom)
    canvas = ground_texture((h, w, 3), rng)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0 : y0 + sh, x0 : x0 + sw] = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas


def main() -> None:
    table = pd.read_parquet("outputs/features/mendeley_tyres.parquet")
    random.seed(SEED)
    paths = random.sample(list(table[table["usable"]]["path"]), N_IMAGES)

    close, wide = [], {z: [] for z in ZOOMS}
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        close.append(subject_coverage(preprocess(image).smoothed))
        for zoom in ZOOMS:
            composite = wide_of(image, np.random.default_rng(SEED), zoom)
            wide[zoom].append(subject_coverage(preprocess(composite).smoothed))

    close_arr = np.array(close)
    print(f"images: {len(close_arr)}\n")
    print("=== subject coverage by framing ===")
    print(f"  {'framing':16}{'median':>9}{'p10':>9}{'p90':>9}")
    print(f"  {'close-up':16}{np.median(close_arr):9.3f}{np.percentile(close_arr,10):9.3f}"
          f"{np.percentile(close_arr,90):9.3f}")
    per_zoom = {}
    for zoom in ZOOMS:
        a = np.array(wide[zoom])
        per_zoom[str(zoom)] = {
            "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)),
            "auc": float(roc_auc_score(
                np.r_[np.zeros(len(close_arr)), np.ones(len(a))], np.r_[-close_arr, -a])),
        }
        print(f"  {f'wide {zoom}x':16}{np.median(a):9.3f}{np.percentile(a,10):9.3f}"
              f"{np.percentile(a,90):9.3f}   AUC vs close-up {per_zoom[str(zoom)]['auc']:.3f}")

    print("\n=== operating points: refuse when coverage < T ===")
    print(f"  {'T':>6}{'false refusal':>15}" + "".join(f"{f'catch {z}x':>12}" for z in ZOOMS))
    print("  " + "-" * (21 + 12 * len(ZOOMS)))
    sweep = []
    for t in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50):
        false_refusal = float(np.mean(close_arr < t))
        catches = {str(z): float(np.mean(np.array(wide[z]) < t)) for z in ZOOMS}
        sweep.append({"threshold": t, "false_refusal": false_refusal, "catch": catches})
        print(f"  {t:6.2f}{false_refusal:15.1%}" + "".join(f"{catches[str(z)]:12.1%}" for z in ZOOMS))

    # A usable operating point refuses few good photographs while catching the framings
    # that actually break the assessment (2.5x and beyond, per exp013/exp015).
    usable = [s for s in sweep if s["false_refusal"] <= 0.05 and s["catch"]["2.5"] >= 0.70]
    best = min(usable, key=lambda s: s["false_refusal"]) if usable else None
    print(f"\n  operating points with <=5% false refusal and >=70% catch at 2.5x: "
          f"{[s['threshold'] for s in usable] if usable else 'NONE'}")
    if best:
        print(f"  recommended threshold: {best['threshold']:.2f}  "
              f"(false refusal {best['false_refusal']:.1%}, "
              f"catches {best['catch']['2.5']:.0%} at 2.5x, {best['catch']['3.2']:.0%} at 3.2x)")

    record_experiment(ExperimentRecord(
        experiment_id="exp016_coverage_check",
        title="Detecting a too-distant photograph, instead of trying to crop one",
        hypothesis=(
            "Texture search is a poor localiser but the underlying signal separates "
            "framings well. Reformulated as a proportion - how much of the frame carries "
            "tyre-like texture - it should support a quality-gate check that refuses "
            "wide shots with 'move closer' at an acceptable false-refusal rate on "
            "well-framed photographs."
        ),
        dataset=(
            f"{len(close_arr)} Mendeley images as close-ups, each also composited at "
            f"{list(ZOOMS)}x shrink into a constant-size frame, so the check is "
            "characterised across a range of distances rather than one."
        ),
        method=(
            "subject_coverage: fraction of coarse cells whose edge energy, weighted "
            "towards dark regions, reaches 40% of the frame's 95th-percentile score. A "
            "high percentile rather than the maximum, so one highlight cannot set the "
            "scale."
        ),
        validation=(
            "Within-subject: every wide composite derives from a close-up in the same "
            "set. ROC-AUC per zoom level, plus a threshold sweep reporting false "
            "refusal on close-ups against catch rate at each distance."
        ),
        metrics={
            "n_images": int(len(close_arr)),
            "close_median": float(np.median(close_arr)),
            "close_p10": float(np.percentile(close_arr, 10)),
            "per_zoom": per_zoom,
            "threshold_sweep": sweep,
            "recommended": best,
        },
        interpretation=(
            "The reformulation works where the localiser did not. As a proportion the "
            "signal uses its whole range instead of saturating: close-ups have a median "
            "coverage of 0.421 with a 10th percentile of 0.307, while wide shots sit "
            "between 0.12 and 0.22. ROC-AUC against close-ups is 0.95 to 0.99 across "
            "1.6x to 2.5x. "
            "That translates into a usable operating point, which is what exp015 could "
            "not find. Refusing below 0.20 rejects **none** of the close-ups in this set "
            "while catching 90% of 2.5x wide shots and 82% at 2.0x; moving to 0.25 "
            "costs 3.3% false refusal and catches 97%. The contrast with the localiser "
            "is the whole point - the same underlying measurement, asked a question it "
            "can answer. "
            "One irregularity is worth recording rather than smoothing over: catch rate "
            "falls at 3.2x (70%) relative to 2.5x (90%), and the 90th percentile of "
            "coverage rises to 0.366. At extreme shrink the tyre is small enough that "
            "the synthetic background begins to dominate the 95th-percentile "
            "normalisation, so the statistic is measured against a different scale. "
            "The check is therefore characterised for moderate framing errors, which is "
            "the realistic case, and is not claimed to degrade monotonically forever."
        ),
        decision=(
            "Adopt subject_coverage as a quality-gate check with a refusal threshold of "
            "0.20, not as a localiser and not as a crop. This is option D: when the tyre "
            "does not fill enough of the frame, say so and ask for a closer photograph, "
            "rather than cropping badly or analysing mostly background. "
            "It also completes the live capture guidance: the same statistic can drive a "
            "'move closer' hint before the shutter, so most users never reach the "
            "refusal. "
            "Not yet implemented in production - reported for approval first. Two "
            "caveats bound the claim. All of this is measured on synthetic composites "
            "with a hard rectangular boundary and a uniform surround, which is easier "
            "than a real wheel arch; the threshold needs confirming on real wide-shot "
            "photographs before it is trusted. And the check detects framing, not "
            "content: it cannot tell a distant tyre from a close-up of something that "
            "is not a tyre at all."
        ),
        seed=SEED,
        config={"zooms": list(ZOOMS)},
    ))
    print("\nrecorded -> experiments/exp016_coverage_check.json")


if __name__ == "__main__":
    main()
