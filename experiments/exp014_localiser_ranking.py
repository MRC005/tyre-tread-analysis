"""exp014 — Which localiser actually finds the tread, and is any of them good enough?

exp013 showed the wide-shot failure is dominantly localisation, and put a ceiling on
what fixing it can buy: an oracle crop recovers about 80% of the probability drift.
This ranks candidate localisers against that ceiling.

Evaluation is on synthetic composites, where the true tyre rectangle is known exactly,
so intersection-over-union needs no annotation and every method is scored on identical
images. The synthetic surround is easier than a real wheel arch, so this ranks methods;
it does not certify one. That is stated in the decision rather than buried.

Two numbers matter and they are not the same:

  **IoU** - does the region land on the tyre?
  **Probability recovery** - does that translate into the right answer?

A method can improve IoU and change nothing downstream, which would be an argument
against adopting it.

Run: python experiments/exp014_localiser_ranking.py
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

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.features.extract import extract_features
from tyretread.imaging.localise import LOCALISERS
from tyretread.imaging.preprocess import normalise_scale, preprocess
from tyretread.models.artifact import load_artifact

N_IMAGES = 40
SEED = 17
ZOOM_OUT = 2.5


def ground_texture(shape, rng):
    h, w = shape[:2]
    coarse = rng.integers(105, 165, (max(2, h // 24), max(2, w // 24), 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR), (21, 21), 7)


def composite(tyre, rng):
    """Constant frame, shrunken tyre - see exp013 for why this is the correct model."""
    h, w = tyre.shape[:2]
    sh, sw = int(h / ZOOM_OUT), int(w / ZOOM_OUT)
    canvas = ground_texture((h, w, 3), rng)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0 : y0 + sh, x0 : x0 + sw] = cv2.resize(tyre, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas, (x0, y0, sw, sh)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def purity(box, true_box):
    """Fraction of the proposed region that is actually tyre - background contamination."""
    bx, by, bw, bh = box
    tx, ty, tw, th = true_box
    x1, y1 = max(bx, tx), max(by, ty)
    x2, y2 = min(bx + bw, tx + tw), min(by + bh, ty + th)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / (bw * bh) if bw * bh else 0.0


def probability_of(bgr, box, artifact):
    """Probability from a given crop, through the real feature path."""
    x, y, w, h = box
    crop = bgr[y : y + h, x : x + w]
    if crop.size == 0 or min(crop.shape[:2]) < 16:
        return None
    try:
        extraction = extract_features(crop)
    except Exception:
        return None
    if not extraction.usable:
        return None
    return float(artifact.probability_defect(extraction.features))


def main() -> None:
    artifact = load_artifact("current")
    table = pd.read_parquet("outputs/features/mendeley_tyres.parquet")
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    paths = random.sample(list(table[table["usable"]]["path"]), N_IMAGES)

    names = list(LOCALISERS)
    ious = {n: [] for n in names}
    purities = {n: [] for n in names}
    confidences = {n: [] for n in names}
    drifts = {n: [] for n in names}
    baseline_drift = []

    for path in paths:
        original = cv2.imread(path)
        if original is None:
            continue
        wide, true_box = composite(original, rng)
        gray = preprocess(wide).smoothed

        p_close = probability_of(original, (0, 0, original.shape[1], original.shape[0]), artifact)
        p_oracle = probability_of(wide, true_box, artifact)
        if p_close is None or p_oracle is None:
            continue
        baseline_drift.append(abs(p_oracle - p_close))

        for name, fn in LOCALISERS.items():
            candidate = fn(gray)
            ious[name].append(iou(candidate.box, true_box))
            purities[name].append(purity(candidate.box, true_box))
            confidences[name].append(candidate.confidence)
            p = probability_of(wide, candidate.box, artifact)
            drifts[name].append(abs(p - p_close) if p is not None else np.nan)

    n = len(baseline_drift)
    print(f"images evaluated: {n}\n")
    print(f"{'localiser':22}{'IoU med':>9}{'purity med':>12}{'|Δp| mean':>11}{'assessable':>12}")
    print("-" * 66)
    rows = []
    for name in names:
        d = np.array(drifts[name], dtype=float)
        assessable = float(np.mean(~np.isnan(d)))
        row = {
            "localiser": name,
            "iou_median": float(np.median(ious[name])),
            "iou_mean": float(np.mean(ious[name])),
            "purity_median": float(np.median(purities[name])),
            "drift_mean": float(np.nanmean(d)) if assessable else float("nan"),
            "assessable_rate": assessable,
            "confidence_mean": float(np.mean(confidences[name])),
        }
        rows.append(row)
        print(f"{name:22}{row['iou_median']:9.3f}{row['purity_median']:12.3f}"
              f"{row['drift_mean']:11.4f}{assessable:12.1%}")

    oracle_drift = float(np.mean(baseline_drift))
    print(f"\n  oracle (perfect localisation) |Δp| = {oracle_drift:.4f}   <- the floor")
    best = min((r for r in rows if not np.isnan(r["drift_mean"])), key=lambda r: r["drift_mean"])
    current = next(r for r in rows if r["localiser"] == "centre_band")
    closed = ((current["drift_mean"] - best["drift_mean"])
              / max(current["drift_mean"] - oracle_drift, 1e-9))
    print(f"  current centre_band          |Δp| = {current['drift_mean']:.4f}")
    print(f"  best candidate ({best['localiser']}) |Δp| = {best['drift_mean']:.4f}")
    print(f"  -> closes {closed:.0%} of the gap between current and perfect")

    # Does ROI confidence predict whether the crop was any good? If it does, it can gate.
    print("\n=== can a localiser's own confidence predict a good crop? ===")
    correlations = {}
    for name in names:
        c = np.array(confidences[name], dtype=float)
        i = np.array(ious[name], dtype=float)
        if c.std() > 1e-9:
            r = float(np.corrcoef(c, i)[0, 1])
            correlations[name] = r
            print(f"  {name:22} corr(confidence, IoU) = {r:+.3f}")
        else:
            correlations[name] = float("nan")
            print(f"  {name:22} confidence is constant - cannot gate on it")

    record_experiment(ExperimentRecord(
        experiment_id="exp014_localiser_ranking",
        title="Ranking simple tread localisers against an exact ground truth",
        hypothesis=(
            "Tread carries dense, dark, directionally coherent texture that its "
            "surroundings do not, so a texture-energy search should localise it better "
            "than a fixed centre-band crop - and that improvement should show up "
            "downstream as a calibrated probability closer to the close-up value, not "
            "merely as a better overlap number."
        ),
        dataset=(
            f"{n} Mendeley images composited into constant-size frames with the tyre "
            f"shrunk {ZOOM_OUT}x, so the true tyre rectangle is known exactly. Identical "
            "images for every method."
        ),
        method=(
            "Five localisers: the production centre band; the production contour "
            "refinement; densest edge energy; edge energy weighted by darkness; and "
            "edge energy weighted by darkness and structure-tensor coherence. Region "
            "search is exhaustive over a coarse score map via an integral image."
        ),
        validation=(
            "Intersection-over-union and purity against the known box, plus the "
            "downstream absolute probability drift from the close-up value. An oracle "
            "crop supplies the floor. Ranking only - the synthetic surround is more "
            "forgiving than a real wheel arch."
        ),
        metrics={
            "n_images": n, "zoom_out": ZOOM_OUT,
            "per_localiser": rows,
            "oracle_drift": oracle_drift,
            "gap_closed_by_best": float(closed),
            "confidence_iou_correlation": correlations,
        },
        interpretation="",
        decision="",
        seed=SEED,
        config={"localisers": names},
    ))
    print("\nrecorded -> experiments/exp014_localiser_ranking.json")


if __name__ == "__main__":
    main()
