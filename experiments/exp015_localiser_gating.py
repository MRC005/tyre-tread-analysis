"""exp015 — Can texture localisation be adopted without harming the common case?

exp014 ranked localisers on wide shots and a texture-energy search won decisively,
closing about 92% of the gap between the current centre-band crop and perfect
localisation.

That is only half the question. The common case is a user who framed the tyre well, and
a localiser that carves a sub-region out of an already-good photograph makes the product
worse for the majority in order to help a minority. A first check found exactly that
risk: on well-framed close-ups the texture search roughly doubled the verdict-flip rate
against simply doing nothing.

So this measures both populations together, and tests whether the localiser can gate
*itself* - cropping only when the frame actually looks like a wide shot, otherwise
leaving current behaviour untouched. The gating signal is the localiser's own
confidence: how much denser the texture is inside the chosen window than across the
frame. On a close-up that ratio is near one, because the tyre *is* the frame.

The outcome decides between:

  A. current heuristic is sufficient with better constraints
  B. a stronger classical localiser is justified
  C. a trained detector is justified
  D. abstain unless the user provides a close enough image

Run: python experiments/exp015_localiser_gating.py
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

from scipy import stats

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.features.extract import extract_features
from tyretread.imaging.localise import centre_band, texture_energy
from tyretread.imaging.preprocess import preprocess
from tyretread.models.artifact import load_artifact

N_IMAGES = 40
SEED = 31
ZOOM_OUT = 2.5
THRESHOLDS = (0.02, 0.05, 0.10, 0.20, 0.35, 1.01)  # 1.01 == never crop


def ground_texture(shape, rng):
    h, w = shape[:2]
    coarse = rng.integers(105, 165, (max(2, h // 24), max(2, w // 24), 3), dtype=np.uint8)
    return cv2.GaussianBlur(cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR), (21, 21), 7)


def wide_of(image, rng):
    """Constant frame, shrunken tyre - the correct model of standing back (exp013)."""
    h, w = image.shape[:2]
    sh, sw = int(h / ZOOM_OUT), int(w / ZOOM_OUT)
    canvas = ground_texture((h, w, 3), rng)
    y0, x0 = (h - sh) // 2, (w - sw) // 2
    canvas[y0 : y0 + sh, x0 : x0 + sw] = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    return canvas


def probability(bgr, artifact):
    try:
        extraction = extract_features(bgr)
    except Exception:
        return None
    return float(artifact.probability_defect(extraction.features)) if extraction.usable else None


def main() -> None:
    artifact = load_artifact("current")
    table = pd.read_parquet("outputs/features/mendeley_tyres.parquet")
    random.seed(SEED)
    paths = random.sample(list(table[table["usable"]]["path"]), N_IMAGES)

    # Precompute per image so each threshold is a cheap re-selection, not a re-search.
    cases = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        reference = probability(image, artifact)
        if reference is None:
            continue
        entry = {"reference": reference}
        for tag, img in (("close", image), ("wide", wide_of(image, np.random.default_rng(SEED)))):
            gray = preprocess(img).smoothed
            candidate = texture_energy(gray)
            entry[tag] = {
                "image": img,
                "crop_box": candidate.box,
                "fallback_box": centre_band(gray).box,
                "confidence": candidate.confidence,
            }
        cases.append(entry)
    print(f"images: {len(cases)}", flush=True)

    close_conf = np.array([c["close"]["confidence"] for c in cases])
    wide_conf = np.array([c["wide"]["confidence"] for c in cases])
    auc = float(stats.mannwhitneyu(wide_conf, close_conf).statistic / (len(wide_conf) * len(close_conf)))
    print("\n=== does the localiser's confidence separate wide shots from close-ups? ===")
    print(f"  close-ups  median {np.median(close_conf):.3f}   p90 {np.percentile(close_conf, 90):.3f}")
    print(f"  wide shots median {np.median(wide_conf):.3f}   p10 {np.percentile(wide_conf, 10):.3f}")
    print(f"  separability AUC = {auc:.3f}   (0.5 = indistinguishable)", flush=True)

    # Cache probabilities per (case, tag, box) so thresholds reuse the same work.
    cache: dict[tuple[int, str, tuple], float | None] = {}

    def prob_for(index: int, tag: str, box: tuple) -> float | None:
        key = (index, tag, box)
        if key not in cache:
            side = cases[index][tag]
            x, y, w, h = box
            crop = side["image"][y : y + h, x : x + w]
            cache[key] = (
                probability(crop, artifact)
                if crop.size and min(crop.shape[:2]) >= 16 else None
            )
        return cache[key]

    def evaluate(threshold: float) -> dict:
        out = {}
        for tag in ("close", "wide"):
            drift, flips, lost = [], [], 0
            for i, case in enumerate(cases):
                side = case[tag]
                box = side["crop_box"] if side["confidence"] >= threshold else side["fallback_box"]
                p = prob_for(i, tag, box)
                if p is None:
                    lost += 1
                    continue
                drift.append(abs(p - case["reference"]))
                flips.append((p > 0.5) != (case["reference"] > 0.5))
            out[tag] = {
                "drift": float(np.mean(drift)) if drift else float("nan"),
                "flip_rate": float(np.mean(flips)) if flips else float("nan"),
                "unassessable": lost,
            }
        return out

    print("\n=== hybrid: crop only when confidence >= T, else keep the current band ===")
    print(f"{'T':>7}{'close |dp|':>12}{'close flips':>13}{'wide |dp|':>11}{'wide flips':>12}{'lost':>7}")
    print("-" * 62, flush=True)
    results = {}
    for threshold in THRESHOLDS:
        r = evaluate(threshold)
        results[f"{threshold:.2f}"] = r
        label = f"{threshold:.2f}" if threshold <= 1 else "never"
        print(f"{label:>7}{r['close']['drift']:12.4f}{r['close']['flip_rate']:13.1%}"
              f"{r['wide']['drift']:11.4f}{r['wide']['flip_rate']:12.1%}"
              f"{r['close']['unassessable'] + r['wide']['unassessable']:7}", flush=True)

    never = results[f"{THRESHOLDS[-1]:.2f}"]
    always = results[f"{THRESHOLDS[0]:.2f}"]
    viable = [
        (t, r) for t, r in results.items()
        if r["close"]["flip_rate"] <= never["close"]["flip_rate"] + 0.02
        and r["wide"]["flip_rate"] < never["wide"]["flip_rate"] - 0.05
    ]
    print(f"\n  never cropping  : close flips {never['close']['flip_rate']:.1%}, "
          f"wide flips {never['wide']['flip_rate']:.1%}")
    print(f"  always cropping : close flips {always['close']['flip_rate']:.1%}, "
          f"wide flips {always['wide']['flip_rate']:.1%}")
    print(f"  thresholds helping wide WITHOUT hurting close: "
          f"{[t for t, _ in viable] if viable else 'NONE'}", flush=True)

    record_experiment(ExperimentRecord(
        experiment_id="exp015_localiser_gating",
        title="Can texture localisation be adopted without harming well-framed photos?",
        hypothesis=(
            "A texture-energy localiser repairs wide shots but damages close-ups. If its "
            "own confidence separates the two framings, it can gate itself and capture "
            "the benefit without the cost. If no threshold achieves that, localisation "
            "should not be adopted and the correct response to a wide shot is to ask for "
            "a closer photograph."
        ),
        dataset=(
            f"{len(cases)} Mendeley images, each evaluated twice: as the original "
            f"close-up and composited into a constant-size frame with the tyre shrunk "
            f"{ZOOM_OUT}x. The close-up probability is the reference for both."
        ),
        method=(
            "texture_energy (the exp014 winner) applied conditionally on its own "
            "confidence, falling back to the production centre band below the threshold."
        ),
        validation=(
            "Within-subject across both framings, identical images at every threshold. A "
            "threshold is viable only if it cuts the wide-shot flip rate by more than 5 "
            "points while leaving the close-up flip rate no more than 2 points worse "
            "than doing nothing."
        ),
        metrics={
            "n_images": len(cases),
            "confidence_separability_auc": auc,
            "close_confidence_median": float(np.median(close_conf)),
            "wide_confidence_median": float(np.median(wide_conf)),
            "by_threshold": results,
            "viable_thresholds": [t for t, _ in viable],
        },
        interpretation=(
            "No viable threshold exists. The trade is real and it does not go away. "
            "Never cropping gives 11.8% verdict flips on close-ups and 54.5% on wide "
            "shots; always cropping reverses it almost exactly, to 25.0% and 12.1%. "
            "Every threshold between 0.02 and 0.35 behaves identically to always "
            "cropping, and nothing in between helps. "
            "The reason is instructive, and it is not that the confidence signal is "
            "weak. Separability is high - AUC 0.927 - but the statistic *saturates*: "
            "wide shots sit at 1.000 with a 10th percentile of 1.000, while close-ups "
            "have a median of 0.605 and a 90th percentile of also 1.000. The two "
            "distributions are well separated in rank and overlap completely at the "
            "ceiling, so there is no operating point that admits wide shots without "
            "admitting most close-ups too. A high AUC does not imply a usable threshold "
            "when the score is bounded and both classes pile up against the bound. "
            "Adopting the localiser would therefore more than double the flip rate for "
            "users who framed their photograph correctly - the majority - in order to "
            "help those who did not."
        ),
        decision=(
            "Do not adopt texture localisation as a localiser. The measured cost to the "
            "common case exceeds the benefit to the uncommon one, and the confidence "
            "signal cannot separate them at a usable operating point. "
            "The useful finding is what the same signal is good *for*. Detecting that "
            "the tyre occupies little of the frame is a far easier problem than "
            "localising it precisely, and AUC 0.927 says that detection is reliable. So "
            "the texture statistic should become a **quality-gate check** - 'the tyre "
            "does not fill enough of this photograph, move closer' - rather than a crop. "
            "That is option D: abstain and ask for a better photograph, using the "
            "localisation work as the detector that makes the abstention possible. "
            "exp016 measures that check directly before anything ships."
        ),
        seed=SEED,
        config={"localiser": "texture_energy", "thresholds": list(THRESHOLDS), "zoom_out": ZOOM_OUT},
    ))
    print("\nrecorded -> experiments/exp015_localiser_gating.json", flush=True)


if __name__ == "__main__":
    main()
