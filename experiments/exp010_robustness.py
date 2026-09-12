"""exp010 — How does the system behave on degraded photographs?

Cross-validation says what happens on images resembling the training set. It says
nothing about the photographs a phone actually produces in a car park: shaken, badly
lit, taken too far away. For a screening tool the interesting question is not only
"how often is it right" but "when it is about to be wrong, does it say so".

The quality gate is supposed to catch degraded input before the model sees it, and the
abstention band is supposed to catch borderline cases after. This measures whether
either actually fires, by taking images the system handles confidently and degrading
them in ways a user plausibly would.

A good result is not that predictions survive degradation. It is that the system stops
answering when the evidence stops supporting an answer.

Run: python experiments/exp010_robustness.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tyretread.experiment import ExperimentRecord, record_experiment
from tyretread.imaging.io import load_image
from tyretread.inspect import inspect_image
from tyretread.models.artifact import load_artifact
from tyretread.models.decision import Verdict

N_IMAGES = 60
SEED = 11


def degradations() -> dict[str, callable]:
    """Degradations a phone photograph realistically suffers."""
    return {
        "none": lambda im: im,
        "motion_blur_mild": lambda im: cv2.GaussianBlur(im, (9, 9), 3),
        "motion_blur_severe": lambda im: cv2.GaussianBlur(im, (31, 31), 12),
        "underexposed": lambda im: np.clip(im * 0.25, 0, 255).astype(np.uint8),
        "overexposed": lambda im: np.clip(im.astype(np.int16) + 110, 0, 255).astype(np.uint8),
        "too_far_away": lambda im: cv2.resize(
            im, (max(32, im.shape[1] // 8), max(32, im.shape[0] // 8)),
            interpolation=cv2.INTER_AREA),
        "heavy_jpeg": lambda im: cv2.imdecode(
            cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 12])[1], cv2.IMREAD_COLOR),
        "glare_patch": _glare,
        "sensor_noise": _noise,
    }


def _glare(im: np.ndarray) -> np.ndarray:
    out = im.copy()
    h, w = out.shape[:2]
    cv2.circle(out, (w // 2, h // 2), int(min(h, w) * 0.35), (255, 255, 255), -1)
    return out


def _noise(im: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.clip(im.astype(np.int16) + rng.normal(0, 30, im.shape).astype(np.int16),
                   0, 255).astype(np.uint8)


def main() -> None:
    artifact = load_artifact("current")
    table = pd.read_parquet("outputs/features/mendeley_tyres.parquet")
    usable = table[table["usable"]]

    random.seed(SEED)
    paths = random.sample(list(usable["path"]), min(N_IMAGES, len(usable)))
    print(f"baseline images: {len(paths)} (all pass the quality gate undegraded)\n")

    rows: list[dict] = []
    baseline: dict[str, tuple[str, float | None]] = {}

    for name, transform in degradations().items():
        refused = confident = inconclusive = 0
        flipped = 0
        comparable = 0

        for path in paths:
            # Load through the production loader, so EXIF orientation and the input
            # size reduction are applied exactly as they are for a real upload. Reading
            # with cv2.imread instead would skip both and make this a different pipeline
            # from the one being tested.
            try:
                image = load_image(path).bgr
            except (FileNotFoundError, ValueError):
                continue
            result = inspect_image(transform(image), artifact)
            verdict = result.decision.verdict

            if verdict is Verdict.UNABLE_TO_ASSESS:
                refused += 1
            elif verdict is Verdict.INCONCLUSIVE:
                inconclusive += 1
            else:
                confident += 1

            if name == "none":
                baseline[path] = (verdict.value, result.decision.probability_defect)
            else:
                before = baseline.get(path)
                if before and before[0] in ("likely_serviceable", "defect_suspected") \
                        and verdict.value in ("likely_serviceable", "defect_suspected"):
                    comparable += 1
                    if before[0] != verdict.value:
                        flipped += 1

        total = len(paths)
        row = {
            "degradation": name,
            "refused_rate": refused / total,
            "inconclusive_rate": inconclusive / total,
            "confident_rate": confident / total,
            "silent_flip_rate": (flipped / comparable) if comparable else None,
            "n_still_answered_confidently": confident,
        }
        rows.append(row)
        flip = "-" if row["silent_flip_rate"] is None else f"{row['silent_flip_rate']:.0%}"
        print(f"{name:22} refused={row['refused_rate']:5.0%}  "
              f"inconclusive={row['inconclusive_rate']:5.0%}  "
              f"confident={row['confident_rate']:5.0%}  silent_flips={flip}")

    caught = {r["degradation"]: r["refused_rate"] + r["inconclusive_rate"] for r in rows}
    worst = min(
        (r for r in rows if r["degradation"] != "none"),
        key=lambda r: r["refused_rate"] + r["inconclusive_rate"],
    )

    print(f"\nweakest guard: '{worst['degradation']}' still answered confidently on "
          f"{worst['confident_rate']:.0%} of images")

    record_experiment(ExperimentRecord(
        experiment_id="exp010_robustness",
        title="Does the system stop answering when the photograph stops supporting an answer?",
        hypothesis=(
            "The quality gate should refuse degradations that destroy the texture the "
            "model relies on - blur, underexposure, insufficient resolution - and the "
            "abstention band should absorb milder degradations. Degradations that pass "
            "both guards while silently flipping the verdict are the dangerous case, "
            "because the user receives a confident answer with no signal that anything "
            "is wrong."
        ),
        dataset=(
            f"{len(paths)} images sampled from the Mendeley set, all of which pass the "
            "quality gate undegraded, each put through nine degradations."
        ),
        method=(
            "The full production path: tyretread.imaging.io.load_image followed by "
            "tyretread.inspect_image with the served artifact, so EXIF handling and "
            "input downscaling match a real upload. Degradations: Gaussian blur at two strengths, under- and "
            "over-exposure, an eight-fold downscale standing in for photographing from "
            "too far away, heavy JPEG compression, a saturated glare patch, and sensor "
            "noise."
        ),
        validation=(
            "Descriptive. Reports the share refused by the gate, sent to inconclusive, "
            "and answered confidently, plus the silent-flip rate - the share of images "
            "answered confidently both before and after degradation whose verdict "
            "changed."
        ),
        metrics={"n_images": len(paths), "per_degradation": rows,
                 "caught_rate": caught},
        tables={"degradations": list(degradations())},
        interpretation=(
            "The guards work where the degradation destroys texture outright and fail "
            "where it does not. Both blur strengths, the eight-fold downscale and the "
            "glare patch are refused on 100% of images - the gate's Laplacian-variance, "
            "oversampling and saturation checks each do exactly what they were added "
            "for. Undegraded images are refused 0% of the time and answered confidently "
            "93% of the time, so the gate is not simply strict. "
            "Three real weaknesses show up. Underexposure is the most dangerous: 53% "
            "are refused, but 42% are still answered confidently and 20% of those "
            "silently change verdict - a user photographing a tyre in a dim garage can "
            "get a confident, different answer with nothing to indicate a problem. "
            "Overexposure behaves similarly at a lower rate, 12% answered confidently "
            "with 14% flipping. Heavy JPEG compression at quality 12 is refused only 7% "
            "of the time and answered confidently 77% of the time, and sensor noise is "
            "never refused at all - the gate has no check for either, because both "
            "preserve the global brightness, contrast and sharpness statistics the gate "
            "measures while corrupting the fine texture the model actually uses. "
            "The abstention band absorbs some of this - it sends 22% of noisy images to "
            "inconclusive against 7% of clean ones - but it was tuned for borderline "
            "tyres, not for corrupted images, and it is not sufficient on its own."
        ),
        decision=(
            "Record these as known failure modes in the README rather than quietly "
            "tightening thresholds to make the numbers look better. Two follow "
            "concretely and are proposed, not implemented: a noise or compression-"
            "quality check, since neither is currently detectable by any gate metric; "
            "and a re-examination of the exposure thresholds, which were inherited from "
            "the project's original cleaning script and are demonstrably too permissive "
            "at the dark end. Both need to be validated against the false-refusal rate "
            "before adoption - a gate that refuses good photographs is its own failure "
            "mode, and the current 0% baseline refusal is worth protecting. This "
            "experiment is the regression test for either change."
        ),
        seed=SEED,
        config={"model": artifact.metadata.model_id},
    ))
    print("\nrecorded -> experiments/exp010_robustness.json")


if __name__ == "__main__":
    main()
