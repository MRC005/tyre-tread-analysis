"""Image quality gate.

Why this is a first-class part of the system
--------------------------------------------
A classifier asked about an unusable photograph will still return a class and a
probability. For a system that speaks about vehicle safety, a confident answer
derived from a blurred or badly framed image is worse than no answer, because the
user has no way to tell the two apart.

The gate runs before any classification and can refuse outright. Refusal is a
first-class outcome, not an error: it returns the specific reasons and the
corresponding retake advice.

The checks originate in the project's own ``src/clean_dataset.py``, which already
contained sound heuristics (Laplacian variance for blur, mean intensity for
exposure, aspect ratio, standard deviation for texture). That logic only ever ran
offline over the training set. Promoting it to inference is where it earns its
keep, and it is extended with two checks the original lacked:

* **Resolution / oversampling.** Measured in ``experiments/exp003_oversampling``:
  the frequency-domain descriptors are only stable from roughly 3x oversampling
  upwards. Below that the analysis window cannot be filled with genuine detail, so
  the features are not comparable with those the model was trained on. This is the
  check that makes the scale-normalisation correction enforceable.
* **Glare.** A specular highlight saturates a region to white, destroying the
  texture there while leaving global brightness acceptable, so it slips past a
  mean-intensity test.
* **Subject coverage.** How much of the frame is actually tyre. This is the only
  check that catches a technically excellent photograph of mostly background, and it
  exists because real-device testing produced exactly that.
* **Exposure measured on the raw luma.** The original checks ran on the
  CLAHE-equalised image. CLAHE exists to normalise local contrast, so it hides
  precisely the exposure problem the check is looking for: an underexposed
  photograph whose raw mean is 30 comes out of CLAHE at 57, comfortably past a
  threshold of 30. Measured in ``exp012``, that let 47% of underexposed images
  through, one in five of which silently changed verdict.
* **Dynamic range**, which separates a genuinely dark tyre photographed well from an
  underexposed photograph. Both have a low mean; only the underexposed one has its
  histogram squashed into a narrow band.
* **Sensor noise and JPEG blockiness**, neither of which the original checks could
  see at all - both preserve mean, contrast and Laplacian variance while destroying
  the fine texture the model actually reads (``exp010``). Blockiness is measured on
  the 8-pixel grid JPEG quantises over, so it responds to heavy compression while
  leaving ordinary phone JPEGs alone.

Thresholds are declared here as data so they can be tuned against a dataset and
recorded in an experiment, rather than being scattered through the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

__all__ = ["QualityIssue", "QualityCheck", "QualityReport", "QualityThresholds",
           "assess_quality"]


class QualityIssue(str, Enum):
    """Every way an image can be refused, with the advice that follows from it."""

    TOO_BLURRY = "too_blurry"
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    LOW_CONTRAST = "low_contrast"
    GLARE = "glare"
    NOISY = "noisy"
    COMPRESSED = "over_compressed"
    RESOLUTION_TOO_LOW = "resolution_too_low"
    SUBJECT_TOO_SMALL = "subject_too_small"
    EXTREME_ASPECT = "extreme_aspect"
    TREAD_NOT_LOCATED = "tread_not_located"
    INSUFFICIENT_TEXTURE = "insufficient_texture"


#: Plain-language guidance shown to the user for each issue. Deliberately phrased
#: as an action, because "too blurry" alone does not tell anyone what to do.
RETAKE_ADVICE: dict[QualityIssue, str] = {
    QualityIssue.TOO_BLURRY: "Hold the phone still and let the camera focus before taking the photo.",
    QualityIssue.TOO_DARK: "Move somewhere brighter, or use your phone's flash.",
    QualityIssue.NOISY: (
        "There is too much image noise to read the surface. More light will let the "
        "camera use a lower sensitivity."
    ),
    QualityIssue.COMPRESSED: (
        "This image has been compressed too heavily to read the surface texture. Send "
        "the original photo rather than one forwarded through a messaging app."
    ),
    QualityIssue.TOO_BRIGHT: "Move out of direct sunlight, or shade the tyre with your body.",
    QualityIssue.LOW_CONTRAST: "Get closer so the grooves fill more of the frame.",
    QualityIssue.GLARE: "Change angle slightly to remove the bright reflection off the rubber.",
    QualityIssue.RESOLUTION_TOO_LOW: "Take the photo closer to the tread, or use a larger image.",
    QualityIssue.SUBJECT_TOO_SMALL: (
        "Move closer — the tyre should fill most of the frame, not sit in the middle "
        "of it."
    ),
    QualityIssue.EXTREME_ASPECT: "Frame the tread roughly squarely rather than as a narrow strip.",
    QualityIssue.TREAD_NOT_LOCATED: "Point the camera straight at the tread so it fills the middle of the frame.",
    QualityIssue.INSUFFICIENT_TEXTURE: "Make sure the tread pattern itself is in frame and in focus.",
}


@dataclass(frozen=True)
class QualityThresholds:
    #: Variance of the Laplacian. Below this an image is too soft to read tread
    #: texture from. Inherited from the project's original cleaning script.
    min_laplacian_variance: float = 80.0
    #: Measured on the RAW luma, not the CLAHE output. See the module docstring.
    min_raw_mean_intensity: float = 40.0
    #: 2nd-to-98th percentile spread of the raw luma. A well-exposed photograph of a
    #: black tyre still has a wide spread; an underexposed one does not. Threshold
    #: chosen in exp012 at 1.0% false rejection over 400 real images.
    min_dynamic_range: float = 50.0
    #: Fraction of raw pixels at the top of the range. Measured over 400 real images:
    #: p99 is 0.076 and the maximum 0.197, so 0.20 refuses none of them while catching
    #: 88.5% of overexposed photographs - against 75.5% at the looser 0.40 (exp012).
    max_bright_fraction: float = 0.20
    #: Mean absolute residual after a 3x3 median filter - structure survives, per-pixel
    #: noise does not. Threshold sits above the maximum seen on 400 real images.
    max_noise: float = 11.0
    #: Ratio of luma steps on the JPEG 8-pixel grid to those off it. 1.0 is no
    #: blocking; quality-85 phone JPEGs measure about 1.19, quality-12 about 3.4.
    max_blockiness: float = 2.2
    min_mean_intensity: float = 30.0
    max_mean_intensity: float = 220.0
    min_intensity_std: float = 15.0
    max_aspect_ratio: float = 4.0
    #: Fraction of pixels at or near saturation that counts as glare.
    max_saturated_fraction: float = 0.10
    saturation_level: int = 250
    #: Native ROI pixels per analysis pixel. See exp003_oversampling.
    min_oversampling: float = 1.0
    #: Below this, refuse; between this and ``min_oversampling`` for comfort,
    #: proceed but record a warning.
    warn_oversampling: float = 3.0
    min_roi_coverage: float = 0.05
    #: Fraction of the frame that must carry tyre-like texture.
    #:
    #: A photograph of a whole wheel from a metre away is sharp, well exposed and
    #: high-resolution - it fails none of the other checks - yet the analysed region is
    #: then mostly ground and bodywork, and the verdict flips more than half the time
    #: (exp013). Cropping to the tyre was measured and rejected: it repairs wide shots
    #: but roughly doubles the error rate on well-framed ones, and no confidence
    #: threshold separates the two (exp015). Detecting the framing is a far easier
    #: problem than localising the tread, so the system refuses and asks for a closer
    #: photograph instead.
    #:
    #: 0.20 refused none of 60 well-framed photographs while catching 90% of shots
    #: taken 2.5x too far back (exp016).
    min_subject_coverage: float = 0.20


@dataclass(frozen=True)
class QualityCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    issue: QualityIssue | None = None


@dataclass(frozen=True)
class QualityReport:
    usable: bool
    checks: list[QualityCheck] = field(default_factory=list)
    issues: list[QualityIssue] = field(default_factory=list)
    warnings: list[QualityIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def advice(self) -> list[str]:
        """Retake guidance for whatever blocked the assessment, most important first."""
        return [RETAKE_ADVICE[i] for i in self.issues]

    @property
    def warning_advice(self) -> list[str]:
        """Guidance for non-blocking caveats.

        A photograph can clear the gate and still be the reason a result came back
        inconclusive - a whole-wheel shot from a metre away passes every hard check
        while giving the model a tread band it never positively located. Real-device
        testing produced exactly that, and the advice the user needed ("get closer")
        existed in the system but was only ever attached to blocking issues.
        """
        return [RETAKE_ADVICE[w] for w in self.warnings if w in RETAKE_ADVICE]

    @property
    def summary(self) -> str:
        if self.usable and not self.warnings:
            return "Good"
        if self.usable:
            return "Acceptable"
        return "Unusable"

    def as_dict(self) -> dict[str, object]:
        return {
            "usable": self.usable,
            "summary": self.summary,
            "issues": [i.value for i in self.issues],
            "warnings": [w.value for w in self.warnings],
            "advice": self.advice,
            "warning_advice": self.warning_advice,
            "metrics": {k: round(v, 5) for k, v in self.metrics.items()},
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "value": round(c.value, 5),
                    "threshold": round(c.threshold, 5),
                }
                for c in self.checks
            ],
        }


def _sensor_metrics(raw: np.ndarray) -> dict[str, float]:
    """Exposure, noise and compression statistics, all from the raw luma."""
    f = raw.astype(np.float32)
    p2, p98 = float(np.percentile(f, 2)), float(np.percentile(f, 98))

    histogram = cv2.calcHist([raw], [0], None, [256], [0, 256]).ravel()
    histogram = histogram / max(histogram.sum(), 1.0)

    median = cv2.medianBlur(raw, 3).astype(np.float32)
    noise = float(np.mean(np.abs(f - median)))

    steps = np.abs(np.diff(f, axis=1))
    on_grid = float(steps[:, 7::8].mean()) if steps.shape[1] > 8 else 0.0
    off_grid = float(np.delete(steps, np.s_[7::8], axis=1).mean()) if steps.shape[1] > 8 else 1.0
    blockiness = on_grid / (off_grid + 1e-6) if off_grid > 0 else 1.0

    return {
        "raw_mean_intensity": float(f.mean()),
        "dynamic_range": p98 - p2,
        "bright_fraction": float(histogram[224:].sum()),
        "noise": noise,
        "blockiness": float(blockiness),
    }


def assess_quality(
    gray: np.ndarray,
    *,
    raw: np.ndarray | None = None,
    subject_coverage: float | None = None,
    oversampling: float | None = None,
    roi_method: str | None = None,
    roi_coverage: float | None = None,
    thresholds: QualityThresholds | None = None,
) -> QualityReport:
    """Decide whether ``gray`` can be assessed at all.

    Parameters
    ----------
    gray
        The preprocessed full-frame greyscale image, before ROI cropping. Exposure
        and blur are properties of the photograph, so they are measured on the
        whole frame rather than on the crop.
    raw
        The *unequalised* luma. Exposure, noise and compression are measured here
        because CLAHE hides all three. When omitted these checks are skipped, which
        keeps older callers working rather than silently passing everything.
    subject_coverage
        Fraction of the frame carrying tyre-like texture, from
        ``tyretread.imaging.localise.subject_coverage``. Catches the whole-wheel
        photograph that passes every other check.
    oversampling
        Native ROI pixels per analysis pixel, from
        ``preprocess.oversampling_factor``.
    roi_method, roi_coverage
        From ``roi.extract_roi``. A centre-band fallback means the tread was never
        positively located, which is worth telling the user about.
    """
    t = thresholds or QualityThresholds()
    checks: list[QualityCheck] = []
    issues: list[QualityIssue] = []
    warnings: list[QualityIssue] = []

    def record(name: str, value: float, threshold: float, ok: bool,
               issue: QualityIssue | None, fatal: bool = True) -> None:
        checks.append(QualityCheck(name, ok, float(value), float(threshold), issue))
        if not ok and issue is not None:
            (issues if fatal else warnings).append(issue)

    h, w = gray.shape[:2]

    sensor: dict[str, float] = {}
    if raw is not None:
        sensor = _sensor_metrics(raw)

        record("exposure_raw_mean", sensor["raw_mean_intensity"], t.min_raw_mean_intensity,
               sensor["raw_mean_intensity"] >= t.min_raw_mean_intensity, QualityIssue.TOO_DARK)
        record("exposure_dynamic_range", sensor["dynamic_range"], t.min_dynamic_range,
               sensor["dynamic_range"] >= t.min_dynamic_range, QualityIssue.TOO_DARK)
        record("exposure_bright_fraction", sensor["bright_fraction"], t.max_bright_fraction,
               sensor["bright_fraction"] <= t.max_bright_fraction, QualityIssue.TOO_BRIGHT)
        record("sensor_noise", sensor["noise"], t.max_noise,
               sensor["noise"] <= t.max_noise, QualityIssue.NOISY)
        record("jpeg_blockiness", sensor["blockiness"], t.max_blockiness,
               sensor["blockiness"] <= t.max_blockiness, QualityIssue.COMPRESSED)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    record("blur_laplacian_variance", lap_var, t.min_laplacian_variance,
           lap_var >= t.min_laplacian_variance, QualityIssue.TOO_BLURRY)

    mean_intensity = float(gray.mean())
    record("exposure_mean_low", mean_intensity, t.min_mean_intensity,
           mean_intensity >= t.min_mean_intensity, QualityIssue.TOO_DARK)
    record("exposure_mean_high", mean_intensity, t.max_mean_intensity,
           mean_intensity <= t.max_mean_intensity, QualityIssue.TOO_BRIGHT)

    std = float(gray.std())
    record("contrast_std", std, t.min_intensity_std,
           std >= t.min_intensity_std, QualityIssue.LOW_CONTRAST)

    saturated = float(np.count_nonzero(gray >= t.saturation_level) / gray.size)
    record("glare_saturated_fraction", saturated, t.max_saturated_fraction,
           saturated <= t.max_saturated_fraction, QualityIssue.GLARE)

    if subject_coverage is not None:
        record("subject_coverage", subject_coverage, t.min_subject_coverage,
               subject_coverage >= t.min_subject_coverage, QualityIssue.SUBJECT_TOO_SMALL)

    aspect = max(h, w) / max(1, min(h, w))
    record("aspect_ratio", aspect, t.max_aspect_ratio,
           aspect <= t.max_aspect_ratio, QualityIssue.EXTREME_ASPECT)

    if oversampling is not None:
        record("oversampling", oversampling, t.min_oversampling,
               oversampling >= t.min_oversampling, QualityIssue.RESOLUTION_TOO_LOW)
        if oversampling >= t.min_oversampling and oversampling < t.warn_oversampling:
            record("oversampling_comfortable", oversampling, t.warn_oversampling,
                   False, QualityIssue.RESOLUTION_TOO_LOW, fatal=False)

    if roi_method is not None and roi_method != "contour":
        record("roi_located", 0.0, 1.0, False,
               QualityIssue.TREAD_NOT_LOCATED, fatal=False)
    if roi_coverage is not None and roi_method == "contour":
        record("roi_coverage", roi_coverage, t.min_roi_coverage,
               roi_coverage >= t.min_roi_coverage, QualityIssue.TREAD_NOT_LOCATED)

    metrics = {
        "blur_laplacian_variance": lap_var,
        "mean_intensity": mean_intensity,
        "intensity_std": std,
        "saturated_fraction": saturated,
        "aspect_ratio": aspect,
        **sensor,
    }
    if oversampling is not None:
        metrics["oversampling"] = float(oversampling)
    if subject_coverage is not None:
        metrics["subject_coverage"] = float(subject_coverage)

    # Deduplicate while preserving order.
    issues = list(dict.fromkeys(issues))
    warnings = [wn for wn in dict.fromkeys(warnings) if wn not in issues]

    return QualityReport(
        usable=not issues, checks=checks, issues=issues,
        warnings=warnings, metrics=metrics,
    )
