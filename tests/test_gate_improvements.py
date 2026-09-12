"""The quality gate's second generation, and the regressions it must not reintroduce.

exp010 found three degradations the gate could not see. exp012 closed them. These
tests pin both halves: the blind spots stay closed, and the fixes do not start
refusing ordinary photographs - the failure mode a stricter gate creates.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tyretread.imaging.preprocess import preprocess
from tyretread.imaging.quality import (
    QualityIssue, QualityThresholds, _sensor_metrics, assess_quality,
)


def _assess(bgr: np.ndarray, **kwargs):
    pre = preprocess(bgr)
    return assess_quality(pre.enhanced, raw=pre.gray, oversampling=4.0,
                          roi_method="contour", roi_coverage=0.9, **kwargs)


# --------------------------------------------------------------------------
# The blind spots, now closed
# --------------------------------------------------------------------------

def test_sensor_noise_is_refused(noisy_image):
    """exp010: noise was refused 0% of the time and was the weakest guard."""
    report = _assess(noisy_image)
    assert not report.usable
    assert QualityIssue.NOISY in report.issues


def test_heavy_compression_is_refused(sample_photograph):
    """exp010: quality-12 JPEG was refused 7% of the time.

    Uses a real photograph rather than a synthetic fixture. The blockiness threshold
    was calibrated on real images, where quality-12 measures about 3.40; the synthetic
    textures compress far more gracefully (about 1.41) because they carry less fine
    detail for the quantiser to discard, so they are not a fair test of this check.
    """
    ok, buffer = cv2.imencode(".jpg", sample_photograph, [cv2.IMWRITE_JPEG_QUALITY, 12])
    assert ok
    report = _assess(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    assert QualityIssue.COMPRESSED in report.issues


def test_underexposure_is_refused(dark_image):
    """exp010: 47% of underexposed images got through, 20% flipping verdict."""
    report = _assess(dark_image)
    assert not report.usable
    assert QualityIssue.TOO_DARK in report.issues


def test_exposure_is_measured_before_clahe_not_after():
    """The root cause of the underexposure blind spot.

    CLAHE normalises local contrast, so measuring exposure on its output hides the
    very defect the check exists to find. If a future refactor passes the enhanced
    image as ``raw``, this test fails.
    """
    base = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
    dark = np.stack([(base * 0.12).astype(np.uint8)] * 3, axis=-1)

    pre = preprocess(dark)
    raw_mean = _sensor_metrics(pre.gray)["raw_mean_intensity"]
    clahe_mean = _sensor_metrics(pre.enhanced)["raw_mean_intensity"]

    assert raw_mean < clahe_mean, (
        "CLAHE is expected to raise apparent brightness; if it does not, the premise "
        "of measuring exposure on the raw luma needs revisiting"
    )
    assert raw_mean < QualityThresholds().min_raw_mean_intensity


# --------------------------------------------------------------------------
# ...without creating a new failure mode
# --------------------------------------------------------------------------

def test_an_ordinary_phone_jpeg_is_not_refused(normal_phone_jpeg):
    """The trap a blockiness check creates: refusing every phone photograph.

    Quality-85 measures about 1.19 blockiness against 3.40 at quality 12, and the
    threshold sits at 2.2.
    """
    report = _assess(normal_phone_jpeg)
    assert QualityIssue.COMPRESSED not in report.issues
    assert report.usable


def test_a_moderate_quality_jpeg_is_not_refused(grooved_tread):
    ok, buffer = cv2.imencode(".jpg", grooved_tread, [cv2.IMWRITE_JPEG_QUALITY, 60])
    report = _assess(cv2.imdecode(buffer, cv2.IMREAD_COLOR))
    assert QualityIssue.COMPRESSED not in report.issues


def test_a_dark_tyre_photographed_well_is_not_refused(dark_but_well_exposed):
    """Underexposure versus a genuinely dark subject.

    Both have a low mean; only the underexposed one has a compressed histogram, which
    is why dynamic range is a separate check from mean intensity.
    """
    metrics = _sensor_metrics(preprocess(dark_but_well_exposed).gray)
    assert metrics["dynamic_range"] >= QualityThresholds().min_dynamic_range, (
        "a well-exposed dark tyre must retain a wide dynamic range"
    )


def test_a_clean_image_passes_everything(grooved_tread):
    report = _assess(grooved_tread)
    assert report.usable
    assert report.issues == []


# --------------------------------------------------------------------------
# Metric behaviour
# --------------------------------------------------------------------------

def test_noise_metric_responds_to_noise_not_to_structure(grooved_tread, noisy_image):
    clean = _sensor_metrics(cv2.cvtColor(grooved_tread, cv2.COLOR_BGR2GRAY))
    noisy = _sensor_metrics(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))
    assert noisy["noise"] > clean["noise"] * 3


def test_blockiness_metric_responds_to_compression(sample_photograph):
    ok, buffer = cv2.imencode(".jpg", sample_photograph, [cv2.IMWRITE_JPEG_QUALITY, 12])
    assert ok
    heavy = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    clean = _sensor_metrics(cv2.cvtColor(sample_photograph, cv2.COLOR_BGR2GRAY))
    compressed = _sensor_metrics(cv2.cvtColor(heavy, cv2.COLOR_BGR2GRAY))
    assert compressed["blockiness"] > clean["blockiness"]
    assert compressed["blockiness"] > QualityThresholds().max_blockiness


def test_synthetic_textures_compress_more_gracefully_than_photographs(
    grooved_tread, over_compressed_image
):
    """Documents why the tests above need a real photograph.

    Kept so the reason is recorded in the suite rather than lost in a commit message.
    """
    clean = _sensor_metrics(cv2.cvtColor(grooved_tread, cv2.COLOR_BGR2GRAY))
    compressed = _sensor_metrics(cv2.cvtColor(over_compressed_image, cv2.COLOR_BGR2GRAY))
    assert compressed["blockiness"] > clean["blockiness"], "compression still shows"
    assert compressed["blockiness"] < QualityThresholds().max_blockiness, (
        "synthetic textures are expected to stay under the threshold; if this changes, "
        "the fixtures have become photograph-like and the tests above could use them"
    )


def test_sensor_checks_are_skipped_when_raw_is_absent(grooved_tread):
    """Backward compatibility: an older caller must not silently pass everything."""
    pre = preprocess(grooved_tread)
    report = assess_quality(pre.enhanced, oversampling=4.0, roi_method="contour",
                            roi_coverage=0.9)
    assert "noise" not in report.metrics
    assert report.usable


def test_every_new_issue_has_retake_advice(noisy_image, over_compressed_image):
    for image in (noisy_image, over_compressed_image):
        report = _assess(image)
        assert len(report.advice) == len(report.issues)
        assert all(a.strip() for a in report.advice)


# --------------------------------------------------------------------------
# Subject coverage — the check that closes the real-device failure
# --------------------------------------------------------------------------

def test_a_whole_wheel_shot_is_refused_with_move_closer(grooved_tread):
    """The failure real-device testing produced.

    A photograph of a whole wheel from a metre is sharp, well exposed and
    high-resolution - it fails none of the other checks - but the analysed region is
    then mostly background and the verdict flips more than half the time (exp013).
    Cropping to the tyre was measured and rejected (exp015); detecting the framing and
    asking for a closer photo was measured and adopted (exp016).
    """
    from tyretread.imaging.localise import subject_coverage as measure
    from tyretread.imaging.quality import QualityIssue

    tyre = grooved_tread
    h, w = tyre.shape[:2]
    small = cv2.resize(tyre, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    frame = np.full((h, w, 3), 140, dtype=np.uint8)
    y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    frame[y0 : y0 + small.shape[0], x0 : x0 + small.shape[1]] = small

    report = _assess(frame, subject_coverage=measure(preprocess(frame).smoothed))
    assert not report.usable
    assert QualityIssue.SUBJECT_TOO_SMALL in report.issues
    assert any("closer" in a.lower() for a in report.advice)


def test_a_well_framed_photo_is_not_refused_for_coverage(grooved_tread):
    """The cost side. exp016 measured 0% false refusal on 60 well-framed photographs."""
    from tyretread.imaging.localise import subject_coverage as measure
    from tyretread.imaging.quality import QualityIssue

    report = _assess(grooved_tread, subject_coverage=measure(preprocess(grooved_tread).smoothed))
    assert QualityIssue.SUBJECT_TOO_SMALL not in report.issues


def test_coverage_check_is_skipped_when_not_supplied(grooved_tread):
    """Backward compatibility: an older caller must not be silently refused."""
    from tyretread.imaging.quality import QualityIssue

    report = _assess(grooved_tread)
    assert QualityIssue.SUBJECT_TOO_SMALL not in report.issues
    assert "subject_coverage" not in report.metrics
