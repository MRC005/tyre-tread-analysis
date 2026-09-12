"""The quality gate must refuse unusable photographs for the right reasons.

A gate that rejects everything is as useless as one that accepts everything, so each
test pins one specific failure mode and checks that the reported issue matches the
defect actually present.
"""

from __future__ import annotations

import numpy as np

from tyretread.imaging.preprocess import preprocess
from tyretread.imaging.quality import QualityIssue, QualityThresholds, assess_quality


def _assess(bgr: np.ndarray, **kwargs):
    return assess_quality(preprocess(bgr).enhanced, **kwargs)


def test_accepts_a_well_formed_tread_image(grooved_tread):
    report = _assess(grooved_tread, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    assert report.usable
    assert report.issues == []


def test_rejects_a_blurred_image_as_blurry(blurred_image):
    report = _assess(blurred_image, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    assert not report.usable
    assert QualityIssue.TOO_BLURRY in report.issues


def test_rejects_a_dark_image_as_dark(dark_image):
    report = _assess(dark_image, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    assert not report.usable
    assert QualityIssue.TOO_DARK in report.issues


def test_rejects_low_resolution_by_oversampling(grooved_tread):
    report = _assess(grooved_tread, oversampling=0.4, roi_method="contour", roi_coverage=0.9)
    assert not report.usable
    assert QualityIssue.RESOLUTION_TOO_LOW in report.issues


def test_rejects_saturated_glare(grooved_tread):
    glared = grooved_tread.copy()
    glared[:, : glared.shape[1] // 2] = 255
    report = _assess(glared, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    assert not report.usable
    assert QualityIssue.GLARE in report.issues


def test_centre_band_fallback_warns_without_refusing(grooved_tread):
    report = _assess(grooved_tread, oversampling=4.0, roi_method="centre_band")
    assert report.usable, "a fallback ROI is a caveat, not a reason to refuse"
    assert QualityIssue.TREAD_NOT_LOCATED in report.warnings


def test_every_issue_carries_actionable_advice(blurred_image):
    report = _assess(blurred_image, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    assert len(report.advice) == len(report.issues)
    assert all(isinstance(a, str) and a.strip() for a in report.advice)


def test_borderline_oversampling_warns_but_proceeds(grooved_tread):
    thresholds = QualityThresholds(min_oversampling=1.0, warn_oversampling=3.0)
    report = _assess(grooved_tread, oversampling=1.5, roi_method="contour",
                     roi_coverage=0.9, thresholds=thresholds)
    assert report.usable
    assert QualityIssue.RESOLUTION_TOO_LOW in report.warnings
    assert QualityIssue.RESOLUTION_TOO_LOW not in report.issues


def test_report_serialises_to_json_safe_types(grooved_tread):
    import json
    report = _assess(grooved_tread, oversampling=4.0, roi_method="contour", roi_coverage=0.9)
    json.dumps(report.as_dict())
