"""Tread-versus-sidewall reporting.

The behavioural contract is unusual and deliberate: this detector must *never* cause
an image to be refused. It reaches AUC 0.797 [0.698, 0.878] on 120 hand labels, which
is real signal but not grounds for rejection - and the production model performs
comparably on both surfaces, so refusing sidewall images would discard working
functionality (exp011).
"""

from __future__ import annotations

import numpy as np
import pytest

from tyretread.imaging.surface import (
    NOT_TREAD_BELOW, SURFACE_FEATURES, TREAD_ABOVE, Surface, assess_surface,
    fit_surface_detector,
)


@pytest.fixture
def detector(rng):
    """A detector where the first feature alone decides the surface."""
    X = rng.normal(size=(200, len(SURFACE_FEATURES)))
    y = (X[:, 0] > 0).astype(int)
    return fit_surface_detector(X, y)


def _features(values):
    return dict(zip(SURFACE_FEATURES, values))


def test_detector_serialises_as_plain_numbers(detector):
    """Stored as coefficients so the artifact carries no second binary."""
    import json
    json.dumps(detector)
    assert set(detector["feature_names"]) == set(SURFACE_FEATURES)
    assert len(detector["coef"]) == len(SURFACE_FEATURES)


def test_high_tread_evidence_reads_as_tread(detector):
    assessment = assess_surface(_features([4.0] + [0.0] * 7), detector)
    assert assessment.surface is Surface.TREAD
    assert assessment.tread_probability >= TREAD_ABOVE


def test_low_tread_evidence_reads_as_not_tread(detector):
    assessment = assess_surface(_features([-4.0] + [0.0] * 7), detector)
    assert assessment.surface is Surface.NOT_TREAD
    assert assessment.tread_probability < NOT_TREAD_BELOW


def test_the_middle_band_admits_uncertainty(detector):
    """The state that matters: the system saying it does not know."""
    found = False
    for value in np.linspace(-2.0, 2.0, 200):
        assessment = assess_surface(_features([float(value)] + [0.0] * 7), detector)
        if assessment.surface is Surface.UNCLEAR:
            found = True
            assert NOT_TREAD_BELOW <= assessment.tread_probability < TREAD_ABOVE
    assert found, "there must be an input range that yields an honest 'unclear'"


def test_a_non_tread_result_disclaims_any_tread_statement(detector):
    """The failure this exists to prevent: a sidewall assessment read as tread."""
    note = assess_surface(_features([-4.0] + [0.0] * 7), detector).note.lower()
    assert "sidewall" in note
    assert "tread" in note
    assert "says nothing about how much tread is left" in note


def test_an_unclear_result_does_not_claim_tread(detector):
    for value in np.linspace(-2.0, 2.0, 200):
        assessment = assess_surface(_features([float(value)] + [0.0] * 7), detector)
        if assessment.surface is Surface.UNCLEAR:
            assert "not clear" in assessment.note.lower()
            return


def test_a_missing_detector_degrades_quietly(detector):
    """An artifact predating the detector must omit the section, not fail."""
    assert assess_surface(_features([0.0] * 8), None) is None
    assert assess_surface(_features([0.0] * 8), {}) is None


def test_missing_features_return_none_rather_than_guessing(detector):
    partial = {name: 0.0 for name in SURFACE_FEATURES[:3]}
    assert assess_surface(partial, detector) is None


def test_surface_never_appears_as_a_quality_issue():
    """The contract: this detector reports, it does not reject.

    Guards against a future change wiring surface into the gate, which exp011
    measured as costing one genuine tread photograph in five.
    """
    from tyretread.imaging.quality import QualityIssue
    values = {issue.value for issue in QualityIssue}
    assert "sidewall_or_shoulder" not in values
    assert not any("surface" in v for v in values)
