"""Measured evidence must stay descriptive and must not impersonate an explanation."""

from __future__ import annotations

import numpy as np
import pytest

from tyretread.models.evidence_features import (
    MIN_DISCRIMINATIVE_AUC, REPORTED_FEATURES, build_reference, feature_evidence,
    summarise_agreement,
)


@pytest.fixture
def reference(rng):
    """A reference where one feature separates the classes and one does not."""
    n = 400
    labels = np.array([0] * n + [1] * n)
    values = {}
    # edge_density: clearly separated -> should earn a "resembles" note
    values["edge_density"] = np.concatenate([rng.normal(0.10, 0.02, n), rng.normal(0.25, 0.02, n)])
    # spectral_slope: identical distributions -> must NOT earn one
    values["spectral_slope"] = np.concatenate([rng.normal(2.0, 0.3, n), rng.normal(2.0, 0.3, n)])
    for name in REPORTED_FEATURES:
        values.setdefault(name, rng.normal(1.0, 0.2, 2 * n))
    return build_reference(values, labels), labels


def test_reference_records_a_univariate_auc(reference):
    ref, _ = reference
    assert ref["edge_density"]["univariate_auc"] > 0.9
    assert abs(ref["spectral_slope"]["univariate_auc"] - 0.5) < 0.1


def test_a_discriminative_feature_earns_a_resemblance_note(reference):
    ref, _ = reference
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.26}, ref)
    edge = next(m for m in evidence.measurements if m.feature == "edge_density")
    assert edge.resembles == "tyres with visible defects"


def test_a_non_discriminative_feature_stays_silent(reference):
    """The guard against manufactured precision.

    A feature whose class medians differ only by noise must not produce a
    confident-sounding resemblance claim.
    """
    ref, _ = reference
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "spectral_slope": 2.4}, ref)
    slope = next(m for m in evidence.measurements if m.feature == "spectral_slope")
    assert slope.resembles is None
    assert ref["spectral_slope"]["univariate_auc"] < MIN_DISCRIMINATIVE_AUC


def test_percentiles_track_the_training_distribution(reference):
    ref, _ = reference
    low = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.02}, ref)
    high = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.40}, ref)
    low_edge = next(m for m in low.measurements if m.feature == "edge_density")
    high_edge = next(m for m in high.measurements if m.feature == "edge_density")
    assert low_edge.percentile < 10 and low_edge.band == "very low"
    assert high_edge.percentile > 90 and high_edge.band == "very high"


def test_a_median_value_reads_as_typical(reference):
    ref, _ = reference
    median = ref["edge_density"]["median"]
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": median}, ref)
    edge = next(m for m in evidence.measurements if m.feature == "edge_density")
    assert edge.band == "typical"


def test_the_note_refuses_to_claim_it_explains_the_decision(reference):
    ref, _ = reference
    note = feature_evidence({n: 1.0 for n in REPORTED_FEATURES}, ref).note.lower()
    assert "not a breakdown" in note
    assert "non-linear" in note
    for claim in ("because", "caused", "the reason"):
        assert claim not in note, f"the evidence note asserts causation via '{claim}'"


def test_diagnostics_are_carried_but_kept_separate(reference):
    ref, _ = reference
    evidence = feature_evidence(
        {n: 1.0 for n in REPORTED_FEATURES}, ref, diagnostics={"legacy_tsci": 0.61}
    )
    assert evidence.diagnostics["legacy_tsci"] == 0.61
    assert not any(m.feature == "legacy_tsci" for m in evidence.measurements), (
        "a diagnostic must not appear as though it were a model input"
    )


def test_agreement_is_stated_when_measurements_back_the_verdict(reference):
    ref, _ = reference
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.26}, ref)
    summarised = summarise_agreement(evidence, verdict="defect_suspected")
    assert summarised.agreement
    assert "point the same way" in summarised.agreement


def test_an_inconclusive_verdict_never_claims_a_side(reference):
    """Regression from real-device testing.

    summarise_agreement used to take a boolean, so the inconclusive verdict fell
    through to the serviceable branch and printed "point the same way as the overall
    assessment of good condition" underneath a result headed "Not conclusive".
    """
    ref, _ = reference
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.09}, ref)
    summarised = summarise_agreement(evidence, verdict="inconclusive")

    assert summarised.agreement
    assert "overall assessment of good condition" not in summarised.agreement
    assert "overall assessment of possible wear" not in summarised.agreement
    assert "inconclusive" in summarised.agreement or "boundary" in summarised.agreement


def test_a_split_of_measurements_is_reported_as_the_reason_for_inconclusive(reference):
    ref, _ = reference
    from tyretread.models.evidence_features import EvidenceReport, FeatureEvidence

    mixed = [
        FeatureEvidence(f"f{i}", "desc", 1.0, 50.0, "typical",
                        "tyres with visible defects" if i < 3 else "tyres in good condition")
        for i in range(5)
    ]
    summarised = summarise_agreement(
        EvidenceReport(measurements=mixed, diagnostics={}, note=""), verdict="inconclusive"
    )
    assert "split" in summarised.agreement
    assert "3 of 5" in summarised.agreement


def test_unanimous_measurements_do_not_imply_the_model_agreed_with_them(reference):
    """Regression from a real-device report.

    Every reported measurement resembled tyres in good condition while the calibrated
    probability (0.683) sat above the decision threshold (0.61) - inside the abstention
    band, so the verdict was inconclusive. The copy said the measurements were merely
    "not strongly enough in combination", which reads as though the combination leaned
    the same way. It did not. The side the model actually landed on must be stated.
    """
    from tyretread.models.evidence_features import EvidenceReport, FeatureEvidence

    unanimous_good = [
        FeatureEvidence(f"f{i}", "desc", 1.0, 50.0, "typical", "tyres in good condition")
        for i in range(5)
    ]
    summarised = summarise_agreement(
        EvidenceReport(measurements=unanimous_good, diagnostics={}, note=""),
        verdict="inconclusive",
        probability=0.683,
        threshold=0.61,
    )
    assert "wear or damage side of the decision line" in summarised.agreement
    assert "lean towards good condition individually" not in summarised.agreement


def test_unanimous_measurements_on_the_models_own_side_read_plainly(reference):
    from tyretread.models.evidence_features import EvidenceReport, FeatureEvidence

    unanimous_good = [
        FeatureEvidence(f"f{i}", "desc", 1.0, 50.0, "typical", "tyres in good condition")
        for i in range(5)
    ]
    summarised = summarise_agreement(
        EvidenceReport(measurements=unanimous_good, diagnostics={}, note=""),
        verdict="inconclusive",
        probability=0.55,
        threshold=0.61,
    )
    assert "lean towards good condition individually" in summarised.agreement
    assert "decision line" not in summarised.agreement


def test_disagreement_is_surfaced_not_hidden(reference):
    """The honesty case: a confident verdict with ordinary-looking measurements."""
    ref, _ = reference
    evidence = feature_evidence({**{n: 1.0 for n in REPORTED_FEATURES}, "edge_density": 0.09}, ref)
    summarised = summarise_agreement(evidence, verdict="defect_suspected")
    assert summarised.agreement
    assert "yet the combined assessment does" in summarised.agreement
    assert "contradiction" in summarised.agreement


def test_agreement_serialises(reference):
    import json
    ref, _ = reference
    evidence = summarise_agreement(
        feature_evidence({n: 1.0 for n in REPORTED_FEATURES}, ref), verdict="likely_serviceable"
    )
    json.dumps(evidence.as_dict())
