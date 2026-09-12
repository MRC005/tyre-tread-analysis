"""The verdict layer must never present a weak prediction as a firm answer."""

from __future__ import annotations

import json

import pytest

from tyretread.models.decision import VERDICT_COPY, Verdict, decide

THRESHOLD = 0.5
BAND = 0.12


def _decide(p):
    return decide(p, threshold=THRESHOLD, abstain_band=BAND)


def test_a_refused_image_never_reaches_the_model():
    decision = _decide(None)
    assert decision.verdict is Verdict.UNABLE_TO_ASSESS
    assert decision.probability_defect is None
    assert decision.confidence is None


def test_probabilities_inside_the_band_are_inconclusive():
    for p in (THRESHOLD - BAND, THRESHOLD, THRESHOLD + BAND):
        assert _decide(p).verdict is Verdict.INCONCLUSIVE


def test_confident_probabilities_resolve_either_way():
    assert _decide(0.02).verdict is Verdict.LIKELY_SERVICEABLE
    assert _decide(0.98).verdict is Verdict.DEFECT_SUSPECTED


def test_confidence_is_zero_at_the_band_edge_and_rises_outward():
    just_outside = _decide(THRESHOLD + BAND + 1e-6).confidence
    far_outside = _decide(0.99).confidence
    assert just_outside == pytest.approx(0.0, abs=1e-4)
    assert far_outside > 0.9
    assert far_outside > just_outside


def test_confidence_stays_within_unit_range():
    for p in [0.0, 0.1, 0.4, 0.5, 0.6, 0.9, 1.0]:
        confidence = _decide(p).confidence
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0


def test_a_zero_band_still_produces_a_decision():
    decision = decide(0.51, threshold=0.5, abstain_band=0.0)
    assert decision.verdict is Verdict.DEFECT_SUSPECTED


def test_every_verdict_has_user_facing_copy_and_a_severity():
    for verdict in Verdict:
        copy = VERDICT_COPY[verdict]
        assert copy.headline and copy.detail and copy.recommendation
        assert copy.severity in {"ok", "caution", "alert", "unknown"}


def test_no_verdict_claims_certainty_or_a_depth_measurement():
    """Responsible-design guard: the wording must stay hedged.

    Blocks a future copy edit from turning a screening result into an assertion of
    safety or legality, or from implying a tread-depth measurement the data cannot
    support. Phrases rather than bare words, because a disclaimer may legitimately
    mention legality ("do not rely on this to decide whether the tyre is legal")
    while a claim may not ("this tyre is legal").
    """
    forbidden_claims = (
        "is legal", "is safe", "safe to drive", "is roadworthy", "guarantee",
        "guaranteed", "certified", "approved", "definitely", "exactly",
        "tread depth is", "millimetre", "millimeter", " mm ",
        # Added when the product was reframed: the data labels tyre condition, not
        # tread depth, so no copy may assert a depth or a remaining-life figure.
        "depth of", "remaining tread", "tread remaining",
    )
    for verdict, copy in VERDICT_COPY.items():
        text = f"{copy.headline} {copy.detail} {copy.recommendation}".lower()
        for phrase in forbidden_claims:
            assert phrase not in text, (
                f"{verdict.value} copy contains the claim-like phrase '{phrase}': {text}"
            )


def test_no_verdict_claims_to_measure_tread_depth():
    """The training labels describe tyre condition, not depth (exp006, exp008).

    The serviceable verdict must actively disclaim depth measurement rather than
    leaving a reader to assume it, since "no visible defects" is easily misread as
    "plenty of tread left".
    """
    serviceable = VERDICT_COPY[Verdict.LIKELY_SERVICEABLE]
    # Checks every field the user sees, not just two: the taxonomy rework moved this
    # disclaimer into `meaning`, and a guard that inspects a subset of the copy will
    # pass a rewrite that quietly dropped it.
    text = " ".join(
        [serviceable.label, serviceable.headline, serviceable.detail,
         serviceable.meaning, serviceable.recommendation]
    ).lower()
    assert "does not tell you" in text or "cannot" in text


def test_defect_copy_does_not_assert_which_defect_it_found():
    """The model cannot distinguish worn tread from cracking (exp006)."""
    copy = VERDICT_COPY[Verdict.DEFECT_SUSPECTED]
    text = f"{copy.detail} {copy.meaning}".lower()
    assert "cannot tell which" in text or "or" in text


def test_the_defect_label_does_not_grade_risk():
    """Risk depends on remaining tread depth, which this system cannot measure.

    "High risk" would assert a severity the evidence cannot support, so the label
    states what was found and the recommendation carries the urgency.
    """
    label = VERDICT_COPY[Verdict.DEFECT_SUSPECTED].label.lower()
    for graded in ("high risk", "dangerous", "critical", "severe", "unsafe"):
        assert graded not in label


def test_every_non_serviceable_verdict_points_at_a_physical_inspection():
    """A result that is not a clean pass must route the user to a real check."""
    for verdict in (Verdict.DEFECT_SUSPECTED, Verdict.INCONCLUSIVE):
        text = VERDICT_COPY[verdict].recommendation.lower()
        # Vocabulary widened deliberately: "looked at by a qualified tyre professional"
        # routes to a physical check just as "inspected by a fitter" does. The guard is
        # about the destination, not about a particular verb.
        routes_to_a_person = any(
            word in text
            for word in ("inspect", "fitter", "checked", "professional", "looked at", "garage")
        )
        assert routes_to_a_person, (
            f"{verdict.value} does not route the user to a physical check"
        )


def test_decision_serialises_to_json():
    json.dumps(_decide(0.8).as_dict())
