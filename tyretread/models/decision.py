"""Turning a probability into a verdict.

Scope. The Mendeley training set labels *tyre condition*: the positive class is mostly
sidewall cracking, splits, perished rubber and worn tread, the negative class is tyres
in good condition (exp006). It carries no tread-depth measurements, and exp008 showed a
model trained on it does not transfer to a dataset labelled for tread wear (0.665
balanced accuracy against 0.769 within-dataset). The two datasets are not measuring the
same property.

So the scope is visible condition screening: does this surface resemble tyres with
visible wear or damage, or tyres in good condition. It cannot measure tread depth,
cannot report millimetres, and cannot say whether a tyre is legal or roadworthy. The
wording below is written to that scope and ``test_decision.py`` enforces it.

Why this is not a three-class model. The original system reported Safe / Warning /
Dangerous, but those came from a rule applied on top of a *binary* classifier's
in-sample predictions, so the three-way output was never validated and the 74.8%
headline was a binary result (docs/AUDIT.md 3.4). No dataset available here carries
three-level expert labels or depth in millimetres, so a real three-class model cannot be
trained or checked.

What the data does support is a binary distinction plus a calibrated probability, so the
four verdicts below are all derived from that one probability:

``UNABLE_TO_ASSESS``
    The quality gate refused the image; no probability is computed at all.
``INCONCLUSIVE``
    Too close to the decision threshold to act on. Saying so beats presenting a
    coin-flip as a safety verdict.
``LIKELY_SERVICEABLE`` / ``DEFECT_SUSPECTED``
    Confident calls either side of the band, shown as **Healthy** and **Defect
    detected**.

The positive label is not "High risk" on purpose: risk depends mostly on remaining tread
depth, which this system cannot measure and for which there is no ground truth in any
dataset here. "Defect detected" states what was found - surface characteristics matching
the damaged group of the training data - and the recommendation carries the urgency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["Verdict", "Decision", "decide", "VERDICT_COPY"]


class Verdict(str, Enum):
    UNABLE_TO_ASSESS = "unable_to_assess"
    INCONCLUSIVE = "inconclusive"
    LIKELY_SERVICEABLE = "likely_serviceable"
    DEFECT_SUSPECTED = "defect_suspected"


@dataclass(frozen=True)
class VerdictCopy:
    #: Short status label, e.g. "Healthy". The taxonomy the user sees.
    label: str
    headline: str
    #: What was found, in one sentence.
    detail: str
    #: What it means in practice - the "so what" a driver actually needs.
    meaning: str
    #: What to do next.
    recommendation: str
    #: Semantic severity for the interface: "ok", "caution", "alert", "unknown".
    severity: str


VERDICT_COPY: dict[Verdict, VerdictCopy] = {
    Verdict.LIKELY_SERVICEABLE: VerdictCopy(
        label="Healthy",
        headline="No visible defects found",
        detail=(
            "The surface in this photograph looks like tyres in good condition — even "
            "tread texture, no visible cracking or perished rubber."
        ),
        meaning=(
            "Nothing in this photograph suggests a problem. This looks at the surface "
            "only, so it does not tell you how much tread depth is left."
        ),
        recommendation=(
            "Keep checking periodically, and have tread depth measured at your next "
            "service."
        ),
        severity="ok",
    ),
    Verdict.DEFECT_SUSPECTED: VerdictCopy(
        label="Defect detected",
        headline="Visible signs of wear or damage",
        detail=(
            "The surface in this photograph shows characteristics associated with worn "
            "tread, cracking or perished rubber."
        ),
        meaning=(
            "Something on this tyre's surface resembles tyres that are worn or "
            "damaged. This screening cannot tell which of those it is seeing, or how "
            "far it has progressed."
        ),
        recommendation=(
            "Have this tyre looked at by a qualified tyre professional before long "
            "journeys."
        ),
        severity="alert",
    ),
    Verdict.INCONCLUSIVE: VerdictCopy(
        label="Attention recommended",
        headline="Not conclusive",
        detail=(
            "This tyre sits close to the boundary between the two groups this system "
            "was built to tell apart."
        ),
        meaning=(
            "A borderline surface, or a photograph that does not show quite enough. "
            "The system is not confident either way, and says so rather than guessing."
        ),
        recommendation=(
            "Try another photograph from a slightly different angle. If it stays "
            "inconclusive, have the tyre checked."
        ),
        severity="caution",
    ),
    Verdict.UNABLE_TO_ASSESS: VerdictCopy(
        label="Unable to assess",
        headline="This photo can't be assessed",
        detail=(
            "The photograph does not meet the minimum quality needed for analysis, so "
            "no assessment was attempted."
        ),
        meaning=(
            "This says nothing about the tyre — only about the photograph. A better "
            "photo will usually give a result."
        ),
        recommendation="Retake the photograph following the guidance shown.",
        severity="unknown",
    ),
}


@dataclass(frozen=True)
class Decision:
    verdict: Verdict
    #: Calibrated probability that the tyre has a visible defect. None if unassessable.
    probability_defect: float | None
    #: How far the probability sits from the abstention band, scaled to [0, 1].
    #: Reported rather than the raw probability because "83% defective" invites being
    #: read as "83% of the tread is gone".
    confidence: float | None
    threshold: float
    abstain_band: float

    @property
    def copy(self) -> VerdictCopy:
        return VERDICT_COPY[self.verdict]

    def as_dict(self) -> dict[str, object]:
        return {
            "verdict": self.verdict.value,
            "severity": self.copy.severity,
            "label": self.copy.label,
            "headline": self.copy.headline,
            "detail": self.copy.detail,
            "meaning": self.copy.meaning,
            "recommendation": self.copy.recommendation,
            "probability_defect": (
                round(self.probability_defect, 4) if self.probability_defect is not None else None
            ),
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "decision_threshold": self.threshold,
            "abstain_band": self.abstain_band,
        }


def decide(
    probability_defect: float | None,
    *,
    threshold: float,
    abstain_band: float,
) -> Decision:
    """Map a calibrated probability onto a verdict.

    Parameters
    ----------
    probability_defect
        ``None`` when the quality gate refused the image, which short-circuits to
        ``UNABLE_TO_ASSESS`` without any model being consulted.
    threshold
        Probability above which a defect is flagged. Chosen by experiment against
        recall on the defect class, not left at 0.5, because missing a defective tyre
        and over-flagging a good one are not equally costly.
    abstain_band
        Half-width of the neutral zone around ``threshold``. Within it the system
        returns ``INCONCLUSIVE``.
    """
    if probability_defect is None:
        return Decision(Verdict.UNABLE_TO_ASSESS, None, None, threshold, abstain_band)

    p = float(probability_defect)
    lower, upper = threshold - abstain_band, threshold + abstain_band

    if lower <= p <= upper:
        return Decision(Verdict.INCONCLUSIVE, p, 0.0, threshold, abstain_band)

    if p > upper:
        verdict = Verdict.DEFECT_SUSPECTED
        span = max(1.0 - upper, 1e-9)
        confidence = (p - upper) / span
    else:
        verdict = Verdict.LIKELY_SERVICEABLE
        span = max(lower, 1e-9)
        confidence = (lower - p) / span

    return Decision(verdict, p, min(1.0, max(0.0, confidence)), threshold, abstain_band)
