"""Measured feature evidence, as distinct from model interpretation.

Why this module is separate from ``explain.py``
-----------------------------------------------
The production model is an RBF-SVM, selected on measured evidence (exp007). A kernel
machine's decision cannot be decomposed into per-feature contributions, and
``explain.py`` correctly refuses to invent one.

Refusing to fabricate an explanation is not the same as having nothing to say. Two
genuinely different things can be reported, and the interface must not blur them:

**Measured evidence** — *what this tyre's surface actually measures, and how unusual
that is compared with the tyres the model was trained on.* This is a fact about the
photograph. It is true whatever the model does with it, and it stays true if the model
is replaced.

**Model interpretation** — *why the model decided what it decided.* Only available
exactly for a linear model. For the SVM it is unavailable, and the report says so.

This module produces the first. A percentile against the training distribution is the
honest unit: saying "edge structure is in the 8th percentile of tyres analysed" states
a measurement and its context without asserting it caused the verdict. Saying "low edge
structure caused this result" would be the fabrication.

The reference distribution is stored in the model artifact, because a percentile is
meaningless without the population it is taken against, and that population must be the
training set the model actually saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .explain import describe_feature

__all__ = ["FeatureEvidence", "EvidenceReport", "build_reference", "feature_evidence",
           "summarise_agreement", "MIN_DISCRIMINATIVE_AUC"]

#: A feature only earns a "resembles group X" note if it separates the classes on its
#: own at least this well. Below it, the class medians differ by too little for the
#: comparison to mean anything, and stating it anyway manufactures false precision.
MIN_DISCRIMINATIVE_AUC = 0.60

#: Features worth showing a user, in reporting order. Chosen for interpretability
#: rather than model importance - importance is a statement about the model, and this
#: module is deliberately not making statements about the model.
REPORTED_FEATURES = (
    "orientation_anisotropy",
    "orientation_coherence",
    "edge_density",
    "gradient_mean",
    "glcm_homogeneity_d1",
    "glcm_contrast_d1",
    "glcm_energy_d1",
    "spectral_slope",
)


def build_reference(
    values_by_feature: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Summarise the training distribution of each reported feature.

    Stored in the artifact so that a served prediction can place a measurement in
    context. Per-class medians are included so the report can say which way a
    measurement leans without claiming that lean drove the decision.

    Parameters
    ----------
    values_by_feature
        Training values, keyed by feature name.
    labels
        0/1 labels aligned with those values; 1 is the positive (defect) class.
    """
    reference: dict[str, dict[str, Any]] = {}
    for name, values in values_by_feature.items():
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size < 10:
            continue
        positive = values[(labels == 1) & np.isfinite(values)]
        negative = values[(labels == 0) & np.isfinite(values)]

        # How well this feature separates the classes on its own, as an AUC. Used to
        # decide whether saying "this value resembles group X" is worth saying at all.
        # Without it, a feature whose class medians differ by noise still produces a
        # confident-sounding resemblance claim, and eight such claims can contradict
        # the verdict for no reason other than chance.
        auc = None
        if positive.size >= 5 and negative.size >= 5:
            from scipy import stats as _stats
            u = _stats.mannwhitneyu(positive, negative, alternative="two-sided")
            auc = float(u.statistic / (positive.size * negative.size))

        reference[name] = {
            # A coarse percentile grid rather than every value: enough to place a
            # measurement, small enough to keep the artifact metadata readable.
            "percentiles": [
                float(np.percentile(finite, p)) for p in range(0, 101, 5)
            ],
            "median": float(np.median(finite)),
            "median_defect": float(np.median(positive)) if positive.size else None,
            "median_serviceable": float(np.median(negative)) if negative.size else None,
            "univariate_auc": auc,
            "n": int(finite.size),
        }
    return reference


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    description: str
    value: float
    #: Position in the training distribution, 0-100.
    percentile: float
    #: "typical", "low", "high", "very low", "very high".
    band: str
    #: Which class median this value sits closer to. Descriptive only.
    resembles: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "description": self.description,
            "value": round(self.value, 5),
            "percentile": round(self.percentile, 1),
            "band": self.band,
            "resembles": self.resembles,
        }


@dataclass(frozen=True)
class EvidenceReport:
    measurements: list[FeatureEvidence]
    #: Diagnostics computed and reported but deliberately not used by the model.
    diagnostics: dict[str, float]
    note: str
    #: One sentence on whether the individual measurements point the same way as the
    #: overall assessment. Set by ``summarise_agreement`` once the verdict is known.
    agreement: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "measurements": [m.as_dict() for m in self.measurements],
            "diagnostics": {k: round(v, 5) for k, v in self.diagnostics.items()},
            "agreement": self.agreement,
            "note": self.note,
        }


def _percentile_of(value: float, percentiles: list[float]) -> float:
    """Where ``value`` falls in a 0-100 step-5 percentile grid, interpolated."""
    grid = np.asarray(percentiles, dtype=np.float64)
    points = np.arange(0, 101, 5, dtype=np.float64)
    if value <= grid[0]:
        return 0.0
    if value >= grid[-1]:
        return 100.0
    return float(np.interp(value, grid, points))


def _band(percentile: float) -> str:
    if percentile < 10:
        return "very low"
    if percentile < 30:
        return "low"
    if percentile > 90:
        return "very high"
    if percentile > 70:
        return "high"
    return "typical"


def feature_evidence(
    features: dict[str, float],
    reference: dict[str, dict[str, Any]],
    *,
    diagnostics: dict[str, float] | None = None,
) -> EvidenceReport:
    """Place this image's measurements against the training distribution.

    Every statement produced here is descriptive. Nothing claims causation, because
    for the production model no exact causal decomposition exists.
    """
    measurements: list[FeatureEvidence] = []
    for name in REPORTED_FEATURES:
        if name not in features or name not in reference:
            continue
        value = float(features[name])
        stats = reference[name]
        percentile = _percentile_of(value, stats["percentiles"])

        resembles = None
        median_defect = stats.get("median_defect")
        median_serviceable = stats.get("median_serviceable")
        auc = stats.get("univariate_auc")
        discriminative = auc is not None and abs(auc - 0.5) >= (MIN_DISCRIMINATIVE_AUC - 0.5)
        if discriminative and median_defect is not None and median_serviceable is not None:
            if abs(median_defect - median_serviceable) > 1e-12:
                nearer_defect = abs(value - median_defect) < abs(value - median_serviceable)
                resembles = "tyres with visible defects" if nearer_defect else "tyres in good condition"

        measurements.append(FeatureEvidence(
            feature=name,
            description=describe_feature(name),
            value=value,
            percentile=percentile,
            band=_band(percentile),
            resembles=resembles,
        ))

    return EvidenceReport(
        measurements=measurements,
        diagnostics=diagnostics or {},
        note=(
            "These are measurements taken from your photograph, shown against the range "
            "seen across the tyres this model was trained on. They describe the surface; "
            "they are not a breakdown of how the model reached its result. The model in "
            "use is non-linear, so an exact per-measurement breakdown is not available, "
            "and none is guessed at. Individual measurements can each look ordinary "
            "while their combination does not, which is precisely why the assessment "
            "uses all of them together rather than any one on its own - so a result may "
            "legitimately differ from what a single line below appears to suggest."
        ),
    )


def summarise_agreement(report: EvidenceReport, *, verdict: str) -> EvidenceReport:
    """State plainly how the individual measurements relate to the overall assessment.

    A multivariate model can be confident while every single measurement looks
    ordinary, because what distinguishes the classes is the combination rather than any
    one value. Showing five lines that all read "resembles tyres in good condition"
    beside a verdict of "possible wear or damage" looks like a contradiction and erodes
    trust in a report whose whole purpose is to be checkable. So it is said out loud.

    ``verdict`` is the decision's own value rather than a boolean. An earlier version
    took ``defect_suspected: bool``, which silently folded the *inconclusive* verdict in
    with the serviceable one and produced the sentence "point the same way as the
    overall assessment of good condition" underneath a result that said "Not
    conclusive". Real-device testing caught it. An inconclusive result has no side for
    measurements to agree with, and saying so is the useful thing to report.
    """
    scored = [m for m in report.measurements if m.resembles is not None]
    if not scored:
        return report

    total = len(scored)
    towards_defect = sum(1 for m in scored if m.resembles == "tyres with visible defects")
    towards_good = total - towards_defect

    if verdict == "inconclusive":
        # No side to agree with. Describe the split, which is why it is inconclusive.
        if towards_defect and towards_good:
            agreement = (
                f"The measurements are split: {towards_defect} of {total} lean towards "
                f"wear or damage and {towards_good} towards good condition. That "
                "disagreement is why the result is inconclusive rather than a call "
                "either way."
            )
        else:
            leaning = "good condition" if towards_good == total else "wear or damage"
            agreement = (
                f"All {total} measurements lean towards {leaning} individually, but not "
                "strongly enough in combination for a confident call. The assessment "
                "uses them together, and together they sit near the boundary."
            )
        return EvidenceReport(
            measurements=report.measurements,
            diagnostics=report.diagnostics,
            note=report.note,
            agreement=agreement,
        )

    defect_suspected = verdict == "defect_suspected"
    verdict_word = "possible wear or damage" if defect_suspected else "good condition"
    agreeing = towards_defect if defect_suspected else towards_good

    if agreeing >= total * 0.6:
        agreement = (
            f"{agreeing} of {total} individual measurements point the same way as the "
            f"overall assessment of {verdict_word}."
        )
    else:
        agreement = (
            f"Only {agreeing} of {total} individual measurements point towards "
            f"{verdict_word} on their own, yet the combined assessment does. That is "
            "expected behaviour rather than a contradiction: the classes are separated "
            "by the combination of measurements, not by any single one. It does mean "
            "this result rests on no single obvious cue, which is worth weighing when "
            "deciding whether to have the tyre looked at."
        )

    return EvidenceReport(
        measurements=report.measurements,
        diagnostics=report.diagnostics,
        note=report.note,
        agreement=agreement,
    )
