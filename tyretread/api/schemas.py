"""Response shapes for the API.

Declared explicitly rather than returning loose dicts so that the contract is
visible, documented in the generated OpenAPI schema, and testable. The frontend
depends on these field names.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QualityCheckOut(BaseModel):
    name: str
    passed: bool
    value: float
    threshold: float


class ImageQualityOut(BaseModel):
    usable: bool
    summary: Literal["Good", "Acceptable", "Unusable"]
    issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    advice: list[str] = Field(
        default_factory=list,
        description="Plain-language retake guidance, one entry per blocking issue.",
    )
    warning_advice: list[str] = Field(
        default_factory=list,
        description=(
            "Guidance for non-blocking caveats. A photo can clear the gate and still "
            "be the reason a result was inconclusive."
        ),
    )
    metrics: dict[str, float] = Field(default_factory=dict)
    checks: list[QualityCheckOut] = Field(default_factory=list)


class ResultOut(BaseModel):
    verdict: Literal["likely_serviceable", "defect_suspected", "inconclusive", "unable_to_assess"]
    severity: Literal["ok", "caution", "alert", "unknown"]
    label: str = Field(
        description='Short status label: "Healthy", "Defect detected", '
        '"Attention recommended" or "Unable to assess".'
    )
    headline: str
    detail: str
    meaning: str = Field(
        description="What the result means in practice, for a non-technical reader."
    )
    recommendation: str
    probability_defect: float | None = Field(
        None,
        description=(
            "Calibrated probability that the tyre shows a visible defect - worn tread, "
            "cracking or perished rubber. Null when the image could not be assessed. "
            "This is not a tread-depth estimate and has no units of depth."
        ),
    )
    confidence: float | None = Field(
        None,
        description=(
            "How far the probability sits beyond the abstention band, scaled to "
            "0-1. Reported instead of the raw probability because a percentage is "
            "easily misread as a proportion of tread remaining."
        ),
    )
    decision_threshold: float
    abstain_band: float


class ContributionOut(BaseModel):
    feature: str
    value: float
    z_score: float
    contribution: float
    direction: str
    description: str


class ExplanationOut(BaseModel):
    """Model interpretation: why the model decided as it did.

    Exact only for a linear model. The production model is an RBF-SVM, selected on
    measured evidence, which cannot be decomposed exactly - in that case ``exact`` is
    false and the lists are empty rather than filled with a guess. Use
    ``evidence_features`` for what was actually measured.
    """

    exact: bool = Field(
        description=(
            "True when the contributions are an exact decomposition of the model's "
            "decision. False means no causal breakdown is available and none was "
            "invented."
        )
    )
    method: str
    reasons: list[str]
    contributions: list[ContributionOut]


class FeatureEvidenceOut(BaseModel):
    feature: str
    description: str
    value: float
    percentile: float = Field(
        description="Where this measurement falls in the training distribution, 0-100."
    )
    band: Literal["very low", "low", "typical", "high", "very high"]
    resembles: str | None = Field(
        None,
        description=(
            "Which training group this value sits closer to. Descriptive only - it "
            "does not assert that this measurement caused the verdict."
        ),
    )


class EvidenceFeaturesOut(BaseModel):
    """Measured evidence: what the surface measures, and how unusual that is.

    Distinct from ``explanation``. These are facts about the photograph, true
    regardless of which model is served.
    """

    measurements: list[FeatureEvidenceOut]
    agreement: str | None = Field(
        None,
        description=(
            "Whether the individual measurements point the same way as the overall "
            "assessment. A disagreement is surfaced rather than hidden."
        ),
    )
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Values computed and reported but deliberately not used by the model - "
            "legacy TSCI among them."
        ),
    )
    note: str


class EvidenceOut(BaseModel):
    """Base64 PNG data URIs of the intermediate stages.

    Returned inline rather than from a second endpoint because the service stores no
    uploads: there would be nothing for a follow-up request to fetch.
    """

    original: str | None = None
    enhanced: str | None = None
    roi: str | None = None
    edges: str | None = None
    spectrum: str | None = None


class SurfaceOut(BaseModel):
    """Which part of the tyre the photograph appears to show.

    Reported rather than enforced: the detector reaches AUC 0.797 [0.698, 0.878] on
    116 hand-labelled images, which is not reliable enough to refuse an image, and the
    model performs comparably on both surfaces.
    """

    surface: Literal["tread", "sidewall_or_shoulder", "unclear"]
    tread_probability: float
    note: str


class ModelMetricsOut(BaseModel):
    """A compact summary of how the served model performed.

    Deliberately not the full experiment record. The artifact's ``metrics`` field
    holds the complete model ranking, confusion matrices and data-hygiene counts -
    3.3 KB of internal experiment detail that was previously serialised onto every
    inspection response. A phone on mobile data should not pay for that, and a
    client should not depend on the shape of an internal record. The full record
    stays in the artifact and in experiments/.
    """

    balanced_accuracy: float | None = None
    balanced_accuracy_std: float | None = None
    roc_auc: float | None = None
    brier_score: float | None = None
    expected_calibration_error: float | None = None
    abstention_rate: float | None = None
    balanced_accuracy_on_decided: float | None = None
    quality_gate_pass_rate: float | None = None


class ModelOut(BaseModel):
    id: str
    artifact_format_version: int
    trained_on: str | None = None
    n_training_samples: int | None = None
    validation: str | None = None
    metrics: ModelMetricsOut | None = None


class InspectionOut(BaseModel):
    result: ResultOut
    image_quality: ImageQualityOut
    measurements: dict[str, float]
    model: ModelOut
    disclaimer: str
    surface: SurfaceOut | None = None
    explanation: ExplanationOut | None = None
    evidence_features: EvidenceFeaturesOut | None = None
    evidence: EvidenceOut | None = None
    features: dict[str, float] | None = None
    elapsed_ms: float | None = None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_loaded: bool
    model_id: str | None = None
    detail: str | None = None


class ErrorOut(BaseModel):
    error: str = Field(description="Stable machine-readable code.")
    message: str = Field(description="Human-readable explanation.")
    detail: dict[str, Any] | None = None
