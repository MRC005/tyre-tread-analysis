"""The production inference path.

One function, ``inspect_image``, is the only way a verdict is ever produced. The
API calls it, the CLI calls it, and the tests call it. The original project had two
divergent paths - a threshold rule in ``main.py`` and an unsaved SVM in the training
script - which disagreed with each other and neither of which matched the reported
metrics (docs/AUDIT.md 3.10). Collapsing them into one function is what makes
"the model we evaluated is the model we serve" a structural property rather than a
promise.

Order of operations matters and is enforced here:

    decode -> preprocess -> locate tread -> quality gate
        -> (refuse)  or  -> normalise scale -> features -> probability
        -> calibrated verdict with abstention -> explanation

The quality gate sits before the model, so it is not possible to obtain a
prediction for an image the gate rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CONFIG, Config
from .features.extract import Extraction, extract_features
from .imaging.quality import QualityReport, QualityThresholds
from .imaging.surface import Surface, SurfaceAssessment, assess_surface
from .models.artifact import ModelArtifact
from .models.decision import Decision, Verdict, decide
from .models.evidence_features import EvidenceReport, feature_evidence, summarise_agreement
from .models.explain import Explanation, explain_prediction

__all__ = ["Inspection", "inspect_image"]


@dataclass
class Inspection:
    """The complete result of assessing one photograph."""

    decision: Decision
    quality: QualityReport
    #: Which part of the tyre the photograph appears to show. Reported, never used to
    #: reject - see tyretread.imaging.surface.
    surface: SurfaceAssessment | None
    #: Measured feature evidence: what this surface measures, in context. Always a
    #: statement about the photograph, never about the model's reasoning.
    evidence: EvidenceReport | None
    #: Model interpretation: why the model decided as it did. Only exact for a linear
    #: model; ``None`` when the served model cannot be decomposed honestly.
    explanation: Explanation | None
    features: dict[str, float]
    #: Raw values of the reported measurements, for the technical section.
    measurements: dict[str, float]
    model_id: str
    model_version: int
    #: Retained so the API can render visual evidence without re-running anything.
    extraction: Extraction

    @property
    def usable(self) -> bool:
        return self.decision.verdict is not Verdict.UNABLE_TO_ASSESS

    def as_dict(self, *, include_features: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "result": self.decision.as_dict(),
            "image_quality": self.quality.as_dict(),
            "measurements": {k: round(v, 5) for k, v in self.measurements.items()},
            "model": {
                "id": self.model_id,
                "artifact_format_version": self.model_version,
            },
            "disclaimer": (
                "This is an assistive visual screening tool. It compares the surface "
                "texture in a photograph against tyres with and without visible "
                "defects. It cannot measure tread depth, cannot distinguish worn tread "
                "from cracking or perished rubber, and cannot determine whether a tyre "
                "is legal or roadworthy. It is not a certified inspection and carries "
                "no regulatory approval. Any tyre flagged as possibly defective, and "
                "any inconclusive result, should be physically inspected by a "
                "qualified professional."
            ),
        }
        if self.surface is not None:
            payload["surface"] = self.surface.as_dict()
        if self.evidence is not None:
            payload["evidence_features"] = self.evidence.as_dict()
        if self.explanation is not None and self.explanation.exact:
            payload["explanation"] = self.explanation.as_dict()
        elif self.explanation is not None:
            # State plainly that no causal breakdown is available rather than
            # omitting the section, which would read as an oversight.
            payload["explanation"] = {
                "exact": False,
                "method": self.explanation.method,
                "reasons": [],
                "contributions": [],
            }
        if include_features:
            payload["features"] = {k: round(v, 6) for k, v in self.features.items()}
        return payload


#: The subset of features worth surfacing to a user, in reporting order. Chosen for
#: interpretability, not by model importance - the explanation covers importance.
HEADLINE_MEASUREMENTS = (
    "orientation_anisotropy",
    "orientation_coherence",
    "spectral_slope",
    "edge_density",
    "gradient_mean",
    "glcm_homogeneity_d1",
    "glcm_contrast_d1",
)


def inspect_image(
    bgr: np.ndarray,
    artifact: ModelArtifact,
    *,
    config: Config | None = None,
    thresholds: QualityThresholds | None = None,
) -> Inspection:
    """Assess one decoded photograph.

    Parameters
    ----------
    bgr
        A decoded, EXIF-corrected image from ``tyretread.imaging.io``.
    artifact
        A loaded model artifact. Passed in rather than loaded here so a server
        loads it once at start-up instead of per request.
    """
    config = config or CONFIG
    metadata = artifact.metadata

    if thresholds is None and metadata.quality_thresholds:
        # Serve with the same gate the model was validated under; a looser gate at
        # inference would feed the model images unlike anything it was tested on.
        thresholds = QualityThresholds(**metadata.quality_thresholds)

    # Diagnostics declared by the artifact are computed even though the model does
    # not consume them, so the report can show them (exp009 keeps TSCI here).
    wants_tsci = "legacy_tsci" in metadata.diagnostic_features
    extraction = extract_features(
        bgr, config=config, thresholds=thresholds, include_legacy_tsci=wants_tsci
    )

    if not extraction.usable:
        return Inspection(
            decision=decide(
                None,
                threshold=metadata.decision_threshold,
                abstain_band=metadata.abstain_band,
            ),
            quality=extraction.quality,
            surface=None,
            evidence=None,
            explanation=None,
            features={},
            measurements={},
            model_id=metadata.model_id,
            model_version=metadata.format_version,
            extraction=extraction,
        )

    probability = artifact.probability_defect(extraction.features)
    decision = decide(
        probability,
        threshold=metadata.decision_threshold,
        abstain_band=metadata.abstain_band,
    )
    explanation = explain_prediction(
        artifact.estimator, metadata.feature_names, extraction.features
    )

    diagnostics = {
        name: extraction.features[name]
        for name in metadata.diagnostic_features
        if name in extraction.features
    }
    diagnostics["oversampling"] = extraction.oversampling
    surface = assess_surface(extraction.features, metadata.surface_detector)

    evidence = None
    if metadata.feature_reference:
        evidence = summarise_agreement(
            feature_evidence(
                extraction.features, metadata.feature_reference, diagnostics=diagnostics
            ),
            verdict=decision.verdict.value,
        )

    measurements = {
        name: extraction.features[name]
        for name in HEADLINE_MEASUREMENTS
        if name in extraction.features
    }
    measurements["oversampling"] = extraction.oversampling

    return Inspection(
        decision=decision,
        quality=extraction.quality,
        surface=surface,
        evidence=evidence,
        explanation=explanation,
        features=extraction.features,
        measurements=measurements,
        model_id=metadata.model_id,
        model_version=metadata.format_version,
        extraction=extraction,
    )
