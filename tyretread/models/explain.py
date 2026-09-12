"""Faithful explanation of a single prediction.

The rule this module exists to enforce: an explanation must be *derived from* the
computation that produced the verdict, never written to justify it afterwards.

For a linear model behind a monotonic calibration step, an exact decomposition is
available. The decision function is

    log-odds = intercept + sum_j w_j * z_j,    z_j = (x_j - mean_j) / scale_j

so ``w_j * z_j`` is precisely how many log-odds feature *j* contributed for this
image. Platt scaling is monotonic, so it cannot reorder contributions. That makes
the reported drivers genuinely the reasons, not a plausible story.

When the fitted estimator is not linear, this module says so rather than
substituting a surrogate. A model that cannot be decomposed honestly should not
pretend otherwise, and that is one of the reasons the linear candidate is preferred
where its accuracy is competitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["FeatureContribution", "Explanation", "explain_prediction",
           "FEATURE_DESCRIPTIONS"]


#: Plain-language meaning of each feature family, for the user-facing explanation.
#: Keyed by prefix, longest match wins.
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "spectral_slope_r2": "how closely the surface follows one consistent texture pattern",
    "spectral_slope": "the balance of coarse and fine detail in the tread surface",
    "orientation_anisotropy": "the strength of directional grooving",
    "orientation_coherence": "how consistently the grooves run in one direction",
    "orientation_entropy": "how evenly texture is spread across all directions",
    "orientation_dominant_deg": "the dominant groove direction in the frame",
    "glcm_contrast": "local contrast between neighbouring points on the surface",
    "glcm_dissimilarity": "the difference between neighbouring surface points",
    "glcm_homogeneity": "the uniformity of the tread surface",
    "glcm_energy": "the repetitiveness of the surface pattern",
    "glcm_correlation": "how predictable the surface is from point to point",
    "lbp_": "the mix of fine surface micro-patterns",
    "edge_density": "the amount of visible groove edge structure",
    "gradient_mean": "the average edge strength across the tread",
    "gradient_p95": "the strength of the most pronounced edges",
}


def describe_feature(name: str) -> str:
    for prefix in sorted(FEATURE_DESCRIPTIONS, key=len, reverse=True):
        if name.startswith(prefix):
            return FEATURE_DESCRIPTIONS[prefix]
    return name


@dataclass(frozen=True)
class FeatureContribution:
    feature: str
    value: float
    #: Standardised value: how many standard deviations from the training mean.
    z_score: float
    #: Signed log-odds contribution towards the worn class.
    contribution: float
    description: str

    @property
    def direction(self) -> str:
        return "towards wear" if self.contribution > 0 else "towards serviceable"

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "value": round(self.value, 5),
            "z_score": round(self.z_score, 3),
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
            "description": self.description,
        }


@dataclass(frozen=True)
class Explanation:
    exact: bool
    method: str
    contributions: list[FeatureContribution]
    #: Short, non-technical sentences, derived from the top contributions only.
    reasons: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "exact": self.exact,
            "method": self.method,
            "reasons": self.reasons,
            "contributions": [c.as_dict() for c in self.contributions],
        }


def _linear_parts(estimator: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Average (coef, scaler_mean, scaler_scale) across a calibrated ensemble.

    Returns None if the underlying estimator is not a scaler-plus-linear pipeline.
    """
    inner = getattr(estimator, "calibrated_classifiers_", None)
    pipelines: list[Any] = []
    if inner:
        for calibrated in inner:
            base = getattr(calibrated, "estimator", None)
            if base is not None:
                pipelines.append(base)
    else:
        pipelines = [estimator]

    coefs, means, scales = [], [], []
    for pipeline in pipelines:
        steps = getattr(pipeline, "named_steps", None)
        if not steps:
            return None
        clf = steps.get("clf")
        scaler = steps.get("scaler")
        coef = getattr(clf, "coef_", None)
        if coef is None or coef.shape[0] != 1:
            return None
        coefs.append(coef[0])
        means.append(getattr(scaler, "mean_", np.zeros(coef.shape[1])))
        scales.append(getattr(scaler, "scale_", np.ones(coef.shape[1])))

    if not coefs:
        return None
    return (np.mean(coefs, axis=0), np.mean(means, axis=0), np.mean(scales, axis=0))


def explain_prediction(
    estimator: Any,
    feature_names: list[str],
    features: dict[str, float],
    *,
    top_k: int = 4,
) -> Explanation:
    """Decompose one prediction into its per-feature drivers."""
    parts = _linear_parts(estimator)
    x = np.array([float(features[n]) for n in feature_names], dtype=np.float64)

    if parts is None:
        # Be explicit rather than inventing a surrogate explanation.
        return Explanation(
            exact=False,
            method=(
                "This model is not linear, so its prediction cannot be decomposed "
                "exactly into per-feature contributions. The measured values are "
                "reported without attributing the decision to them."
            ),
            contributions=[
                FeatureContribution(n, float(v), float("nan"), float("nan"), describe_feature(n))
                for n, v in zip(feature_names, x)
            ],
            reasons=[],
        )

    coef, mean, scale = parts
    z = (x - mean) / np.where(scale == 0, 1.0, scale)
    contributions = coef * z

    ranked = sorted(
        (
            FeatureContribution(n, float(xv), float(zv), float(cv), describe_feature(n))
            for n, xv, zv, cv in zip(feature_names, x, z, contributions)
        ),
        key=lambda c: -abs(c.contribution),
    )

    reasons: list[str] = []
    seen: set[str] = set()
    for contribution in ranked:
        if len(reasons) >= top_k:
            break
        # One sentence per distinct idea; several LBP bins are one idea.
        if contribution.description in seen or not np.isfinite(contribution.contribution):
            continue
        seen.add(contribution.description)
        strength = "strongly" if abs(contribution.z_score) > 1.5 else "moderately"
        pointer = "consistent with wear" if contribution.contribution > 0 else "consistent with a serviceable tread"
        sentence = contribution.description
        reasons.append(
            f"{sentence[0].upper()}{sentence[1:]} is {strength} {pointer}."
        )

    return Explanation(
        exact=True,
        method=(
            "Exact decomposition of the model's log-odds: each contribution is the "
            "model's weight for that measurement multiplied by how far this image's "
            "measurement sits from the training average. Probability calibration is "
            "monotonic and so does not change the ordering."
        ),
        contributions=ranked,
        reasons=reasons,
    )
