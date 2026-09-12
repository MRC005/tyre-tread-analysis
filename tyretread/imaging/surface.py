"""Which part of the tyre is this photograph actually showing?

The problem this solves
-----------------------
Hand-labelling a random sample of 120 images that pass the quality gate found that
**48% of them are not clean tread**: 41% are sidewall close-ups and 7% are mixed
(`experiments/exp011_tread_vs_sidewall`). The ROI stage cannot tell the difference - it
returns a plausible "tread band" from a photograph of a sidewall - so a user who
photographs the side of their tyre receives an assessment worded as though it were
about the tread.

Why this is a reporting problem, not a rejection problem
--------------------------------------------------------
The obvious response is to refuse non-tread images. Measurement argues against it. The
production model performs comparably on both surfaces, and the surface a photograph
shows is **not** associated with its condition label (chi-square p = 0.80), so
sidewall images are not noise - the system genuinely detects cracking and perished
rubber there, which is useful.

Meanwhile the detector below reaches only AUC 0.806 [0.718, 0.884]. At a threshold
catching three-quarters of non-tread images it falsely rejects one genuine tread
photograph in five. Refusing on that basis would discard working functionality to
enforce a distinction the system cannot make reliably.

So the surface is **reported, not enforced**. The verdict says which surface it
believes it assessed, and says when it does not know. That is the honest version of
"do not pretend the system can identify tread".

Model provenance and limits
---------------------------
A balanced logistic regression over eight interpretable structure features, fitted on
**120 hand-labelled images** - a small sample, and the wide confidence interval above
is the direct consequence. Its coefficients are stored in the model artifact as plain
JSON rather than a second binary, so they can be read without unpickling anything.

It should be refitted on a larger annotated sample before it is relied on more heavily
than it is here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

__all__ = ["Surface", "SurfaceAssessment", "SURFACE_FEATURES", "fit_surface_detector",
           "assess_surface", "SURFACE_COPY"]

#: Features the detector reads. Chosen for interpretability and measured
#: discrimination: the power-law goodness-of-fit separates best (AUC 0.747 on its own),
#: because tread carries periodic groove structure that a single power law describes
#: badly, while smoother sidewall rubber fits one well.
SURFACE_FEATURES = (
    "spectral_slope_r2",
    "spectral_slope",
    "glcm_correlation_d1",
    "glcm_correlation_d2",
    "orientation_anisotropy",
    "orientation_coherence",
    "edge_density",
    "gradient_mean",
)

#: Below this the image is reported as probably not tread. Chosen from the measured
#: operating curve at roughly 5% false rejection of genuine tread, because the cost of
#: wrongly telling a user "this may not be tread" is small while the cost of silently
#: assessing a sidewall as tread is the failure this exists to prevent.
NOT_TREAD_BELOW = 0.35
#: Above this the image is reported as tread. Between the two, the answer is "unclear".
TREAD_ABOVE = 0.60


class Surface(str, Enum):
    TREAD = "tread"
    NOT_TREAD = "sidewall_or_shoulder"
    UNCLEAR = "unclear"


SURFACE_COPY: dict[Surface, str] = {
    Surface.TREAD: "This looks like the tread surface.",
    Surface.NOT_TREAD: (
        "This looks like the sidewall or shoulder rather than the tread. The "
        "assessment still applies to the rubber in the photograph, but it says nothing "
        "about how much tread is left."
    ),
    Surface.UNCLEAR: (
        "It is not clear whether this shows the tread or the sidewall, so treat the "
        "result as being about the rubber in the photograph rather than about tread "
        "specifically. Photographing the tread straight on would give a clearer answer."
    ),
}


@dataclass(frozen=True)
class SurfaceAssessment:
    surface: Surface
    #: Probability the image shows tread, from the stored detector.
    tread_probability: float
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface.value,
            "tread_probability": round(self.tread_probability, 4),
            "note": self.note,
        }


def fit_surface_detector(
    X: np.ndarray, y: np.ndarray, feature_names: tuple[str, ...] = SURFACE_FEATURES
) -> dict[str, Any]:
    """Fit the detector and return it as plain JSON-serialisable parameters.

    Stored as coefficients rather than a pickled estimator so the artifact carries no
    second binary and the model can be inspected by eye.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ]).fit(X, y)

    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    return {
        "feature_names": list(feature_names),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "n_training_samples": int(len(y)),
        "not_tread_below": NOT_TREAD_BELOW,
        "tread_above": TREAD_ABOVE,
    }


def assess_surface(
    features: dict[str, float], detector: dict[str, Any] | None
) -> SurfaceAssessment | None:
    """Decide which surface the photograph shows.

    Returns ``None`` when no detector is available, so an older artifact without one
    simply omits the section rather than failing.
    """
    if not detector:
        return None

    names = detector["feature_names"]
    missing = [n for n in names if n not in features]
    if missing:
        return None

    x = np.array([float(features[n]) for n in names], dtype=np.float64)
    mean = np.array(detector["mean"], dtype=np.float64)
    scale = np.array(detector["scale"], dtype=np.float64)
    coef = np.array(detector["coef"], dtype=np.float64)

    z = (x - mean) / np.where(scale == 0, 1.0, scale)
    logit = float(np.dot(coef, z) + detector["intercept"])
    probability = float(1.0 / (1.0 + np.exp(-logit)))

    low = detector.get("not_tread_below", NOT_TREAD_BELOW)
    high = detector.get("tread_above", TREAD_ABOVE)
    if probability < low:
        surface = Surface.NOT_TREAD
    elif probability >= high:
        surface = Surface.TREAD
    else:
        surface = Surface.UNCLEAR

    return SurfaceAssessment(
        surface=surface, tread_probability=probability, note=SURFACE_COPY[surface]
    )
