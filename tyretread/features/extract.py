"""The single feature-extraction path shared by training and inference.

The original project had two divergent inference paths: ``main.py`` classified from
hardcoded TSCI thresholds, while ``train_and_evaluate.py`` trained an SVM whose
weights were never saved. The two disagreed, and neither was what a user would have
been served (docs/AUDIT.md 3.10). There is now one function, used by both, so the
model that is evaluated is by construction the model that runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CONFIG, Config
from ..imaging.preprocess import Preprocessed, normalise_scale, oversampling_factor, preprocess
from ..imaging.localise import subject_coverage as measure_subject_coverage
from ..imaging.quality import QualityReport, QualityThresholds, assess_quality
from ..imaging.roi import RoiResult, extract_roi
from .spectral import legacy_tsci, spectral_features
from .texture import texture_features

__all__ = ["Extraction", "extract_features", "feature_names"]


@dataclass(frozen=True)
class Extraction:
    """Everything one image yields: features, quality verdict and the evidence."""

    features: dict[str, float]
    quality: QualityReport
    roi: RoiResult
    preprocessed: Preprocessed
    normalised_roi: np.ndarray | None
    oversampling: float

    @property
    def usable(self) -> bool:
        return self.quality.usable and self.normalised_roi is not None


def extract_features(
    bgr: np.ndarray,
    *,
    config: Config | None = None,
    thresholds: QualityThresholds | None = None,
    include_legacy_tsci: bool = False,
) -> Extraction:
    """Run the full analysis pipeline on one already-decoded image.

    The quality gate runs *before* feature extraction, and when it refuses, no
    features are computed at all. That ordering is deliberate: it makes it
    impossible for a caller to accidentally obtain a prediction for an image the
    gate rejected.

    Parameters
    ----------
    include_legacy_tsci
        Adds the original TSCI value. Used by experiments that compare against the
        published pipeline; never enabled in production.
    """
    config = config or CONFIG

    pre = preprocess(bgr, config.preprocess)
    roi = extract_roi(pre.smoothed, config.roi)
    over = oversampling_factor(roi.roi.shape, config.scale)

    quality = assess_quality(
        pre.enhanced,
        raw=pre.gray,
        subject_coverage=measure_subject_coverage(pre.smoothed),
        oversampling=over,
        roi_method=roi.method,
        roi_coverage=roi.coverage,
        thresholds=thresholds,
    )

    if not quality.usable:
        return Extraction(
            features={}, quality=quality, roi=roi, preprocessed=pre,
            normalised_roi=None, oversampling=over,
        )

    normalised = normalise_scale(roi.roi, config.scale)

    features: dict[str, float] = {}
    features.update(spectral_features(normalised, config.spectral).as_dict())
    features.update(texture_features(normalised, config.texture).as_dict())
    if include_legacy_tsci:
        features["legacy_tsci"] = legacy_tsci(roi.roi, config.spectral)

    return Extraction(
        features=features, quality=quality, roi=roi, preprocessed=pre,
        normalised_roi=normalised, oversampling=over,
    )


def feature_names(*, include_legacy_tsci: bool = False) -> list[str]:
    """Canonical feature order.

    Persisted alongside a model artifact, because a scaler and a coefficient vector
    are meaningless without knowing which column is which.
    """
    probe = np.random.default_rng(0).integers(0, 255, (256, 256), dtype=np.uint8)
    names = list(spectral_features(probe).as_dict())
    names += list(texture_features(probe).as_dict())
    if include_legacy_tsci:
        names.append("legacy_tsci")
    return names
