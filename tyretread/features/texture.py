"""Texture descriptors of a tread ROI.

Changes from the original implementation, each with a reason:

* **Grey levels reduced from 256 to 32.** A 256x256 co-occurrence matrix built
  from a 256x128 ROI has at most 32768 counts spread over 65536 cells, so it is
  mostly zeros and the resulting statistics are dominated by sampling noise.
  Quantisation is standard practice in the texture literature and is also much
  faster.
* **Four angles instead of one.** The original used 0 degrees only, which makes
  every descriptor depend on how the tread happened to be rotated in frame. A
  phone is not held at a fixed angle, so the descriptors are averaged over
  0/45/90/135 degrees to become rotation-insensitive.
* **Three distances instead of one.** Groove pitch is a real, physically
  meaningful scale, and a single one-pixel offset only probes the finest one.
* **Rotation-invariant uniform LBP is now actually used.** The original computed
  an LBP histogram on every image and then discarded it before training
  (docs/AUDIT.md 3.10). Either use it or do not pay for it.

Edge density is retained but, unlike the above, it is inherently scale-dependent:
edges per pixel changes when an image is resampled. It is only comparable once
scale normalisation has been applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from ..config import CONFIG, TextureConfig

__all__ = ["TextureFeatures", "glcm_features", "lbp_histogram", "edge_density",
           "texture_features"]

_GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")


def _quantise(gray: np.ndarray, levels: int) -> np.ndarray:
    return (gray.astype(np.uint16) * levels // 256).clip(0, levels - 1).astype(np.uint8)


def glcm_features(
    gray: np.ndarray, config: TextureConfig | None = None
) -> dict[str, float]:
    """Grey-level co-occurrence statistics, averaged over angles per distance.

    Averaging over angles (rather than keeping each angle separately) is what buys
    rotation insensitivity. Distances are kept separate because they carry
    genuinely different information about groove scale.
    """
    config = config or CONFIG.texture
    quantised = _quantise(gray, config.levels)
    angles = [np.deg2rad(a) for a in config.angles_deg]

    glcm = graycomatrix(
        quantised,
        distances=list(config.distances),
        angles=angles,
        levels=config.levels,
        symmetric=True,
        normed=True,
    )

    out: dict[str, float] = {}
    for prop in _GLCM_PROPS:
        # shape (n_distances, n_angles)
        values = graycoprops(glcm, prop)
        for di, distance in enumerate(config.distances):
            out[f"glcm_{prop}_d{distance}"] = float(np.nanmean(values[di, :]))
        # Spread across angles is itself a directionality cue: a strongly grooved
        # tread is anisotropic, a worn one much less so.
        out[f"glcm_{prop}_anisotropy"] = float(
            np.nanmean(np.nanstd(values, axis=1) / (np.abs(np.nanmean(values, axis=1)) + 1e-9))
        )
    return out


def lbp_histogram(
    gray: np.ndarray, config: TextureConfig | None = None
) -> dict[str, float]:
    """Rotation-invariant uniform LBP histogram.

    ``uniform`` LBP collapses the 2**P patterns into P+2 rotation-invariant bins,
    which is what makes it usable on images whose orientation is uncontrolled.
    """
    config = config or CONFIG.texture
    lbp = local_binary_pattern(gray, config.lbp_points, config.lbp_radius, method="uniform")
    n_bins = config.lbp_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(n_bins + 1), range=(0, n_bins))
    hist = hist.astype(np.float64)
    hist /= hist.sum() + 1e-12
    return {f"lbp_{i}": float(v) for i, v in enumerate(hist)}


def edge_density(gray: np.ndarray) -> dict[str, float]:
    """Canny edge density with a median-adaptive threshold.

    The adaptive threshold is the original author's choice and a good one: a fixed
    threshold makes edge density a brightness measurement rather than a structure
    measurement.
    """
    median = float(np.median(gray))
    lower = int(max(0, 0.67 * median))
    upper = int(min(255, 1.33 * median))
    edges = cv2.Canny(gray, lower, upper)
    density = float(np.count_nonzero(edges) / edges.size)

    # Gradient statistics describe edge strength, which edge density alone cannot:
    # a faint but pervasive texture and a few strong grooves can share a density.
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    return {
        "edge_density": density,
        "gradient_mean": float(magnitude.mean()),
        "gradient_p95": float(np.percentile(magnitude, 95)),
    }


@dataclass(frozen=True)
class TextureFeatures:
    values: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)


def texture_features(
    gray: np.ndarray, config: TextureConfig | None = None
) -> TextureFeatures:
    config = config or CONFIG.texture
    values: dict[str, float] = {}
    values.update(glcm_features(gray, config))
    values.update(lbp_histogram(gray, config))
    values.update(edge_density(gray))
    return TextureFeatures(values=values)
