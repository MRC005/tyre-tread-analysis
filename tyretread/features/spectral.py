"""Frequency-domain descriptors of a tread ROI.

The original project's headline feature, TSCI = E_high / E_total, was shown to be
dominated by an artefact rather than by tread condition: sweeping only the source
resolution of an unchanged tyre photograph moved TSCI by +0.266, while the whole
good-versus-worn difference in the dataset was 0.074 (exp001; docs/AUDIT.md 3.1-3.2).

The cause is dimensional. Spatial frequency in a digital image is measured in
cycles per pixel, and an absolute radius threshold in that space only means the
same thing across two images if both were sampled at the same pixel scale. No
photograph of a tyre carries a calibration reference, so millimetres per pixel is
unknown and absolute frequency features are not comparable between images.

This module therefore separates two kinds of descriptor:

``legacy_tsci``
    Kept verbatim. exp004 revised the initial judgement on it: the ratio is not
    intrinsically invalid, but it has an unstated precondition. Its sensitivity to
    resolution falls from 2.5x the good-versus-worn signal to 0.6x once oversampling
    exceeds about 3x. It is therefore a candidate feature *behind the quality gate's
    oversampling floor*, never on a raw image, and its published interpretation
    (decreasing with wear) remains contradicted by the data.

Scale-invariant descriptors
    Quantities that are unchanged by isotropic rescaling of the image, so they
    remain comparable when the unknown pixel scale differs:

    * ``spectral_slope`` - the exponent a in P(f) ~ f**-a. Rescaling maps
      f -> f/s, which shifts the log-log spectrum horizontally without changing
      its gradient, so a is scale-invariant by construction. It measures how
      texture energy is distributed across scales: a flatter spectrum (small a)
      means detail at many scales, a steeper one means a smooth surface whose
      energy sits at coarse scales.
    * ``orientation_*`` - the angular distribution of spectral energy.
      Isotropic rescaling does not rotate anything, so every angular statistic is
      scale-invariant. This is physically the right thing to ask of a tyre: an
      unworn tread has deep, strongly oriented grooves, so its spectrum is
      anisotropic; a worn tread flattens towards an isotropic, unstructured
      surface.

Each descriptor's resolution-stability is measured, not assumed - see
``experiments/exp002_scale_invariance``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CONFIG, SpectralConfig

__all__ = [
    "SpectralFeatures",
    "legacy_tsci",
    "power_spectrum",
    "radial_profile",
    "spectral_slope",
    "orientation_statistics",
    "spectral_features",
]


def power_spectrum(roi: np.ndarray) -> np.ndarray:
    """Centred power spectrum of ``roi`` with the DC term removed.

    A Hann window is applied first. Without it the implicit discontinuity between
    opposite image edges injects a cross-shaped spectral artefact that contaminates
    every orientation statistic computed downstream.
    """
    img = roi.astype(np.float64)
    img = img - img.mean()

    wy = np.hanning(img.shape[0])
    wx = np.hanning(img.shape[1])
    img = img * np.outer(wy, wx)

    spectrum = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(spectrum) ** 2
    power[power.shape[0] // 2, power.shape[1] // 2] = 0.0
    return power


def _radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """Radial distance from the spectrum centre, normalised so 1.0 is Nyquist.

    Normalising by the half-extent of each axis independently means the grid is
    expressed in cycles-per-sample rather than cycles-per-image, which is what
    makes the resulting statistics independent of the analysis window's size.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    yy = (np.arange(h) - cy) / max(cy, 1)
    xx = (np.arange(w) - cx) / max(cx, 1)
    return np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)


def radial_profile(power: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Mean power in ``n_bins`` annuli, against normalised frequency."""
    radius = _radius_grid(power.shape)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    idx = np.clip(np.digitize(radius.ravel(), edges) - 1, 0, n_bins - 1)
    total = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    count = np.bincount(idx, minlength=n_bins)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return centres, mean


def spectral_slope(
    power: np.ndarray, config: SpectralConfig | None = None
) -> tuple[float, float]:
    """Fit ``log P = c - a log f`` and return ``(a, r_squared)``.

    ``a`` is the scale-invariant descriptor; ``r_squared`` reports how well a
    single power law actually describes this image, which is useful evidence in
    its own right - a tread with one dominant groove pitch is poorly described by
    a power law, and that is informative rather than a defect.
    """
    config = config or CONFIG.spectral
    lo, hi = config.slope_band

    centres, mean = radial_profile(power)
    band = (centres >= lo) & (centres <= hi) & (mean > 0)
    if band.sum() < 4:
        return float("nan"), float("nan")

    x = np.log(centres[band])
    y = np.log(mean[band])
    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(-slope), float(r2)


def orientation_statistics(
    power: np.ndarray, config: SpectralConfig | None = None
) -> dict[str, float]:
    """Angular energy distribution of the spectrum.

    Only the mid-frequency band is used. Very low frequencies describe overall
    illumination gradients rather than tread structure, and the band just below
    Nyquist is shaped by whatever anti-alias filtering the image has already been
    through.

    Returns
    -------
    orientation_anisotropy
        Peak angular energy divided by the mean, minus one. Zero for a perfectly
        isotropic surface and growing with directional structure.
    orientation_coherence
        ``1 - circular_variance`` of the doubled angles. Orientation is defined
        modulo 180 degrees, so the angles are doubled before the circular
        statistic is taken. 0 is unstructured, 1 is a single sharp direction.
    orientation_dominant_deg
        The dominant orientation in degrees, in [0, 180).
    orientation_entropy
        Shannon entropy of the angular distribution, normalised to [0, 1]. High
        means energy spread evenly over directions.
    """
    config = config or CONFIG.spectral
    lo, hi = config.slope_band
    n_bins = config.orientation_bins

    h, w = power.shape
    cy, cx = h // 2, w // 2
    yy = (np.arange(h) - cy) / max(cy, 1)
    xx = (np.arange(w) - cx) / max(cx, 1)
    radius = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    band = (radius >= lo) & (radius <= hi)

    if not band.any():
        return {
            "orientation_anisotropy": float("nan"),
            "orientation_coherence": float("nan"),
            "orientation_dominant_deg": float("nan"),
            "orientation_entropy": float("nan"),
        }

    theta = np.mod(np.arctan2(yy[:, None] + 0.0 * xx[None, :], xx[None, :] + 0.0 * yy[:, None]), np.pi)
    idx = np.clip((theta[band] / np.pi * n_bins).astype(int), 0, n_bins - 1)
    energy = np.bincount(idx, weights=power[band], minlength=n_bins)

    total = energy.sum()
    if total <= 0:
        return {
            "orientation_anisotropy": float("nan"),
            "orientation_coherence": float("nan"),
            "orientation_dominant_deg": float("nan"),
            "orientation_entropy": float("nan"),
        }

    p = energy / total
    anisotropy = float(energy.max() / energy.mean() - 1.0)

    # Orientation is modulo pi, so double the angle for the circular statistic.
    bin_centres = (np.arange(n_bins) + 0.5) / n_bins * np.pi
    vec = np.sum(p * np.exp(2j * bin_centres))
    coherence = float(np.abs(vec))
    dominant = float(np.mod(np.angle(vec) / 2.0, np.pi) * 180.0 / np.pi)

    nz = p[p > 0]
    entropy = float(-np.sum(nz * np.log(nz)) / np.log(n_bins))

    return {
        "orientation_anisotropy": anisotropy,
        "orientation_coherence": coherence,
        "orientation_dominant_deg": dominant,
        "orientation_entropy": entropy,
    }


def legacy_tsci(roi: np.ndarray, config: SpectralConfig | None = None) -> float:
    """The original TSCI, reproduced exactly.

    Retained so the published 74.8% result stays reproducible and so regression
    tests can detect drift. Its documented physical interpretation does not hold
    on real data - see the module docstring. Not used in production inference.
    """
    import cv2

    config = config or CONFIG.spectral
    resized = cv2.resize(roi, (256, 128), interpolation=cv2.INTER_AREA)

    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(resized.astype(np.float64))))
    h, w = magnitude.shape
    magnitude[h // 2, w // 2] = 0.0

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mask = dist > config.legacy_hf_radius_fraction * min(h, w)

    total = float(magnitude.sum())
    return float(magnitude[mask].sum() / total) if total > 0 else 0.0


@dataclass(frozen=True)
class SpectralFeatures:
    spectral_slope: float
    spectral_slope_r2: float
    orientation_anisotropy: float
    orientation_coherence: float
    orientation_dominant_deg: float
    orientation_entropy: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def spectral_features(
    roi: np.ndarray, config: SpectralConfig | None = None
) -> SpectralFeatures:
    """All scale-invariant spectral descriptors for one ROI."""
    power = power_spectrum(roi)
    slope, r2 = spectral_slope(power, config)
    orient = orientation_statistics(power, config)
    return SpectralFeatures(
        spectral_slope=slope,
        spectral_slope_r2=r2,
        **orient,
    )
