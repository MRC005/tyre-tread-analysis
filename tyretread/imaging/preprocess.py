"""Greyscale conversion, contrast equalisation and scale normalisation.

The scale-normalisation step is the correction at the heart of this revision. See
``normalise_scale`` and docs/AUDIT.md 3.2 for why the original fixed-size resize
was invalid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CONFIG, PreprocessConfig, ScaleConfig

__all__ = ["Preprocessed", "to_luma", "enhance", "preprocess", "normalise_scale",
           "oversampling_factor"]


@dataclass(frozen=True)
class Preprocessed:
    gray: np.ndarray
    enhanced: np.ndarray
    smoothed: np.ndarray


def to_luma(bgr: np.ndarray) -> np.ndarray:
    """BGR to a single luma channel."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def enhance(gray: np.ndarray, config: PreprocessConfig | None = None) -> np.ndarray:
    """Local contrast equalisation.

    CLAHE rather than global histogram equalisation because tread photographs
    routinely contain a bright sunlit region and a deep shadow inside the grooves;
    a global transform sacrifices one to the other.
    """
    config = config or CONFIG.preprocess
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_tile_grid
    )
    return clahe.apply(gray)


def preprocess(bgr: np.ndarray, config: PreprocessConfig | None = None) -> Preprocessed:
    config = config or CONFIG.preprocess
    gray = to_luma(bgr)
    enhanced = enhance(gray, config)
    smoothed = cv2.GaussianBlur(enhanced, config.gaussian_ksize, 0)
    return Preprocessed(gray=gray, enhanced=enhanced, smoothed=smoothed)


def oversampling_factor(
    roi_shape: tuple[int, int], config: ScaleConfig | None = None
) -> float:
    """How many native ROI pixels are available per analysis pixel, per axis.

    A factor below 1 means the analysis window cannot be filled without inventing
    detail. Measurements show the frequency-domain descriptors only stabilise from
    roughly 3x upwards (experiments/exp003_oversampling).
    """
    config = config or CONFIG.scale
    h, w = roi_shape[:2]
    return min(w / config.analysis_width, h / config.analysis_height)


def normalise_scale(
    roi: np.ndarray, config: ScaleConfig | None = None
) -> np.ndarray:
    """Resample an ROI onto the common analysis grid, downsampling only.

    Why this is not simply ``cv2.resize`` to a fixed size
    ----------------------------------------------------
    Spatial frequency in a sampled image is measured in cycles per pixel, so a
    frequency-domain feature is only comparable between two images if both were
    sampled at the same pixel scale *and* both actually carry real detail up to
    the analysis grid's Nyquist limit.

    The original pipeline resized every ROI to 256x128 whether that meant
    discarding detail from a 5000 px photograph or stretching a 148 px thumbnail.
    Upsampling cannot create detail, so a thumbnail arrived at the FFT with its
    upper octaves empty while a large photograph arrived with them full. Every
    frequency feature therefore encoded the image's resolution and compression
    history. Measured: TSCI moved +0.306 across a resolution sweep of unchanged
    tyres, against a 0.074 good-versus-worn difference.

    The remedy here is to refuse to upsample. ``INTER_AREA`` applies a box filter
    before decimating, so downsampling is properly band-limited and the analysis
    grid is filled with genuine detail. An ROI too small to fill the grid is not
    stretched; it is rejected upstream by the quality gate, which is the honest
    response to an image that does not contain the required information.

    Raises
    ------
    ValueError
        If the ROI would need upsampling and ``allow_upsampling`` is False.
    """
    config = config or CONFIG.scale
    target = (config.analysis_width, config.analysis_height)
    h, w = roi.shape[:2]

    if (w < config.analysis_width or h < config.analysis_height) and not config.allow_upsampling:
        raise ValueError(
            f"ROI {w}x{h} is smaller than the {target[0]}x{target[1]} analysis grid; "
            "upsampling would fabricate detail. Reject the image instead."
        )

    interpolation = cv2.INTER_AREA if (w >= target[0] and h >= target[1]) else cv2.INTER_LINEAR
    return cv2.resize(roi, target, interpolation=interpolation)
