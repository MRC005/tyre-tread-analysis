"""Tread region-of-interest extraction.

The original implementation is preserved as the primary strategy because it is
well reasoned: a centre band avoids the wheel rim and the background, and contour
refinement is only accepted when it finds a plausibly tread-shaped band. What is
added here is that the ROI reports *how* it was found and how confident that was,
because the quality gate and the user-facing explanation both need to know whether
the tread was actually located or merely assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CONFIG, RoiConfig

__all__ = ["RoiResult", "extract_roi"]


@dataclass(frozen=True)
class RoiResult:
    roi: np.ndarray
    #: Bounding box in the source image, as (x, y, w, h).
    box: tuple[int, int, int, int]
    #: "contour" when a tread-shaped band was located, "centre_band" when the
    #: fallback crop was used. The fallback is a guess, and the report says so.
    method: str
    edges: np.ndarray
    closed: np.ndarray
    #: Fraction of the centre band occupied by the accepted contour. Low values
    #: mean the tread could not be confidently separated from its surroundings.
    coverage: float


def extract_roi(smoothed: np.ndarray, config: RoiConfig | None = None) -> RoiResult:
    """Locate the tread band within a preprocessed greyscale image."""
    config = config or CONFIG.roi
    h, w = smoothed.shape[:2]

    top = int(h * config.centre_band_top)
    bottom = int(h * config.centre_band_bottom)
    band = smoothed[top:bottom, :]

    edges = cv2.Canny(cv2.GaussianBlur(band, (5, 5), 0), config.canny_low, config.canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, config.close_kernel)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    roi = band
    method = "centre_band"
    box = (0, top, w, bottom - top)
    coverage = 0.0

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        aspect = cw / ch if ch > 0 else 0.0
        band_height = band.shape[0]
        band_area = band_height * band.shape[1]
        candidate_coverage = (cw * ch) / band_area if band_area else 0.0

        # All four conditions must hold. Aspect and width alone admitted thin
        # horizontal slivers, which made the ROI unstable under trivial re-encoding -
        # see RoiConfig.min_height_fraction. The centre band is a safe fallback, so
        # the bar for preferring a contour over it should be high.
        wide_enough = cw > w * config.min_width_fraction
        tall_enough = ch > band_height * config.min_height_fraction
        covers_enough = candidate_coverage >= config.min_band_coverage

        if aspect > config.min_aspect_ratio and wide_enough and tall_enough and covers_enough:
            roi = band[y : y + ch, x : x + cw]
            method = "contour"
            box = (x, top + y, cw, ch)
            coverage = candidate_coverage

    return RoiResult(
        roi=roi, box=box, method=method, edges=edges, closed=closed, coverage=coverage
    )
