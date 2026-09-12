"""Visual evidence for an inspection.

The point is not decoration. Each panel answers a question a sceptical user or
interviewer would reasonably ask: did it actually find the tread, what did it look at,
and what structure did it measure? A verdict accompanied by the ROI it was computed
from is checkable; a bare percentage is not.

Images are returned as base64 data URIs. The service keeps no uploads, so there would
be nothing for a second request to fetch, and inlining avoids inventing a storage
layer purely to serve thumbnails.

Photographic panels are encoded as JPEG and binary panels (the edge map) as PNG.
Encoding everything as PNG produced a 571 KB response for five panels, which is a
poor thing to send to a phone on mobile data; the mixed encoding brings that down by
roughly 80% with no loss that matters, since the panels are for looking at and the
measurements were taken before any of this.
"""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from PIL import Image

from .features.extract import Extraction
from .features.spectral import power_spectrum

__all__ = ["render_evidence", "to_data_uri"]

#: Long edge of every returned panel. Large enough to see groove structure on a
#: phone screen, small enough that five panels do not dominate the response.
PANEL_LONG_EDGE = 480


def to_data_uri(
    image: np.ndarray,
    *,
    long_edge: int = PANEL_LONG_EDGE,
    lossless: bool = False,
    quality: int = 78,
) -> str:
    """Encode a greyscale or BGR array as a base64 data URI.

    Parameters
    ----------
    lossless
        Use PNG. Appropriate for binary masks such as the edge map, where JPEG
        ringing would add structure that is not in the data. Photographic panels use
        JPEG, which is far smaller for the same perceived quality.
    """
    if image.ndim == 2:
        pil = Image.fromarray(image.astype(np.uint8), mode="L")
    else:
        pil = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

    scale = long_edge / max(pil.size)
    if scale < 1.0:
        pil = pil.resize(
            (max(1, round(pil.size[0] * scale)), max(1, round(pil.size[1] * scale))),
            Image.LANCZOS,
        )

    buffer = io.BytesIO()
    if lossless:
        pil.save(buffer, format="PNG", optimize=True)
        media_type = "image/png"
    else:
        pil.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
        media_type = "image/jpeg"
    return f"data:{media_type};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _spectrum_panel(roi: np.ndarray) -> np.ndarray:
    """Log power spectrum, contrast-stretched for display only.

    Purely presentational: the displayed image is not what the descriptors are
    computed from, and the stretch is applied after all measurement.
    """
    log_power = np.log1p(power_spectrum(roi))
    span = log_power.max() - log_power.min()
    if span <= 0:
        return np.zeros_like(log_power, dtype=np.uint8)
    normalised = (log_power - log_power.min()) / span
    return (normalised * 255).astype(np.uint8)


def _annotate_roi(bgr: np.ndarray, box: tuple[int, int, int, int], located: bool) -> np.ndarray:
    """Draw the detected tread band on a copy of the original photograph."""
    annotated = bgr.copy()
    x, y, w, h = box
    # Green when the tread was positively located, amber when the centre-band
    # fallback was used - the difference is a real caveat, so it is shown.
    colour = (76, 175, 80) if located else (0, 170, 255)
    thickness = max(2, round(min(annotated.shape[:2]) / 200))
    cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, thickness)
    return annotated


def render_evidence(
    bgr: np.ndarray, extraction: Extraction, *, include_spectrum: bool = True
) -> dict[str, str]:
    """Build the evidence panels available for this extraction.

    A refused image still yields the panels that were computed before the refusal,
    because seeing the failed ROI detection is exactly what tells a user why their
    photograph was rejected.
    """
    panels: dict[str, str] = {
        "original": to_data_uri(
            _annotate_roi(bgr, extraction.roi.box, extraction.roi.method == "contour")
        ),
        "enhanced": to_data_uri(extraction.preprocessed.enhanced),
        # Lossless, because the edge map is binary and JPEG would invent grey
        # structure in it - but smaller, because a dense binary map is the worst case
        # for PNG and this panel is schematic rather than detail-critical.
        "edges": to_data_uri(extraction.roi.edges, lossless=True, long_edge=360),
    }

    roi = extraction.normalised_roi
    if roi is None:
        # The gate refused before normalisation; show the raw crop instead so the
        # user can still see what the system was looking at.
        panels["roi"] = to_data_uri(extraction.roi.roi)
        return panels

    panels["roi"] = to_data_uri(roi)
    if include_spectrum:
        panels["spectrum"] = to_data_uri(_spectrum_panel(roi))
    return panels
