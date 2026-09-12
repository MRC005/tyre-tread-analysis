"""Candidate tread-region localisers.

exp013 established that when a user photographs a whole wheel, the dominant failure is
localisation: an oracle crop back to the tyre recovers about 80% of the damage. These
are the simplest defensible approaches, implemented so they can be measured against one
another rather than chosen by intuition.

Nothing here is wired into production. exp014 ranks them; adoption is a separate
decision, and "none of these is good enough, so abstain instead" is an acceptable
outcome.

The shared idea is that tread has a texture signature the surroundings do not:
strong, densely packed, *directionally coherent* edges. Ground, bodywork and sky are
either smooth or randomly textured. Each candidate below turns that observation into a
region proposal differently.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

__all__ = ["Candidate", "centre_band", "contour_band", "texture_energy",
           "dark_texture", "multi_scale_score", "LOCALISERS"]


@dataclass(frozen=True)
class Candidate:
    """A proposed region, with a self-reported confidence in [0, 1]."""

    box: tuple[int, int, int, int]
    confidence: float
    method: str


def _band_of(gray: np.ndarray, top: float = 0.20, bottom: float = 0.80) -> tuple[int, int]:
    h = gray.shape[0]
    return int(h * top), int(h * bottom)


def centre_band(gray: np.ndarray) -> Candidate:
    """The current production fallback: assume the tread is in the middle band.

    Baseline. It is not a localiser at all - it is a guess that happens to be right when
    the user fills the frame, which is exactly the case where localisation is easy.
    """
    h, w = gray.shape[:2]
    top, bottom = _band_of(gray)
    return Candidate((0, top, w, bottom - top), 0.0, "centre_band")


def contour_band(gray: np.ndarray) -> Candidate:
    """The current contour refinement: largest closed edge blob, if band-shaped."""
    h, w = gray.shape[:2]
    top, bottom = _band_of(gray)
    band = gray[top:bottom, :]

    edges = cv2.Canny(cv2.GaussianBlur(band, (5, 5), 0), 50, 150)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return centre_band(gray)

    x, y, cw, ch = cv2.boundingRect(max(contours, key=cv2.contourArea))
    band_h = bottom - top
    aspect = cw / ch if ch else 0.0
    coverage = (cw * ch) / max(1, band_h * w)
    if aspect > 2.0 and cw > w * 0.4 and ch > band_h * 0.35 and coverage >= 0.25:
        return Candidate((x, top + y, cw, ch), min(1.0, coverage), "contour_band")
    return centre_band(gray)


def _texture_map(gray: np.ndarray, cell: int) -> np.ndarray:
    """Per-cell edge energy: how much structured detail each block contains."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    h, w = gray.shape[:2]
    rows, cols = max(1, h // cell), max(1, w // cell)
    return cv2.resize(magnitude, (cols, rows), interpolation=cv2.INTER_AREA)


#: The score map is resampled to this many cells on its longest side before the window
#: search. The search is O(rows^2 * cols^2), so on a 4000 px photograph an unbounded
#: grid ran for tens of seconds per image - unusable in a request and slow enough to
#: obstruct its own evaluation. Normalising the grid makes the cost independent of
#: input size, which is what a production component needs.
SEARCH_GRID = 20
#: Step between candidate window sizes. Halves the search again at negligible cost to
#: the located box, which is then mapped back to full resolution anyway.
SIZE_STEP = 2


def _best_window(score: np.ndarray, cell: int, shape: tuple[int, int],
                 min_frac: float = 0.25) -> tuple[tuple[int, int, int, int], float]:
    """Highest-scoring axis-aligned window over a coarse score map.

    Uses an integral image so every candidate position is evaluated exactly rather than
    by a greedy search. The map is first normalised to a fixed grid so the cost does not
    depend on the input resolution.
    """
    src_rows, src_cols = score.shape
    scale = SEARCH_GRID / max(src_rows, src_cols)
    if scale < 1.0:
        score = cv2.resize(
            score, (max(2, int(src_cols * scale)), max(2, int(src_rows * scale))),
            interpolation=cv2.INTER_AREA,
        )
    rows, cols = score.shape
    # Cells in the resampled grid map back to this many source pixels.
    cell_y = cell * src_rows / rows
    cell_x = cell * src_cols / cols

    integral = cv2.integral(score.astype(np.float64))

    best = None
    best_mean = -1.0
    for wh in range(max(1, int(rows * min_frac)), rows + 1, SIZE_STEP):
        for ww in range(max(1, int(cols * min_frac)), cols + 1, SIZE_STEP):
            for y in range(0, rows - wh + 1):
                for x in range(0, cols - ww + 1):
                    total = (integral[y + wh, x + ww] - integral[y, x + ww]
                             - integral[y + wh, x] + integral[y, x])
                    mean = total / (wh * ww)
                    # Prefer dense texture, but do not let a single hot cell win: the
                    # area term keeps the window from collapsing onto one groove.
                    value = mean * (1.0 + 0.15 * np.log((wh * ww) / (rows * cols) + 1e-9))
                    if value > best_mean:
                        best_mean, best = value, (x, y, ww, wh)

    x, y, ww, wh = best  # type: ignore[misc]
    h, w = shape
    px, py = int(x * cell_x), int(y * cell_y)
    box = (
        px, py,
        int(min(ww * cell_x, w - px)), int(min(wh * cell_y, h - py)),
    )
    # Confidence: how much denser the chosen window is than the frame as a whole.
    overall = float(score.mean()) or 1e-6
    inside = float(score[y : y + wh, x : x + ww].mean())
    confidence = float(np.clip((inside / overall - 1.0), 0.0, 1.0))
    return box, confidence


def texture_energy(gray: np.ndarray, cell: int = 32) -> Candidate:
    """Find the densest region of edge energy.

    Tread is the most structurally detailed thing in a tyre photograph; ground and
    bodywork are comparatively smooth.
    """
    score = _texture_map(gray, cell)
    box, confidence = _best_window(score, cell, gray.shape[:2])
    return Candidate(box, confidence, "texture_energy")


def dark_texture(gray: np.ndarray, cell: int = 32) -> Candidate:
    """Edge energy weighted towards dark regions.

    Rubber is dark. Weighting by darkness suppresses bright ground and sky, which can
    carry surprising amounts of edge energy from gravel or paving joints.
    """
    score = _texture_map(gray, cell)
    h, w = gray.shape[:2]
    rows, cols = score.shape
    coarse = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Peaks for dark pixels, falls away smoothly towards white.
    darkness = np.clip(1.0 - coarse / 200.0, 0.0, 1.0)
    box, confidence = _best_window(score * darkness, cell, (h, w))
    return Candidate(box, confidence, "dark_texture")


def multi_scale_score(gray: np.ndarray, cell: int = 32) -> Candidate:
    """Edge energy weighted by darkness *and* directional coherence.

    Tread grooves run in a consistent direction; gravel and foliage do not. Coherence
    is the magnitude of the doubled-angle gradient mean, which is high for locally
    parallel structure and near zero for random texture.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    # Structure tensor components, smoothed over a neighbourhood.
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 3)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 3)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 3)
    trace = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (trace + 1e-6)

    h, w = gray.shape[:2]
    rows, cols = max(1, h // cell), max(1, w // cell)
    energy = cv2.resize(magnitude, (cols, rows), interpolation=cv2.INTER_AREA)
    coh = cv2.resize(coherence, (cols, rows), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
    darkness = np.clip(1.0 - coarse / 200.0, 0.0, 1.0)

    box, confidence = _best_window(energy * coh * darkness, cell, (h, w))
    return Candidate(box, confidence, "multi_scale_score")


LOCALISERS = {
    "centre_band": centre_band,
    "contour_band": contour_band,
    "texture_energy": texture_energy,
    "dark_texture": dark_texture,
    "multi_scale_score": multi_scale_score,
}


def subject_coverage(gray: np.ndarray, cell: int = 32) -> float:
    """Fraction of the frame carrying tyre-like texture.

    Not a localiser. exp015 found that texture search is a poor localiser to adopt -
    it helps wide shots and hurts well-framed ones, with no threshold separating the
    two - but that the same signal detects *how much of the frame is tyre* very
    reliably. Detecting "the subject is small in this photograph" is a much easier
    problem than finding exactly where it is, and it is the one the product actually
    needs: a wide shot should be refused with "move closer", not silently cropped.

    Returns roughly 1.0 when structured, dark, tyre-like texture fills the frame and
    falls towards 0 as the tyre shrinks within it. Deliberately unbounded-free of the
    saturation that made the localiser's own confidence unusable as a threshold: this
    is a proportion, so it uses the whole range.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)

    h, w = gray.shape[:2]
    rows, cols = max(2, h // cell), max(2, w // cell)
    energy = cv2.resize(magnitude, (cols, rows), interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
    darkness = np.clip(1.0 - coarse / 200.0, 0.0, 1.0)
    score = energy * darkness

    # Compare against a high percentile rather than the max, so one specular highlight
    # or a single bright groove edge cannot set the scale for the whole frame.
    peak = float(np.percentile(score, 95))
    if peak <= 1e-6:
        return 0.0
    return float(np.mean(score >= 0.40 * peak))
