import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def _best_projection(clean):
    """
    Try both row-wise and column-wise projections.
    Pick the one with the larger peak-to-valley ratio —
    that axis is aligned with the groove direction.
    Returns: (projection_normalized, axis_label)
    """
    def evaluate(proj):
        s = gaussian_filter1d(proj.astype(float), sigma=8)
        if np.max(s) == 0:
            return s, 0.0
        s = s / np.max(s)
        return s, float(np.max(s) - np.min(s))

    row_proj = np.mean(clean, axis=1)
    col_proj = np.mean(clean, axis=0)

    row_s, row_score = evaluate(row_proj)
    col_s, col_score = evaluate(col_proj)

    if row_score >= col_score:
        return row_s, "row"
    else:
        return col_s, "col"


def compute_tpdi(roi):
    # -----------------------------------------------
    # Step 1: Otsu threshold (global)
    # Better than adaptive for this image — the grooves
    # are large dark regions, not fine local texture.
    # Adaptive was picking up surface texture noise.
    # -----------------------------------------------
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # -----------------------------------------------
    # Step 2: Aggressive morphological cleaning
    # Large opening (15x15) kills all small noise blobs
    # Large closing (25x25) fills groove band regions fully
    # Square kernels work for any groove orientation
    # -----------------------------------------------
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,  5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))

    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel_open)
    clean = cv2.morphologyEx(clean,  cv2.MORPH_CLOSE, kernel_close)

    # -----------------------------------------------
    # Step 3: Auto-detect groove direction
    # Heavier smoothing (sigma=8) to suppress noise peaks
    # -----------------------------------------------
    projection, axis_used = _best_projection(clean)

    # -----------------------------------------------
    # Step 4: Peak Detection
    # Higher threshold (0.6) — only clear groove peaks
    # Larger min_distance (25) — real grooves are wide
    # -----------------------------------------------
    peaks        = []
    min_distance = 25
    last_peak    = -min_distance

    for i in range(3, len(projection) - 3):
        if (
            projection[i] > 0.6 and
            projection[i] > projection[i - 1] and
            projection[i] > projection[i + 1] and
            projection[i] > projection[i - 2] and
            projection[i] > projection[i + 2] and
            projection[i] > projection[i - 3] and
            projection[i] > projection[i + 3] and
            (i - last_peak) >= min_distance
        ):
            peaks.append(i)
            last_peak = i

    # -----------------------------------------------
    # Step 5: Groove Spacing via DFT
    # Dominant frequency of the cleaned projection signal
    # -----------------------------------------------
    if axis_used == "col":
        raw_signal = np.mean(clean, axis=0).astype(float)
    else:
        raw_signal = np.mean(clean, axis=1).astype(float)

    signal  = gaussian_filter1d(raw_signal, sigma=5)
    fft_mag = np.abs(np.fft.rfft(signal))
    fft_mag[0] = 0

    cutoff = int(len(fft_mag) * 0.85)
    fft_mag[cutoff:] = 0

    dominant_idx = np.argmax(fft_mag[1:]) + 1
    spacing = len(signal) / dominant_idx if dominant_idx > 0 else max(roi.shape) / 4

    # Sanity clamp: spacing must be between 5% and 50% of image dimension
    dim = len(signal)
    spacing = np.clip(spacing, dim * 0.05, dim * 0.50)

    # -----------------------------------------------
    # Step 6: Shadow Width
    # Count DARK pixels (groove = dark = 0 in clean)
    # -----------------------------------------------
    if axis_used == "row":
        shadow_widths = [np.sum(clean[r, :] == 0) for r in range(clean.shape[0])]
    else:
        shadow_widths = [np.sum(clean[:, c] == 0) for c in range(clean.shape[1])]

    avg_shadow_width = float(np.mean(shadow_widths))

    # -----------------------------------------------
    # Step 7: TPDI = avg_shadow_width / groove_spacing
    # Both in pixels — ratio is dimensionless & scale-invariant
    # -----------------------------------------------
    tpdi       = avg_shadow_width / spacing
    band_count = len(peaks)

    return tpdi, binary, clean, projection, peaks, band_count, spacing, avg_shadow_width, axis_used


def classify_tpdi(tpdi):
    """
    Thresholds empirically determined from dataset.
    Always say this in viva — never present as fixed truth.
    Adjust after full dataset validation.
    """
    if tpdi > 0.45:
        return "Safe"
    elif tpdi > 0.25:
        return "Warning"
    else:
        return "Dangerous"


def show_stage3(roi, binary, clean, projection, peaks, tpdi, band_count, label, axis_used,
                save_path="outputs/output_stage3.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"STAGE 3 — TPDI: {tpdi:.4f}  |  Label: {label}  |  Bands: {band_count}  |  Axis: {axis_used}",
        fontsize=13, fontweight='bold'
    )

    axes[0, 0].imshow(roi,    cmap='gray')
    axes[0, 0].set_title("ROI")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title("Otsu Binary")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(clean,  cmap='gray')
    axes[1, 0].set_title("Cleaned Shadows (open 5x5, close 25x25)")
    axes[1, 0].axis("off")

    axes[1, 1].plot(projection, label="smoothed projection")
    if peaks:
        axes[1, 1].scatter(peaks, [projection[p] for p in peaks],
                           color='red', zorder=5, label=f"{band_count} peaks")
    axes[1, 1].axhline(0.6, color='gray', linestyle='--', linewidth=0.8, label="threshold=0.6")
    axes[1, 1].set_title(f"{'Row' if axis_used == 'row' else 'Column'} Projection (Peaks = Grooves)")
    axes[1, 1].set_xlabel(f"{'Row' if axis_used == 'row' else 'Column'} index")
    axes[1, 1].set_ylabel("Normalized intensity")
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Stage 3 done → {save_path}")