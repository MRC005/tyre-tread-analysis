"""
STAGE 3 — Tyre Surface Clarity Index (TSCI) via 2-D DFT
=========================================================
Module : M3 (Frequency-Domain Analysis)
Purpose: Quantify tread wear severity through the spatial-frequency
         energy distribution of the tread-band ROI.

Core Novel Contribution
-----------------------
We define the Tyre Surface Clarity Index (TSCI):

        TSCI = E_HF / E_total

where
  E_HF    = sum of 2-D DFT magnitude coefficients whose spatial
            frequency radius exceeds a threshold r_th
  E_total = sum of ALL 2-D DFT magnitude coefficients (excl. DC)

Physical Justification
----------------------
A new tyre has deep, sharp grooves that produce strong high-spatial-
frequency edges in the tread image. As the tyre wears, grooves become
shallower and the surface flattens, attenuating high-frequency energy
and concentrating spectral power in the low-frequency (DC) band.
TSCI therefore DECREASES monotonically with tread wear — a physically
well-founded and lighting-invariant metric.

Unlike shadow-width methods (which depend on illumination angle and
camera distance), 2-D DFT energy ratios are insensitive to uniform
brightness changes and moderate perspective distortions.

Classification thresholds (empirically determined from dataset;
see calibration notebook):
  TSCI > T_safe    → Safe
  T_warn < TSCI ≤ T_safe → Warning
  TSCI ≤ T_warn    → Dangerous

Steps
-----
1. Resize ROI to fixed 256×128 (standardises spatial frequency scale)
2. 2-D FFT → shift DC to centre → log-magnitude spectrum
3. Build annular high-frequency mask (r > r_th = 20 % of min dimension)
4. Compute TSCI = E_HF / E_total
5. Classify using empirically set thresholds
6. Visualise: ROI | log spectrum | frequency mask | radial energy profile
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Tunable constants — adjust after dataset calibration
# ---------------------------------------------------------------------------
ROI_W        = 256       # fixed resize width
ROI_H        = 128       # fixed resize height
FREQ_RADIUS  = 0.20      # high-freq threshold as fraction of min(H, W)
TSCI_SAFE    = 0.55      # TSCI > this → Safe      (calibrate on dataset)
TSCI_WARN    = 0.35      # TSCI > this → Warning   (calibrate on dataset)


def _build_hf_mask(h: int, w: int, radius_frac: float) -> np.ndarray:
    """
    Build a binary annular mask in the frequency domain.
    Pixels OUTSIDE the central low-frequency disc = 1 (high frequency).

    Parameters
    ----------
    h, w         : image height and width (after resize)
    radius_frac  : threshold radius as a fraction of min(h, w)

    Returns
    -------
    mask : ndarray bool (h, w)  — True where high-frequency
    """
    cy, cx = h // 2, w // 2
    r_th   = radius_frac * min(h, w)
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    return dist > r_th


def compute_tsci(roi: np.ndarray):
    """
    Compute the Tyre Surface Clarity Index (TSCI) for a tread ROI.

    Parameters
    ----------
    roi : ndarray (h, w)   Grayscale tread ROI from Stage 2.

    Returns
    -------
    tsci        : float   TSCI value in [0, 1]
    label       : str     'Safe' | 'Warning' | 'Dangerous'
    roi_resized : ndarray standardised ROI used for DFT
    log_mag     : ndarray log-magnitude spectrum (shifted)
    hf_mask     : ndarray high-frequency binary mask
    e_hf        : float   high-frequency energy
    e_total     : float   total energy (excl. DC)
    radial_prof : ndarray mean magnitude vs radius (for visualisation)
    """
    # Step 1: Standardise size
    roi_resized = cv2.resize(roi, (ROI_W, ROI_H),
                             interpolation=cv2.INTER_AREA)

    # Step 2: 2-D FFT
    f_complex = np.fft.fft2(roi_resized.astype(np.float64))
    f_shifted = np.fft.fftshift(f_complex)
    magnitude = np.abs(f_shifted)

    # Suppress DC component for energy ratio (avoid DC domination)
    h, w     = magnitude.shape
    cy, cx   = h // 2, w // 2
    magnitude[cy, cx] = 0.0

    # Log-magnitude for visualisation
    log_mag = np.log1p(magnitude)

    # Step 3: High-frequency mask
    hf_mask = _build_hf_mask(h, w, FREQ_RADIUS)

    # Step 4: TSCI
    e_total = float(np.sum(magnitude))
    e_hf    = float(np.sum(magnitude[hf_mask]))
    tsci    = e_hf / e_total if e_total > 0 else 0.0

    # Step 5: Classification
    if tsci > TSCI_SAFE:
        label = "Safe"
    elif tsci > TSCI_WARN:
        label = "Warning"
    else:
        label = "Dangerous"

    # Radial energy profile (for visualisation / paper figure)
    Y, X    = np.ogrid[:h, :w]
    dist    = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    r_max   = int(np.max(dist))
    radial_prof = np.array([
        np.mean(magnitude[dist == r]) if np.any(dist == r) else 0.0
        for r in range(r_max + 1)
    ])

    return tsci, label, roi_resized, log_mag, hf_mask, e_hf, e_total, radial_prof


def classify_tsci(tsci: float) -> str:
    """Standalone classifier (used in fusion stage)."""
    if tsci > TSCI_SAFE:
        return "Safe"
    elif tsci > TSCI_WARN:
        return "Warning"
    else:
        return "Dangerous"


def show_stage3(roi_resized, log_mag, hf_mask, tsci, label,
                e_hf, e_total, radial_prof,
                save_path="outputs/output_stage3.png"):
    """Visualise and save Stage 3 outputs as a 2×2 figure."""
    h, w = log_mag.shape
    cy, cx = h // 2, w // 2
    r_th_px = FREQ_RADIUS * min(h, w)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Stage 3 — TSCI = {tsci:.4f}  |  Label: {label}  |  "
        f"E_HF/E_total = {e_hf:.1f}/{e_total:.1f}",
        fontsize=12, fontweight='bold'
    )

    # Panel 1: Standardised ROI
    axes[0, 0].imshow(roi_resized, cmap='gray')
    axes[0, 0].set_title("Standardised Tread ROI (256×128)", fontsize=10)
    axes[0, 0].axis("off")

    # Panel 2: Log-magnitude DFT spectrum
    axes[0, 1].imshow(log_mag, cmap='inferno')
    circle = plt.Circle((cx, cy), r_th_px, color='cyan',
                         fill=False, linewidth=1.5,
                         label=f"HF boundary (r={r_th_px:.0f}px)")
    axes[0, 1].add_patch(circle)
    axes[0, 1].set_title("Log-Magnitude DFT Spectrum", fontsize=10)
    axes[0, 1].legend(fontsize=8, loc='lower right')
    axes[0, 1].axis("off")

    # Panel 3: High-frequency mask
    overlay = log_mag / (log_mag.max() + 1e-9)
    overlay_rgb = np.stack([overlay] * 3, axis=-1)
    overlay_rgb[hf_mask, 0] = 1.0   # red channel highlights HF region
    overlay_rgb[hf_mask, 1] = 0.0
    overlay_rgb[hf_mask, 2] = 0.0
    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title(
        f"HF Mask (r > {FREQ_RADIUS*100:.0f}% of min-dim)\n"
        f"Red = high-frequency energy region", fontsize=9
    )
    axes[1, 0].axis("off")

    # Panel 4: Radial energy profile
    axes[1, 1].plot(radial_prof, color='steelblue', linewidth=1.2)
    axes[1, 1].axvline(r_th_px, color='red', linestyle='--',
                       linewidth=1.2, label=f"HF threshold (r={r_th_px:.0f})")
    axes[1, 1].set_title("Mean Radial Energy Profile", fontsize=10)
    axes[1, 1].set_xlabel("Spatial frequency radius (px)", fontsize=9)
    axes[1, 1].set_ylabel("Mean DFT magnitude", fontsize=9)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Stage 3] ✅ Saved → {save_path}")
    print(f"  [Stage 3]    TSCI       = {tsci:.4f}")
    print(f"  [Stage 3]    E_HF       = {e_hf:.2f}")
    print(f"  [Stage 3]    E_total    = {e_total:.2f}")
    print(f"  [Stage 3]    Label      = {label}")