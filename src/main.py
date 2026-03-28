"""
TYRE TREAD WEAR SEVERITY ESTIMATION
=====================================
Run this file from the project ROOT folder:

    cd tyre-tread-project
    python src/main.py

Output images saved to: outputs/
"""

import sys
import os

# Resolve absolute paths so this works from any working directory
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

# Add src/ to path so stage imports work
sys.path.insert(0, SRC_DIR)

from stage1_preprocessing import preprocess_image, show_stage1
from stage2_roi           import extract_roi,      show_stage2
from stage3_tpdi          import compute_tpdi,     classify_tpdi, show_stage3
from stage4_texture       import extract_texture_features, show_stage4

# ════════════════════════════════════════════
#  SET YOUR IMAGE PATH HERE
# ════════════════════════════════════════════
IMAGE_PATH = os.path.join(ROOT_DIR, "data", "images", "test.jpg")


def fusion_decision(tpdi, band_count, texture_features=None, knn_model=None):
    """
    Stage 5 — Fusion Decision
    band_count >= 3 → trust TPDI (shadow path reliable)
    band_count <  3 → fall back to KNN texture classifier
    """
    if band_count >= 3:
        if tpdi > 0.45:
            label = "Safe"
        elif tpdi > 0.25:
            label = "Warning"
        else:
            label = "Dangerous"
        method = "Shadow-based (TPDI)"
    else:
        if knn_model is not None and texture_features is not None:
            pred   = knn_model.predict([texture_features])[0]
            label  = {0: "Safe", 1: "Warning", 2: "Dangerous"}.get(pred, "Unknown")
            method = "Texture-based (KNN)"
        else:
            # KNN not trained yet — report texture fallback needed
            label  = "Indeterminate (train KNN for texture fallback)"
            method = "KNN not loaded"

    return label, method


def main():
    print(f"\n{'='*55}")
    print(f"  TYRE TREAD WEAR SEVERITY ESTIMATION PIPELINE")
    print(f"{'='*55}")
    print(f"  Image: {IMAGE_PATH}\n")

    # ── Stage 1: Preprocessing ──────────────────────────────
    print("Running Stage 1 — Preprocessing...")
    img, gray, enhanced, blurred = preprocess_image(IMAGE_PATH)
    show_stage1(img, gray, enhanced, blurred,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage1.png"))

    # ── Stage 2: ROI Extraction ─────────────────────────────
    print("\nRunning Stage 2 — ROI Extraction...")
    refined_roi, edges, closed = extract_roi(blurred)
    show_stage2(blurred, blurred[int(blurred.shape[0]*0.2):int(blurred.shape[0]*0.8), :],
                edges, refined_roi,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage2.png"))

    # ── Stage 3: TPDI Computation ───────────────────────────
    print("\nRunning Stage 3 — TPDI Computation...")
    tpdi, binary, clean, projection, peaks, band_count, spacing, avg_shadow, axis_used = compute_tpdi(refined_roi)
    label_s3 = classify_tpdi(tpdi)
    show_stage3(refined_roi, binary, clean, projection, peaks,
                tpdi, band_count, label_s3, axis_used,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage3.png"))

    # ── Stage 4: Texture Features ───────────────────────────
    print("\nRunning Stage 4 — Texture Feature Extraction...")
    feature_vector, glcm_features, lbp_hist = extract_texture_features(refined_roi)
    show_stage4(refined_roi, glcm_features, lbp_hist,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage4.png"))

    # ── Stage 5: Fusion Decision ────────────────────────────
    label, method = fusion_decision(tpdi, band_count, feature_vector)

    print(f"\n{'='*55}")
    print(f"  STAGE 5 — FINAL RESULT")
    print(f"{'='*55}")
    print(f"  Groove spacing  : {spacing:.2f} px")
    print(f"  Avg shadow width: {avg_shadow:.2f} px")
    print(f"  TPDI value      : {tpdi:.4f}")
    print(f"  Bands detected  : {band_count}")
    print(f"  Axis used       : {axis_used}")
    print(f"  Method used     : {method}")
    print(f"  ➜  RESULT       : {label}")
    print(f"{'='*55}\n")
    print("Screenshot all 4 output figures + paste terminal output for review.")


if __name__ == "__main__":
    main()