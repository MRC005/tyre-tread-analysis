"""
main.py — Single-Image Tyre Tread Wear Severity Estimation
===========================================================
Run: python src/main.py

This runs the complete 5-stage pipeline on ONE test image and
prints the final severity label with full diagnostic output.

To process the full dataset and train KNN, run:
  python src/batch_process.py
"""

import sys
import os

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
IMAGE_PATH  = os.path.join(ROOT_DIR, "data", "images", "test.jpg")

sys.path.insert(0, SRC_DIR)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

from stage1_preprocessing import preprocess_image, show_stage1
from stage2_roi           import extract_roi,      show_stage2
from stage3_tsci          import compute_tsci,     show_stage3
from stage4_texture       import extract_texture_features, show_stage4
from stage5_fusion        import fusion_decision,  load_knn


def main():
    print("\n" + "=" * 60)
    print("  TYRE TREAD WEAR SEVERITY ESTIMATION")
    print("  Classical Digital Image Processing Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1 — Preprocessing & Enhancement (M2)
    # ------------------------------------------------------------------
    print("\n  [Stage 1] Preprocessing ...")
    img, gray, enhanced, blurred = preprocess_image(IMAGE_PATH)
    show_stage1(img, gray, enhanced, blurred,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage1.png"))

    # ------------------------------------------------------------------
    # Stage 2 — ROI Extraction (M5)
    # ------------------------------------------------------------------
    print("  [Stage 2] ROI Extraction ...")
    h = blurred.shape[0]
    coarse_roi            = blurred[int(h * 0.20): int(h * 0.80), :]
    refined_roi, edges, closed = extract_roi(blurred)
    show_stage2(blurred, coarse_roi, edges, refined_roi,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage2.png"))

    # ------------------------------------------------------------------
    # Stage 3 — TSCI via 2-D DFT (M3)  ← Core Novel Contribution
    # ------------------------------------------------------------------
    print("  [Stage 3] Computing TSCI (2-D DFT energy ratio) ...")
    (tsci, label_s3, roi_resized, log_mag,
     hf_mask, e_hf, e_total, radial_prof) = compute_tsci(refined_roi)

    show_stage3(roi_resized, log_mag, hf_mask, tsci, label_s3,
                e_hf, e_total, radial_prof,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage3.png"))

    # ------------------------------------------------------------------
    # Stage 4 — Texture Feature Extraction (M6)
    # ------------------------------------------------------------------
    print("  [Stage 4] Extracting GLCM + LBP texture features ...")
    feat_vec, glcm_feats, lbp_hist, lbp_img = extract_texture_features(refined_roi)
    show_stage4(refined_roi, glcm_feats, lbp_hist, lbp_img,
                save_path=os.path.join(OUTPUTS_DIR, "output_stage4.png"))

    # ------------------------------------------------------------------
    # Stage 5 — Hybrid Fusion Decision
    # ------------------------------------------------------------------
    print("  [Stage 5] Fusion decision ...")
    knn, scaler = load_knn(save_dir=OUTPUTS_DIR)
    label, method = fusion_decision(tsci, feat_vec, knn, scaler)

    # ------------------------------------------------------------------
    # Final Result
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STAGE 5 — FINAL RESULT")
    print("=" * 60)
    print(f"  TSCI value         : {tsci:.4f}")
    print(f"  HF energy          : {e_hf:.2f}")
    print(f"  Total energy       : {e_total:.2f}")
    print(f"  Decision method    : {method}")
    print(f"  ➜  SEVERITY LABEL  : {label}")
    print("=" * 60)

    if knn is None:
        print("\n  ℹ️  KNN not trained yet.")
        print("     Run:  python src/batch_process.py")
        print("     to process the full dataset and enable the texture fallback.\n")


if __name__ == "__main__":
    main()


# 1.
# -Preprocessing of Image using CLAHE and Gaussian blur to normalize lighting and remove noise.
# -extract the tread region using edge detection and morphological operations
# -computation TSCI using DFT, which measures high-frequency energy — sharper grooves give higher values.
# -extract texture features using GLCM and LBP.

# python src/main.py

# 2.RUNNING FULL DATASET
# python src/batch_process.py


# 3.TRAINING + EVALUATING MODEL
# python src/train_and_evaluate.py