import os
import pandas as pd

from stage1_preprocessing import preprocess_image
from stage2_roi import extract_roi
from stage3_tsci import compute_tsci
from stage4_texture import extract_texture_features


DATA_DIR = "data"
OUTPUT_CSV = "outputs/results.csv"


def process_image(image_path, folder_label):
    try:
        # -------------------------------
        # Stage 1 — Preprocessing
        # -------------------------------
        img, gray, enhanced, blurred = preprocess_image(image_path)

        # -------------------------------
        # Stage 2 — ROI Extraction
        # -------------------------------
        roi, _, _ = extract_roi(blurred)

        # -------------------------------
        # Stage 3 — TSCI
        # -------------------------------
        tsci, _, *_ = compute_tsci(roi)

        # -------------------------------
        # Stage 4 — Texture Features (GLCM + LBP)
        # -------------------------------
        feat_vec, glcm_feats, lbp_hist, lbp_img = extract_texture_features(roi)

        # -------------------------------
        # Save all features
        # -------------------------------
        return {
            "filename": os.path.basename(image_path),
            "folder_label": folder_label,
            "tsci": round(tsci, 4),

            # GLCM features
            "contrast":      glcm_feats["contrast"],
            "dissimilarity": glcm_feats["dissimilarity"],
            "homogeneity":   glcm_feats["homogeneity"],
            "energy":        glcm_feats["energy"],
            "correlation":   glcm_feats["correlation"],

            # Edge feature (NEW)
            "edge_density":  glcm_feats["edge_density"],   # ← THIS WAS MISSING
        }

    except Exception as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None


def main():
    results = []

    # -----------------------------------
    # Loop through dataset folders
    # -----------------------------------
    for label in ["good", "bad"]:
        folder_path = os.path.join(DATA_DIR, label)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        print(f"\nProcessing {label} images...")

        for file in os.listdir(folder_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(folder_path, file)

                result = process_image(image_path, label)
                if result:
                    results.append(result)

    # -----------------------------------
    # Save results to CSV
    # -----------------------------------
    df = pd.DataFrame(results)

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ DONE!")
    print(f"Saved results → {OUTPUT_CSV}")
    print(f"Total images processed: {len(df)}")


if __name__ == "__main__":
    main()