import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def extract_texture_features(roi):
    # Resize to fixed dimensions for consistent feature extraction
    roi_resized = cv2.resize(roi, (128, 64))

    # -----------------------------------------------
    # GLCM — 4 angles, averaged
    # -----------------------------------------------
    glcm = graycomatrix(
        roi_resized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=256,
        symmetric=True,
        normed=True
    )

    glcm_features = np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean()
    ])

    # -----------------------------------------------
    # LBP — radius=3, n_points=24, uniform
    # -----------------------------------------------
    lbp      = local_binary_pattern(roi_resized, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)

    # Final feature vector: 5 GLCM + 26 LBP = 31 features
    feature_vector = np.concatenate([glcm_features, lbp_hist])

    return feature_vector, glcm_features, lbp_hist


def show_stage4(roi, glcm_features, lbp_hist, save_path="outputs/output_stage4.png"):
    roi_resized = cv2.resize(roi, (128, 64))
    lbp_image   = local_binary_pattern(roi_resized, P=24, R=3, method='uniform')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("STAGE 4 — Texture Features (GLCM + LBP)", fontsize=13, fontweight='bold')

    axes[0].imshow(roi,       cmap='gray')
    axes[0].set_title("ROI")
    axes[0].axis("off")

    axes[1].imshow(lbp_image, cmap='gray')
    axes[1].set_title("LBP Image")
    axes[1].axis("off")

    axes[2].bar(range(26), lbp_hist)
    axes[2].set_title("LBP Histogram (26 bins)")
    axes[2].set_xlabel("LBP bin")
    axes[2].set_ylabel("Normalized frequency")

    feature_labels = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for i, (lbl, val) in enumerate(zip(feature_labels, glcm_features)):
        axes[2].text(0.98, 0.95 - i * 0.10, f"{lbl}: {val:.4f}",
                     transform=axes[2].transAxes, ha='right', fontsize=7, color='darkred')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Stage 4 done → {save_path}")