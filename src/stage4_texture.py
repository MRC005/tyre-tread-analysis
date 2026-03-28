import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import matplotlib.pyplot as plt


def extract_texture_features(image):
    # -------------------------------
    # Ensure grayscale
    # -------------------------------
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # -------------------------------
    # GLCM
    # -------------------------------
    glcm = graycomatrix(gray, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # -------------------------------
    # LBP
    # -------------------------------
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    # -------------------------------
    # Convert to dictionary
    # -------------------------------
    glcm_dict = {
        "contrast": contrast,
        "dissimilarity": dissimilarity,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation
    }

    # Feature vector
    feat_vec = [
        contrast,
        dissimilarity,
        homogeneity,
        energy,
        correlation
    ]

    return feat_vec, glcm_dict, lbp_hist, lbp


# -------------------------------
# SHOW FUNCTION (IMPORTANT)
# -------------------------------
def show_stage4(roi, glcm_feats, lbp_hist, lbp_img,
                save_path="outputs/output_stage4.png"):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    fig.suptitle("Stage 4 — Texture Analysis (GLCM + LBP)",
                 fontsize=12, fontweight='bold')

    axes[0].imshow(roi, cmap='gray')
    axes[0].set_title("Tread ROI")
    axes[0].axis("off")

    axes[1].imshow(lbp_img, cmap='gray')
    axes[1].set_title("LBP Image")
    axes[1].axis("off")

    axes[2].plot(lbp_hist)
    axes[2].set_title("LBP Histogram")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"  [Stage 4] ✅ Saved → {save_path}")