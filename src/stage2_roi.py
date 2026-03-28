import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_roi(blurred):
    h, w = blurred.shape

    # Primary strategy: center crop — avoids background edge interference
    # and handles angled tyres where contour detection fails
    roi = blurred[int(h * 0.2):int(h * 0.8), :]

    # Edge detection — run on smoothed signal only (not raw roi)
    smooth = cv2.GaussianBlur(roi, (5, 5), 0)
    edges  = cv2.Canny(smooth, 50, 150)

    # Morphological closing to connect groove edges into bands
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Optional refinement: use contour only if a clear horizontal band exists
    refined_roi = roi
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest      = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest)
        aspect_ratio  = cw / ch if ch > 0 else 0
        if aspect_ratio > 2 and cw > w * 0.4:
            refined_roi = roi[y:y+ch, x:x+cw]

    return refined_roi, edges, closed


def show_stage2(blurred, roi, edges, refined_roi, save_path="outputs/output_stage2.png"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("STAGE 2 — ROI Extraction", fontsize=13, fontweight='bold')

    axes[0].imshow(blurred,     cmap='gray')
    axes[0].set_title("Blurred Input")
    axes[0].axis("off")

    axes[1].imshow(roi,         cmap='gray')
    axes[1].set_title("Center Crop ROI")
    axes[1].axis("off")

    axes[2].imshow(edges,       cmap='gray')
    axes[2].set_title("Canny Edges")
    axes[2].axis("off")

    axes[3].imshow(refined_roi, cmap='gray')
    axes[3].set_title("Final ROI")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Stage 2 done → {save_path}")