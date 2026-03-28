import cv2
import numpy as np
import matplotlib.pyplot as plt


def extract_roi(image):
    h, w = image.shape

    # -------------------------------
    # Step 1: Center Crop (MAIN ROI)
    # -------------------------------
    start_h = int(h * 0.2)
    end_h = int(h * 0.8)
    roi = image[start_h:end_h, :]

    # -------------------------------
    # Step 2: Edge Detection (for visualization)
    # -------------------------------
    # Smooth before edge detection (IMPORTANT)
    smooth = cv2.GaussianBlur(roi, (5, 5), 0)
    edges = cv2.Canny(smooth, 70, 180)
    edges = cv2.Canny(roi, 50, 150)

    # -------------------------------
    # Step 3: Morphological Closing (connect edges)
    # -------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    return roi, edges, closed


def show_results(original, roi, edges, closed):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("ROI (Center Crop)")
    plt.imshow(roi, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("Edges")
    plt.imshow(edges, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Morphological Closing")
    plt.imshow(closed, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "data/images/test.jpg"

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found!")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    roi, edges, closed = extract_roi(gray)

    show_results(gray, roi, edges, closed)