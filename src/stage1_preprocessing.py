import cv2
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    # Read image
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Image not found. Check path!")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE (improves local contrast)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    # Apply Gaussian Blur (removes noise)
    blurred = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    return img, gray, clahe_img, blurred


def show_results(original, gray, clahe_img, blurred):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.title("CLAHE")
    plt.imshow(clahe_img, cmap='gray')
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.title("Blurred")
    plt.imshow(blurred, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "data/images/test.jpg"

    original, gray, clahe_img, blurred = preprocess_image(image_path)
    show_results(original, gray, clahe_img, blurred)