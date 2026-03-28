import cv2
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    gray     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred  = cv2.GaussianBlur(enhanced, (5, 5), 0)

    return img, gray, enhanced, blurred


def show_stage1(img, gray, enhanced, blurred, save_path="outputs/output_stage1.png"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle("STAGE 1 — Preprocessing", fontsize=13, fontweight='bold')

    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap='gray')
    axes[1].set_title("Grayscale")
    axes[1].axis("off")

    axes[2].imshow(enhanced, cmap='gray')
    axes[2].set_title("CLAHE")
    axes[2].axis("off")

    axes[3].imshow(blurred, cmap='gray')
    axes[3].set_title("Blurred (output)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.show()
    print(f"✅ Stage 1 done → {save_path}")