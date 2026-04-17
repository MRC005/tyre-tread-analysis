import cv2
import os
import numpy as np

def augment_image(img):
    variants = []

    # Flip
    variants.append(cv2.flip(img, 1))

    # Rotation
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(
            (img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        variants.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))

    # Brightness
    for factor in [0.8, 1.2]:
        bright = np.clip(img * factor, 0, 255).astype(np.uint8)
        variants.append(bright)

    # Noise
    noise = np.random.normal(0, 8, img.shape).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    variants.append(noisy)

    return variants


src_base = "data/clean"
dst_base = "data/augmented"

for label in ["good", "bad"]:
    src = os.path.join(src_base, label)
    dst = os.path.join(dst_base, label)

    os.makedirs(dst, exist_ok=True)

    for fname in os.listdir(src):
        img = cv2.imread(os.path.join(src, fname))
        if img is None:
            continue

        # Original
        cv2.imwrite(os.path.join(dst, fname), img)

        # Augmented
        for i, aug in enumerate(augment_image(img)):
            new_name = f"{fname.split('.')[0]}_aug{i}.jpg"
            cv2.imwrite(os.path.join(dst, new_name), aug)

    print(f"{label} done")