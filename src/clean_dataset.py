import cv2
import os
import shutil
import numpy as np

def is_valid_tyre_image(path):
    img = cv2.imread(path)
    if img is None:
        return False

    h, w = img.shape[:2]

    # Too small
    if h < 100 or w < 100:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Too dark / bright
    if gray.mean() < 30 or gray.mean() > 220:
        return False

    # Blur detection
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if lap_var < 80:
        return False

    # Aspect ratio
    ratio = max(h, w) / min(h, w)
    if ratio > 4.0:
        return False

    # Texture check
    if gray.std() < 15:
        return False

    return True


base = "data"

for label in ["good", "bad"]:
    src = os.path.join(base, label)
    clean = os.path.join(base, "clean", label)

    os.makedirs(clean, exist_ok=True)

    kept = 0
    removed = 0

    for fname in os.listdir(src):
        fpath = os.path.join(src, fname)

        if is_valid_tyre_image(fpath):
            shutil.copy(fpath, os.path.join(clean, fname))
            kept += 1
        else:
            removed += 1

    print(f"{label}: kept {kept}, removed {removed}")