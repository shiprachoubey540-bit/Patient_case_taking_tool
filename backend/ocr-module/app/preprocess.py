"""Task 1 - OpenCV prescription image preprocessing."""
from __future__ import annotations

import cv2
import numpy as np


def read_image(image_bytes: bytes) -> np.ndarray:
    """Step 1: turn raw file bytes into an OpenCV image."""
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not read image (is it a valid JPG/PNG?)")
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Step 2: colour -> shades of grey."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def otsu_threshold(gray: np.ndarray) -> np.ndarray:
    """Step 3: black text on white paper. OpenCV picks the cut-off itself."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def remove_ink_noise(binary: np.ndarray, min_blob_area: int = 15) -> np.ndarray:
    """Step 4: paint over black specks smaller than min_blob_area pixels."""
    inverted = cv2.bitwise_not(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    cleaned = inverted.copy()
    for label in range(1, count):  # label 0 is the background
        if stats[label, cv2.CC_STAT_AREA] < min_blob_area:
            cleaned[labels == label] = 0
    return cv2.bitwise_not(cleaned)


def preprocess(image_bytes: bytes) -> bytes:
    """Run all four steps and return the clean image as PNG bytes."""
    image = read_image(image_bytes)
    gray = to_grayscale(image)
    binary = otsu_threshold(gray)
    clean = remove_ink_noise(binary)
    ok, png = cv2.imencode(".png", clean)
    if not ok:
        raise ValueError("could not encode cleaned image")
    return png.tobytes()


if __name__ == "__main__":
    # Try from the terminal:  python -m app.preprocess in.jpg out.png
    import sys

    src, dst = sys.argv[1], sys.argv[2]
    with open(src, "rb") as f:
        cleaned = preprocess(f.read())
    with open(dst, "wb") as f:
        f.write(cleaned)
    print(f"saved cleaned image to {dst}")