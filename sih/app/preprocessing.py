"""
Preprocessing: cleans up phone-photographed / scanned medical documents
before OCR. Handles both image files and PDFs.
"""
import cv2
import numpy as np
from PIL import Image
from pdf2image import convert_from_bytes


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 300) -> list[Image.Image]:
    """Convert a PDF (as bytes) into a list of PIL images, one per page."""
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def deskew(image: np.ndarray) -> np.ndarray:
    """Corrects rotation/tilt common in phone-photographed documents."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def denoise_and_binarize(image: np.ndarray) -> np.ndarray:
    """Grayscale + denoise + adaptive threshold — improves OCR accuracy a lot
    on low-quality phone photos of prescriptions/reports."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    binarized = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    return binarized


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Full pipeline: PIL image -> cleaned, OCR-ready cv2 image."""
    cv_img = pil_to_cv2(image)
    cv_img = deskew(cv_img)
    return denoise_and_binarize(cv_img)
