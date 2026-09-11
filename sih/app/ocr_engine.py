"""
OCR engine: runs Tesseract over preprocessed pages and returns raw text
plus per-word confidence (useful for flagging low-confidence extractions
for human review — important for a medical document, don't silently trust
garbage OCR on a drug dosage).
"""
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from pytesseract import Output
import numpy as np


def ocr_image(cv_image: np.ndarray, lang: str = "eng") -> dict:
    """
    Returns:
        {
          "text": full extracted text,
          "avg_confidence": float 0-100,
          "low_confidence_words": [words below 60% conf, for review flagging]
        }
    """
    data = pytesseract.image_to_data(cv_image, lang=lang, output_type=Output.DICT)

    words, confidences, low_conf_words = [], [], []
    for word, conf in zip(data["text"], data["conf"]):
        word = word.strip()
        if not word:
            continue
        conf = float(conf)
        words.append(word)
        if conf >= 0:
            confidences.append(conf)
        if 0 <= conf < 60:
            low_conf_words.append(word)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": " ".join(words),
        "avg_confidence": round(avg_conf, 2),
        "low_confidence_words": low_conf_words,
    }


def ocr_pages(cv_images: list[np.ndarray], lang: str = "eng") -> dict:
    """OCR multiple pages (for multi-page PDFs) and merge results."""
    full_text_parts = []
    all_low_conf = []
    confs = []

    for img in cv_images:
        result = ocr_image(img, lang=lang)
        full_text_parts.append(result["text"])
        all_low_conf.extend(result["low_confidence_words"])
        confs.append(result["avg_confidence"])

    return {
        "text": "\n\n".join(full_text_parts),
        "avg_confidence": round(sum(confs) / len(confs), 2) if confs else 0.0,
        "low_confidence_words": all_low_conf,
        "needs_review": (sum(confs) / len(confs) if confs else 0) < 70,
    }
