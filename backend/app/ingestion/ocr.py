"""
OCR subsystem for SentinelRAG.
Uses Tesseract (via pytesseract) with automatic fallback detection.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Optional

import pypdf

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Minimum characters per page to consider text "sufficient"
MIN_CHARS_PER_PAGE = 30


@dataclass
class PageOCRResult:
    page_number: int      # 1-indexed
    text: str
    confidence: Optional[float] = None
    ocr_used: bool = False


@dataclass
class OCRResult:
    pages: list[PageOCRResult] = field(default_factory=list)
    ocr_used: bool = False
    avg_confidence: Optional[float] = None
    low_quality: bool = False  # True if OCR confidence is very poor


def _try_tesseract(page_image) -> tuple[str, float]:
    """Run Tesseract on a PIL Image and return (text, confidence)."""
    import pytesseract

    data = pytesseract.image_to_data(
        page_image,
        output_type=pytesseract.Output.DICT,
        config="--psm 3",
    )
    texts = []
    confidences = []
    for i, conf in enumerate(data["conf"]):
        try:
            c = int(conf)
        except (ValueError, TypeError):
            continue
        if c > 0:
            word = data["text"][i].strip()
            if word:
                texts.append(word)
                confidences.append(c)
    text = " ".join(texts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf / 100.0  # Normalise to [0,1]


def run_ocr(pdf_bytes: bytes) -> OCRResult:
    """
    Attempt to extract text from each PDF page.
    If a page has insufficient text, run Tesseract OCR on it.
    """
    try:
        import fitz as pymupdf  # PyMuPDF for rendering
    except ImportError:
        logger.error("PyMuPDF (fitz) not installed; OCR unavailable")
        return OCRResult()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages: list[PageOCRResult] = []
    any_ocr = False
    confidences: list[float] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # type: ignore[call-arg]

        if len(text.strip()) >= MIN_CHARS_PER_PAGE:
            pages.append(PageOCRResult(page_number=page_num + 1, text=text))
        else:
            # Render page to image and OCR
            any_ocr = True
            try:
                pix = page.get_pixmap(dpi=200)  # type: ignore[attr-defined]
                img_bytes = pix.tobytes("png")
                from PIL import Image

                pil_img = Image.open(io.BytesIO(img_bytes))
                ocr_text, conf = _try_tesseract(pil_img)
                confidences.append(conf)
                pages.append(
                    PageOCRResult(
                        page_number=page_num + 1,
                        text=ocr_text,
                        confidence=conf,
                        ocr_used=True,
                    )
                )
                if conf < 0.3:
                    logger.warning(
                        "Low OCR confidence (%.0f%%) on page %d", conf * 100, page_num + 1
                    )
            except Exception as exc:
                logger.error("OCR failed on page %d: %s", page_num + 1, exc)
                pages.append(PageOCRResult(page_number=page_num + 1, text=""))

    doc.close()

    avg_conf = sum(confidences) / len(confidences) if confidences else None
    low_quality = bool(avg_conf is not None and avg_conf < 0.4)

    return OCRResult(
        pages=pages,
        ocr_used=any_ocr,
        avg_confidence=avg_conf,
        low_quality=low_quality,
    )
