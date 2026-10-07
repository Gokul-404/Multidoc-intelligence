"""
PDF validation: MIME type, extension, size, structure, and page count.
Returns useful errors rather than crashing.
"""
from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional

try:
    import magic  # python-magic
    HAS_MAGIC = True
except ImportError:
    magic = None
    HAS_MAGIC = False
import pypdf

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Maximum number of pages we will process (prevent DoS)
MAX_PAGES = 2000


@dataclass
class ValidationResult:
    valid: bool
    error: Optional[str] = None
    pages: int = 0
    content_hash: str = ""
    is_encrypted: bool = False
    is_scanned_candidate: bool = False  # Will be refined during extraction


def validate_pdf(file_bytes: bytes, filename: str) -> ValidationResult:
    """
    Comprehensive PDF validation.
    Returns a ValidationResult with details about the file.
    """
    current_settings = get_settings()

    # 1. Zero-byte check
    if not file_bytes:
        return ValidationResult(valid=False, error="Empty file uploaded.")

    # 2. Size check
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > current_settings.max_upload_size_mb:
        return ValidationResult(
            valid=False,
            error=f"File size {size_mb:.1f} MB exceeds limit of {current_settings.max_upload_size_mb} MB.",
        )

    # 3. Extension check
    ext = Path(filename).suffix.lower()
    if ext not in current_settings.allowed_extensions:
        return ValidationResult(
            valid=False,
            error=f"Unsupported file extension '{ext}'. Allowed: {current_settings.allowed_extensions}",
        )

    # 4. MIME type check (prevents extension spoofing)
    try:
        if HAS_MAGIC and magic is not None:
            detected_mime = magic.from_buffer(file_bytes[:2048], mime=True)
            if detected_mime not in current_settings.allowed_mime_types:
                return ValidationResult(
                    valid=False,
                    error=f"File content is '{detected_mime}', expected PDF. Possible MIME spoofing.",
                )
        elif not file_bytes.startswith(b"%PDF"):
            return ValidationResult(valid=False, error="File does not appear to be a valid PDF.")
    except Exception as exc:
        logger.warning("MIME detection failed: %s", exc)
        # Fallback: check PDF magic bytes
        if not file_bytes.startswith(b"%PDF"):
            return ValidationResult(valid=False, error="File does not appear to be a valid PDF.")

    # 5. PDF structure + page count
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes), strict=False)

        if reader.is_encrypted:
            # Attempt empty-password decrypt
            try:
                result = reader.decrypt("")
                if result == pypdf.PasswordType.NOT_DECRYPTED:
                    return ValidationResult(
                        valid=False,
                        error="PDF is password-protected and cannot be processed.",
                    )
            except Exception:
                return ValidationResult(
                    valid=False,
                    error="PDF is encrypted and cannot be processed.",
                )

        pages = len(reader.pages)
        if pages == 0:
            return ValidationResult(valid=False, error="PDF has no pages.")
        if pages > MAX_PAGES:
            return ValidationResult(
                valid=False,
                error=f"PDF has {pages} pages, maximum allowed is {MAX_PAGES}.",
            )

        # Quick scanned-PDF heuristic: if first page has no extractable text
        first_text = ""
        try:
            first_text = reader.pages[0].extract_text() or ""
        except Exception:
            pass
        is_scanned_candidate = len(first_text.strip()) < 50

    except pypdf.errors.PdfReadError as exc:
        return ValidationResult(valid=False, error=f"Corrupted or malformed PDF: {exc}")
    except Exception as exc:
        return ValidationResult(valid=False, error=f"PDF parsing error: {exc}")

    # 6. Content hash (for duplicate detection)
    content_hash = hashlib.sha256(file_bytes).hexdigest()

    return ValidationResult(
        valid=True,
        pages=pages,
        content_hash=content_hash,
        is_scanned_candidate=is_scanned_candidate,
    )
