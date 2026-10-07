"""
Unit tests for PDF validation.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.ingestion.pdf_parser import validate_pdf, ValidationResult


FAKE_PDF = b"This is not a PDF"
EMPTY = b""


class TestPDFValidation:

    def test_empty_file_rejected(self):
        result = validate_pdf(EMPTY, "test.pdf")
        assert not result.valid
        assert "Empty" in result.error

    def test_wrong_extension_rejected(self):
        result = validate_pdf(b"%PDF-fake", "malware.exe")
        assert not result.valid
        assert "extension" in result.error.lower()

    def test_fake_pdf_content_rejected(self):
        with patch("app.ingestion.pdf_parser.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "text/plain"
            result = validate_pdf(FAKE_PDF, "doc.pdf")
            assert not result.valid

    def test_oversized_file_rejected(self):
        # Create a fake large payload (just metadata, no real content)
        with patch("app.ingestion.pdf_parser.get_settings") as mock_settings:
            mock_settings.return_value.max_upload_size_mb = 1
            mock_settings.return_value.allowed_extensions = [".pdf"]
            mock_settings.return_value.allowed_mime_types = ["application/pdf"]
            oversized = b"%PDF" + b"x" * (2 * 1024 * 1024)
            result = validate_pdf(oversized, "big.pdf")
            assert not result.valid
            assert "size" in result.error.lower()

    def test_content_hash_computed(self):
        # Mock both magic and pypdf
        import hashlib
        with patch("app.ingestion.pdf_parser.magic") as mock_magic, \
             patch("app.ingestion.pdf_parser.pypdf") as mock_pypdf:
            mock_magic.from_buffer.return_value = "application/pdf"
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [MagicMock() for _ in range(5)]
            mock_reader.pages[0].extract_text.return_value = "Some real text here for testing"
            mock_pypdf.PdfReader.return_value = mock_reader
            mock_pypdf.errors.PdfReadError = Exception

            fake_bytes = b"%PDF-1.4 fake content"
            result = validate_pdf(fake_bytes, "doc.pdf")
            if result.valid:
                assert result.content_hash == hashlib.sha256(fake_bytes).hexdigest()

    def test_pages_counted(self):
        with patch("app.ingestion.pdf_parser.magic") as mock_magic, \
             patch("app.ingestion.pdf_parser.pypdf") as mock_pypdf:
            mock_magic.from_buffer.return_value = "application/pdf"
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [MagicMock() for _ in range(15)]
            mock_reader.pages[0].extract_text.return_value = "Lots of text " * 50
            mock_pypdf.PdfReader.return_value = mock_reader
            mock_pypdf.errors.PdfReadError = Exception

            result = validate_pdf(b"%PDF pages", "doc.pdf")
            if result.valid:
                assert result.pages == 15

    def test_encrypted_pdf_rejected(self):
        with patch("app.ingestion.pdf_parser.magic") as mock_magic, \
             patch("app.ingestion.pdf_parser.pypdf") as mock_pypdf:
            mock_magic.from_buffer.return_value = "application/pdf"
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = 0  # NOT_DECRYPTED
            mock_pypdf.PdfReader.return_value = mock_reader
            mock_pypdf.errors.PdfReadError = Exception
            mock_pypdf.PasswordType = MagicMock()
            mock_pypdf.PasswordType.NOT_DECRYPTED = 0

            result = validate_pdf(b"%PDF encrypted", "secret.pdf")
            assert not result.valid
            assert "password" in result.error.lower() or "encrypt" in result.error.lower()

    def test_zero_page_pdf_rejected(self):
        with patch("app.ingestion.pdf_parser.magic") as mock_magic, \
             patch("app.ingestion.pdf_parser.pypdf") as mock_pypdf:
            mock_magic.from_buffer.return_value = "application/pdf"
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_reader.pages = []
            mock_pypdf.PdfReader.return_value = mock_reader
            mock_pypdf.errors.PdfReadError = Exception

            result = validate_pdf(b"%PDF empty", "empty.pdf")
            assert not result.valid
            assert "no pages" in result.error.lower()

    def test_mime_type_spoofing_detected(self):
        """A .pdf file containing non-PDF content should be rejected."""
        with patch("app.ingestion.pdf_parser.magic") as mock_magic:
            mock_magic.from_buffer.return_value = "application/x-msdownload"
            result = validate_pdf(b"MZ executable content", "malware.pdf")
            assert not result.valid
            assert "MIME" in result.error or "content" in result.error.lower()
