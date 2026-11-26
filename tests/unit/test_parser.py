import hashlib
import io
import sys
import types

import fitz
import pytest
from docx import Document

from app.services.parser import (
    CVValidationError,
    count_non_pii_text_characters,
    parse_cv,
    scrub_pii,
    validate_and_hash_cv,
)


def make_docx(*, long_text: bool = True) -> bytes:
    document = Document()
    document.add_heading("Skills", level=1)
    document.add_paragraph("Python, FastAPI, PostgreSQL, Docker")
    document.add_heading("Experience", level=1)
    text = (
        "Built and maintained backend APIs, automated tests, and deployment pipelines. "
        "Worked with product and engineering teams to deliver reliable services."
        if long_text
        else "Short CV"
    )
    document.add_paragraph(text)
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "An ordinary CV PDF")
    content = document.tobytes()
    document.close()
    return content


def make_encrypted_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Confidential CV content")
    content = document.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()
    return content


def test_docx_is_detected_and_sha256_is_computed_from_original_bytes():
    content = make_docx()
    validated = validate_and_hash_cv(content, "candidate.docx")
    assert validated.file_type == "docx"
    assert validated.cv_hash == hashlib.sha256(content).hexdigest()
    assert len(validated.cv_hash) == 64


def test_pdf_conversion_uses_pymupdf4llm_without_implicit_ocr(monkeypatch):
    calls = {}
    fake_module = types.ModuleType("pymupdf4llm")

    def to_markdown(document, **kwargs):
        calls.update(kwargs)
        assert document.page_count == 1
        return "## Experience\nBuilt an API service and maintained production systems."

    fake_module.to_markdown = to_markdown
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake_module)

    result = parse_cv(make_pdf(), "candidate.pdf")
    assert result.file_type == "pdf"
    assert "## Experience" in result.markdown
    assert calls["show_progress"] is False
    assert calls["use_ocr"] is False


def test_docx_conversion_preserves_headings_and_body_text():
    result = parse_cv(make_docx(), "candidate.docx")
    assert "## Skills" in result.markdown
    assert "## Experience" in result.markdown
    assert "Python, FastAPI" in result.markdown
    assert result.file_type == "docx"


def test_extension_cannot_override_real_file_signature():
    with pytest.raises(CVValidationError) as exc:
        validate_and_hash_cv(make_docx(), "candidate.pdf")
    assert exc.value.status_code == 415


def test_unsupported_extension_is_rejected():
    with pytest.raises(CVValidationError) as exc:
        validate_and_hash_cv(make_docx(), "candidate.doc")
    assert exc.value.status_code == 415


def test_oversized_file_is_rejected_before_parsing():
    with pytest.raises(CVValidationError) as exc:
        validate_and_hash_cv(b"x" * 101, "candidate.pdf", max_file_size_bytes=100)
    assert exc.value.status_code == 413


def test_encrypted_pdf_is_rejected():
    with pytest.raises(CVValidationError, match="encrypted") as exc:
        validate_and_hash_cv(make_encrypted_pdf(), "candidate.pdf")
    assert exc.value.status_code == 422


def test_docx_expansion_limit_is_enforced():
    content = make_docx()
    with pytest.raises(CVValidationError) as exc:
        validate_and_hash_cv(
            content,
            "candidate.docx",
            max_docx_uncompressed_bytes=1,
        )
    assert exc.value.status_code == 413


def test_non_docx_zip_is_not_accepted_as_docx():
    import zipfile

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("hello.txt", "not a Word document")
    with pytest.raises(CVValidationError):
        validate_and_hash_cv(stream.getvalue(), "candidate.docx")


def test_pii_scrubber_removes_email_url_phone_and_11_digit_id():
    raw = (
        "Email: person@example.com\n"
        "Website: https://example.com/profile\n"
        "Mobile: +963 944 123 456\n"
        "National ID: 12345678901\n"
        "Alternate mobile: 0944123456"
    )
    scrubbed = scrub_pii(raw)
    assert "person@example.com" not in scrubbed
    assert "https://example.com/profile" not in scrubbed
    assert "+963 944 123 456" not in scrubbed
    assert "12345678901" not in scrubbed
    assert "0944123456" not in scrubbed
    assert "[REDACTED_EMAIL]" in scrubbed
    assert "[REDACTED_URL]" in scrubbed
    assert "[REDACTED_PHONE]" in scrubbed
    assert "[REDACTED_ID]" in scrubbed
    assert count_non_pii_text_characters(scrubbed) > 0
    assert count_non_pii_text_characters(scrub_pii("12345678901 0944123456")) == 0


def test_pii_scrubber_does_not_destroy_common_technology_names():
    skills_text = "ASP.NET Core, VB.NET, Node.js, C++, React.js"
    assert scrub_pii(skills_text) == skills_text
