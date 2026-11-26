from __future__ import annotations

import hashlib
import inspect
import io
import logging
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from app.core.config import settings

logger = logging.getLogger(__name__)

SupportedFileType = Literal["pdf", "docx"]


class CVValidationError(ValueError):
    """An upload rejected before document conversion."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


class CVParsingError(RuntimeError):
    """A supported and hashed file could not be converted to Markdown."""

    def __init__(self, message: str, *, cv_hash: str) -> None:
        super().__init__(message)
        self.cv_hash = cv_hash


@dataclass(frozen=True, slots=True)
class ValidatedCV:
    content: bytes
    filename: str
    file_type: SupportedFileType
    cv_hash: str


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    cv_hash: str
    file_type: SupportedFileType
    markdown: str

    @property
    def non_whitespace_char_count(self) -> int:
        return len(re.sub(r"\s+", "", self.markdown))


def validate_and_hash_cv(
    content: bytes,
    filename: str,
    *,
    max_file_size_bytes: int | None = None,
    max_docx_uncompressed_bytes: int | None = None,
    max_docx_members: int | None = None,
) -> ValidatedCV:
    """Validate file size/type/container and return its SHA-256 digest.

    The client-provided Content-Type header is intentionally not accepted as a
    source of truth. The extension must agree with the detected file signature.
    """

    size_limit = max_file_size_bytes or settings.CV_MAX_FILE_SIZE_BYTES
    uncompressed_limit = (
        max_docx_uncompressed_bytes or settings.CV_MAX_DOCX_UNCOMPRESSED_BYTES
    )
    member_limit = max_docx_members or settings.CV_MAX_DOCX_MEMBERS

    if not isinstance(content, bytes) or not content:
        raise CVValidationError("The uploaded CV is empty.", status_code=422)
    if len(content) > size_limit:
        raise CVValidationError(
            f"The uploaded CV exceeds the {size_limit}-byte size limit.",
            status_code=413,
        )

    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".pdf", ".docx"}:
        raise CVValidationError(
            "Unsupported CV file extension. Only .pdf and .docx are accepted.",
            status_code=415,
        )

    # PDF readers permit a short prefix before the header, so check the first
    # 1 KiB rather than trusting a .pdf suffix or multipart Content-Type.
    pdf_signature_present = b"%PDF-" in content[:1024]
    docx_signature_present = content.startswith(b"PK\x03\x04")

    if pdf_signature_present:
        detected_type: SupportedFileType = "pdf"
    elif docx_signature_present:
        detected_type = "docx"
    else:
        raise CVValidationError(
            "The file contents do not match a supported PDF or DOCX document.",
            status_code=415,
        )

    expected_suffix = f".{detected_type}"
    if suffix != expected_suffix:
        raise CVValidationError(
            f"The file extension does not match its detected {detected_type.upper()} content.",
            status_code=415,
        )

    if detected_type == "pdf":
        _reject_encrypted_pdf(content)
    else:
        _validate_docx_container(
            content,
            max_uncompressed_bytes=uncompressed_limit,
            max_members=member_limit,
        )

    digest = hashlib.sha256(content).hexdigest()
    return ValidatedCV(
        content=content,
        filename=Path(filename).name,
        file_type=detected_type,
        cv_hash=digest,
    )


def _reject_encrypted_pdf(content: bytes) -> None:
    try:
        import fitz

        with fitz.open(stream=content, filetype="pdf") as document:
            if document.is_encrypted or document.needs_pass:
                raise CVValidationError(
                    "Password-protected or encrypted PDF files are not accepted.",
                    status_code=422,
                )
    except CVValidationError:
        raise
    except Exception as exc:
        raise CVValidationError(
            "The uploaded PDF is malformed or cannot be opened.", status_code=422
        ) from exc


def _validate_docx_container(
    content: bytes,
    *,
    max_uncompressed_bytes: int,
    max_members: int,
) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > max_members:
                raise CVValidationError(
                    "The DOCX archive contains too many internal files.", status_code=422
                )

            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed > max_uncompressed_bytes:
                raise CVValidationError(
                    "The DOCX expands beyond the permitted uncompressed size.",
                    status_code=413,
                )

            names = set(archive.namelist())
            required = {"[Content_Types].xml", "word/document.xml"}
            if not required.issubset(names):
                raise CVValidationError(
                    "The uploaded ZIP file is not a valid DOCX document.",
                    status_code=415,
                )
            # Validate the actual OPC main-part type as well as ZIP magic and
            # required members; a renamed arbitrary ZIP is not a DOCX.
            content_types = archive.read("[Content_Types].xml")
            expected_main_type = (
                b"application/vnd.openxmlformats-officedocument."
                b"wordprocessingml.document.main+xml"
            )
            if expected_main_type not in content_types:
                raise CVValidationError(
                    "The uploaded Office package is not a standard DOCX document.",
                    status_code=415,
                )
            if "word/vbaProject.bin" in names:
                raise CVValidationError(
                    "Macro-enabled Word documents are not accepted.", status_code=415
                )
            # Read the core document part now so bad CRCs/truncated archives are
            # rejected before python-docx attempts to interpret the package.
            archive.read("word/document.xml")
    except CVValidationError:
        raise
    except Exception as exc:
        raise CVValidationError(
            "The uploaded DOCX archive is malformed or cannot be opened.",
            status_code=422,
        ) from exc


def parse_cv(
    content: bytes,
    filename: str,
    *,
    max_file_size_bytes: int | None = None,
) -> ParsedDocument:
    """Validate, hash, and convert an uploaded CV to Markdown."""

    validated = validate_and_hash_cv(
        content,
        filename,
        max_file_size_bytes=max_file_size_bytes,
    )
    try:
        if validated.file_type == "pdf":
            markdown = _pdf_to_markdown(validated.content)
        else:
            markdown = _docx_to_markdown(validated.content)
        markdown = _normalize_markdown(markdown)
    except CVValidationError:
        raise
    except Exception as exc:
        logger.exception("CV conversion failed for hash %s", validated.cv_hash)
        raise CVParsingError(
            f"The validated {validated.file_type.upper()} file could not be converted.",
            cv_hash=validated.cv_hash,
        ) from exc

    return ParsedDocument(
        cv_hash=validated.cv_hash,
        file_type=validated.file_type,
        markdown=markdown,
    )


def _supported_pymupdf4llm_kwargs(to_markdown_fn: object) -> dict[str, object]:
    """Build keyword arguments supported by the installed pymupdf4llm version.

    OCR is disabled when supported so short/scanned PDFs surface ``needs_ocr``
    explicitly instead of silently triggering implicit OCR or failing with a
    TypeError on versions without the ``use_ocr`` parameter.
    """

    desired: dict[str, object] = {"show_progress": False, "use_ocr": False}
    try:
        signature = inspect.signature(to_markdown_fn)
    except (TypeError, ValueError):
        return {"show_progress": False}

    parameters = signature.parameters
    accepts_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    if accepts_var_kw:
        return desired
    return {key: value for key, value in desired.items() if key in parameters}


def _pdf_to_markdown(content: bytes) -> str:
    """Use PyMuPDF4LLM's layout-aware Markdown conversion."""

    try:
        import fitz
        import pymupdf4llm
    except ImportError as exc:
        raise RuntimeError(
            "PDF conversion dependencies are unavailable; install pymupdf4llm."
        ) from exc

    with fitz.open(stream=content, filetype="pdf") as document:
        if document.is_encrypted or document.needs_pass:
            # Recheck at conversion time in case this function is called alone.
            raise CVValidationError(
                "Password-protected or encrypted PDF files are not accepted.",
                status_code=422,
            )
        kwargs = _supported_pymupdf4llm_kwargs(pymupdf4llm.to_markdown)
        markdown = pymupdf4llm.to_markdown(document, **kwargs)
    return str(markdown or "")


def _iter_docx_blocks(parent: DocumentObject | _Cell):
    """Yield Word paragraphs and tables in their original document order."""

    parent_element = parent.element.body if isinstance(parent, DocumentObject) else parent._tc
    for child in parent_element.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def _docx_to_markdown(content: bytes) -> str:
    """Convert DOCX paragraphs/tables while retaining heading hierarchy."""

    document = Document(io.BytesIO(content))
    output: list[str] = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            if not text:
                continue

            style_name = getattr(block.style, "name", "") or ""
            heading_match = re.match(r"^Heading\s*(\d+)$", style_name, re.IGNORECASE)
            if heading_match:
                # Word Heading 1 maps to Markdown H2; this gives the CV a
                # predictable section-heading level without a document title H1.
                level = min(6, max(2, int(heading_match.group(1)) + 1))
                output.append(f"{'#' * level} {text}")
            elif style_name.lower().startswith(("list bullet", "list number")):
                output.append(f"- {text}")
            elif style_name.lower() in {"title", "subtitle"}:
                output.append(f"## {text}")
            else:
                output.append(text)
        elif isinstance(block, Table):
            for row in block.rows:
                # Horizontally merged Word cells share the same underlying <w:tc>
                # element; deduplicate by XML node identity to avoid repeating text.
                cells: list[str] = []
                seen_tc: set[object] = set()
                for cell in row.cells:
                    if cell._tc in seen_tc:
                        continue
                    seen_tc.add(cell._tc)
                    cell_text = re.sub(r"\s+", " ", cell.text).strip()
                    if cell_text:
                        cells.append(cell_text)
                if cells:
                    output.append(" | ".join(cells))

    return "\n\n".join(output)


def _normalize_markdown(markdown: str) -> str:
    # Normalize line endings and remove control characters while preserving
    # Markdown structure and paragraph boundaries.
    text = (markdown or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# PII patterns are intentionally applied to the entire converted document
# before it is truncated or passed to any extraction provider.
_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
_URL_RE = re.compile(
    r"(?i)\b(?:https?://|ftp://|www\.)[^\s<>()\[\]{}]+"
)

# Well-known technology names that look like bare domains without a URL path.
# If followed by a path (e.g. ``socket.io/users/alice``) or scheme/email, they
# are still redacted as URLs/emails.
_TECH_DOMAIN_ALLOWLIST = (
    r"(?:asp|vb|ml|paint)\.net"
    r"|(?:socket|draw|swagger|redis|grpc|prisma|typeorm|angular|kubernetes"
    r"|leaflet|expo|capacitor|ionic|sentry|storybook|cypress|playwright|fly)\.io"
    r"|(?:flutter|dart|react|vite|vitejs|pydantic|web|go|deno|bun|astro|nuxt|svelte)\.dev"
)
_BARE_DOMAIN_RE = re.compile(
    rf"(?i)\b(?!(?:{_TECH_DOMAIN_ALLOWLIST})\b(?!/))"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"(?:com\.sy|com|org|net|edu|gov|mil|int|io|co|me|info|biz|dev|app|uk|de|nl|sy)"
    r"(?:/[^\s<>()\[\]{}]*)?"
)
_NATIONAL_ID_RE = re.compile(r"(?<!\d)\d{11}(?!\d)")

# Three-group phone formats: +963 944 123 456, 00963 944 123 456, (011) 234 5678
_PHONE_3GROUP_RE = re.compile(
    r"(?<![\w])(?:(?:\+|00)\s?\d{1,3}[\s().-]?)?"
    r"(?:\(?\d{2,4}\)?[\s().-]?)\d{3,4}[\s.-]\d{3,4}(?!\w)"
)
# Two-group local/mobile/landline formats: 0944 123456, 0944-123456,
# (011) 2345678, +963 944 123456. Requiring 6-9 trailing digits avoids matching
# year ranges like 2020-2024 or ISO dates like 2024-05-15.
_PHONE_2GROUP_RE = re.compile(
    r"(?<![\w])(?:(?:\+|00)\s?\d{1,3}[\s().-]+)?"
    r"\(?\d{2,4}\)?[\s.-]\d{6,9}(?!\w)"
)
# International country code followed by a single subscriber block: +963 944123456
_PHONE_INTL_2GROUP_RE = re.compile(
    r"(?<![\w])(?:\+|00)\s?\d{1,3}[\s().-]+\d{6,11}(?!\w)"
)
_CONTIGUOUS_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+|00)?\d{7,10}(?!\d)|(?<!\d)(?:\+|00)?\d{12,15}(?!\d)"
)


def count_non_pii_text_characters(scrubbed_markdown: str) -> int:
    """Count meaningful extracted characters without counting redaction tags."""

    without_markers = re.sub(r"\[REDACTED_[A-Z_]+\]", "", scrubbed_markdown or "")
    return len(re.sub(r"\s+", "", without_markers))


def scrub_pii(markdown: str) -> str:
    """Redact emails, URLs, phone numbers, and 11-digit identifiers.

    Replacement markers preserve document structure while ensuring the
    underlying identifiers are not sent to an LLM provider.
    """

    text = markdown or ""
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    text = _URL_RE.sub("[REDACTED_URL]", text)
    text = _BARE_DOMAIN_RE.sub("[REDACTED_URL]", text)
    text = _NATIONAL_ID_RE.sub("[REDACTED_ID]", text)
    text = _PHONE_3GROUP_RE.sub("[REDACTED_PHONE]", text)
    text = _PHONE_2GROUP_RE.sub("[REDACTED_PHONE]", text)
    text = _PHONE_INTL_2GROUP_RE.sub("[REDACTED_PHONE]", text)
    text = _CONTIGUOUS_PHONE_RE.sub("[REDACTED_PHONE]", text)
    return text