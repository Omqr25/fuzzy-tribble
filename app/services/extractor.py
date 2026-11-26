from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from app.core.config import settings
from app.schemas.candidate import ParsedCV

logger = logging.getLogger(__name__)


class StructuredExtractionProvider(Protocol):
    async def extract(self, scrubbed_markdown: str) -> ParsedCV:
        """Return a schema-validated ParsedCV from scrubbed Markdown."""


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    parsed_cv: ParsedCV
    extraction_method: str
    extraction_error: str | None = None


class ExtractionProviderError(RuntimeError):
    pass


class OpenAICompatibleExtractionProvider:
    """Adapter for OpenAI-compatible Chat Completions APIs with JSON Schema.

    ``base_url`` is the API root (for example ``https://host.example/v1``),
    not the ``/chat/completions`` path. The provider sends the exact Pydantic
    JSON schema using strict structured output mode.
    """

    RETRYABLE_STATUS_CODES = {408, 409, 425, 429}

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        self.transport = transport

    async def extract(self, scrubbed_markdown: str) -> ParsedCV:
        schema = _strict_parsed_cv_json_schema()
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You extract factual information from CVs. Return only the "
                        "requested ParsedCV object. Do not infer missing facts, "
                        "invent skills, or restore redacted personal information. "
                        "Use empty strings/lists for missing sections. Compute "
                        "cv_completeness_score as a float between 0.0 and 1.0 "
                        "from the evidence present in the CV."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Extract this PII-scrubbed CV into the required schema.\n\n"
                        f"CV MARKDOWN:\n{scrubbed_markdown}"
                    ),
                },
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "parsed_cv",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last_error: Exception | None = None

        # Reuse a single AsyncClient across retry attempts for connection pooling.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds),
            transport=self.transport,
        ) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(url, headers=headers, json=payload)

                    if response.status_code >= 400:
                        response.raise_for_status()

                    response_payload = response.json()
                    content = _extract_message_content(response_payload)
                    parsed = ParsedCV.model_validate_json(_strip_json_fence(content))
                    return parsed
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    status_code = exc.response.status_code
                    retryable = (
                        status_code in self.RETRYABLE_STATUS_CODES
                        or status_code >= 500
                    )
                    if not retryable or attempt >= self.max_retries:
                        break
                except (
                    httpx.TimeoutException,
                    httpx.TransportError,
                    ValueError,
                    KeyError,
                    TypeError,
                ) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        break

                await asyncio.sleep(self.retry_backoff_seconds * (2**attempt))

        raise ExtractionProviderError(
            f"Structured extraction provider failed after {self.max_retries + 1} attempt(s)."
        ) from last_error


_UNSUPPORTED_STRICT_SCHEMA_KEYS = {
    "default",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
}


def _strict_parsed_cv_json_schema() -> dict[str, Any]:
    """Return a JSON Schema compatible with strict structured-output APIs.

    Strict Structured Outputs requires every property to be listed in
    ``required``, ``additionalProperties`` set to ``false``, and rejects
    validation keywords such as ``default``, ``minimum``, and ``maximum``.
    Pydantic enforces all numeric bounds locally during ``model_validate_json``.
    """

    schema = ParsedCV.model_json_schema()
    properties = schema.get("properties", {})
    schema["required"] = list(properties)
    schema["additionalProperties"] = False
    for property_schema in properties.values():
        for unsupported_key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
            property_schema.pop(unsupported_key, None)
    return schema


def _extract_message_content(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response did not contain a chat message.") from exc

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if text_parts:
            return "\n".join(text_parts)
    raise ValueError("LLM response content was not a JSON string.")


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped


def get_configured_provider() -> StructuredExtractionProvider | None:
    """Build the optional provider from environment-backed settings."""

    if not settings.EXTRACTION_LLM_BASE_URL:
        return None
    if not settings.EXTRACTION_LLM_MODEL:
        logger.warning(
            "EXTRACTION_LLM_BASE_URL is set but EXTRACTION_LLM_MODEL is missing; using fallback"
        )
        return None

    return OpenAICompatibleExtractionProvider(
        base_url=settings.EXTRACTION_LLM_BASE_URL,
        model=settings.EXTRACTION_LLM_MODEL,
        api_key=settings.EXTRACTION_LLM_API_KEY,
        timeout_seconds=settings.EXTRACTION_LLM_TIMEOUT_SECONDS,
        max_retries=settings.EXTRACTION_LLM_MAX_RETRIES,
        retry_backoff_seconds=settings.EXTRACTION_LLM_RETRY_BACKOFF_SECONDS,
    )


async def extract_structured_cv(
    scrubbed_markdown: str,
    *,
    provider: StructuredExtractionProvider | None = None,
    max_markdown_chars: int | None = None,
) -> ExtractionResult:
    """Extract a ParsedCV, falling back deterministically on provider errors."""

    text = (scrubbed_markdown or "").strip()
    if not text:
        raise ValueError("Cannot extract a CV from empty Markdown.")

    limit = (
        settings.EXTRACTION_MAX_MARKDOWN_CHARS
        if max_markdown_chars is None
        else max_markdown_chars
    )
    if limit > 0 and len(text) > limit:
        text = text[:limit]

    selected_provider = provider if provider is not None else get_configured_provider()
    if selected_provider is None:
        return ExtractionResult(
            parsed_cv=extract_with_heading_fallback(text),
            extraction_method="fallback_heuristic",
        )

    try:
        retry_count = settings.EXTRACTION_LLM_MAX_RETRIES
        retry_wait = settings.EXTRACTION_LLM_RETRY_BACKOFF_SECONDS * (
            (2**retry_count) - 1
        )
        overall_timeout = (
            settings.EXTRACTION_LLM_TIMEOUT_SECONDS * (retry_count + 1) + retry_wait
        )
        parsed = await asyncio.wait_for(
            selected_provider.extract(text),
            timeout=overall_timeout,
        )
        # Enforce the public contract even for custom/injected providers.
        validated = ParsedCV.model_validate(parsed)
        return ExtractionResult(parsed_cv=validated, extraction_method="llm")
    except Exception as exc:  # Provider failures must never discard the parsed CV.
        logger.warning(
            "Structured CV extraction failed; using deterministic fallback (%s)",
            type(exc).__name__,
        )
        return ExtractionResult(
            parsed_cv=extract_with_heading_fallback(text),
            extraction_method="fallback_heuristic",
            extraction_error="LLM extraction failed or timed out; deterministic fallback was used.",
        )


_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<heading>.+?)\s*#*\s*$")
_EMPHASIS_WRAPPED_RE = re.compile(
    r"^(?P<marker>\*{1,3}|_{1,3})(?P<inner>.+?)(?P=marker)\s*:?\s*$"
)
_INLINE_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(.+?)\1")
_BULLET_LINE_RE = re.compile(r"^(?:[-*+•>]\s+|[-+•>]|\d+[.)]\s+)")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0640\u064b-\u065f\u0670]")
_ARABIC_ALEF_RE = re.compile(r"[أإآٱ]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
_TITLE_CONNECTORS = {"and", "of", "the", "in", "for", "&"}


def _normalize_arabic(text: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    return _ARABIC_ALEF_RE.sub("ا", text)


def _normalize_heading(heading: str) -> str:
    normalized = _normalize_arabic(heading)
    normalized = re.sub(r"[`*_]", "", normalized).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"\s*[:|].*$", "", normalized)
    return normalized.strip(" :-")


_RAW_SECTION_ALIASES: dict[str, set[str]] = {
    "skills": {
        "skill", "skills", "technical skills", "key skills", "core skills",
        "competencies", "technical competencies", "areas of expertise",
        "المهارات", "مهارات", "المهارات التقنية", "المهارات الفنية",
        "مهارات تقنية", "المهارات الأساسية", "الكفاءات", "الكفاءات التقنية",
    },
    "experience": {
        "experience", "work experience", "professional experience",
        "employment", "employment history", "work history", "internships",
        "professional background",
        "الخبرات", "الخبرة", "الخبرة العملية", "الخبرات العملية",
        "الخبرة المهنية", "الخبرات المهنية", "التاريخ الوظيفي",
        "التدريب", "التدريب العملي", "خبرات", "خبرات عملية",
    },
    "education": {
        "education", "educational background", "academic background",
        "academic qualifications", "qualifications", "education and training",
        "التعليم", "المؤهلات العلمية", "المؤهل العلمي", "التحصيل العلمي",
        "التحصيل الأكاديمي", "الخلفية الأكاديمية", "الدراسة",
        "الدراسة الأكاديمية", "الشهادات العلمية",
    },
    "projects": {
        "projects", "selected projects", "academic projects", "personal projects",
        "project experience",
        "المشاريع", "مشاريع", "المشاريع الأكاديمية", "المشاريع الشخصية",
        "أبرز المشاريع", "المشاريع المختارة", "مشاريع التخرج", "مشروع التخرج",
    },
    "certifications": {
        "certification", "certifications", "certificates", "licenses",
        "licenses and certifications",
        "الشهادات", "شهادات", "الشهادات المهنية", "الدورات",
        "الدورات التدريبية", "الشهادات والدورات", "الدورات والشهادات",
        "التراخيص والشهادات",
    },
    "languages": {
        "language", "languages", "language skills",
        "اللغات", "لغات", "المهارات اللغوية", "مهارات اللغة",
    },
}

_SECTION_ALIASES: dict[str, set[str]] = {
    section: {_normalize_heading(alias) for alias in aliases}
    for section, aliases in _RAW_SECTION_ALIASES.items()
}


def _classify_heading(heading: str) -> str | None:
    normalized = _normalize_heading(heading)
    for section, aliases in _SECTION_ALIASES.items():
        if normalized in aliases:
            return section
    return None


def _is_title_like(text: str) -> bool:
    """Accept ALL CAPS and Title Case headings including lowercase connectors."""

    if text.isupper() or text.istitle():
        return True
    words = text.split()
    if not words:
        return False
    for index, word in enumerate(words):
        if index > 0 and word.lower() in _TITLE_CONNECTORS:
            continue
        if not (word.isupper() or word[0].isupper()):
            return False
    return True


def _extract_plain_heading_section(line: str) -> str | None:
    """Identify standalone bold/italic or plain-text section headings."""

    stripped = line.strip()
    if not stripped or _BULLET_LINE_RE.match(stripped):
        return None

    is_emphasized = False
    emphasis_match = _EMPHASIS_WRAPPED_RE.match(stripped)
    if emphasis_match:
        candidate = emphasis_match.group("inner").strip().strip(":")
        is_emphasized = True
    else:
        candidate = stripped.strip(":")

    if not candidate or len(candidate) > 60 or len(candidate.split()) > 7:
        return None

    section = _classify_heading(candidate)
    if section is None:
        return None

    # Bold/italic standalone headings (common in pymupdf4llm output) and
    # non-Latin scripts such as Arabic do not require English title/upper case.
    if is_emphasized or not _LATIN_LETTER_RE.search(candidate):
        return section
    if _is_title_like(candidate):
        return section
    return None


def _split_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {key: [] for key in _SECTION_ALIASES}
    active: str | None = None

    for line in markdown.splitlines():
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            active = _classify_heading(heading_match.group("heading"))
            continue
        plain_section = _extract_plain_heading_section(line)
        if plain_section is not None:
            active = plain_section
            continue
        if active is not None:
            sections[active].append(line)

    return sections


def _clean_section(lines: list[str], *, max_chars: int = 6_000) -> str:
    text = "\n".join(lines)
    text = re.sub(r"\[REDACTED_[A-Z_]+\]", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"[*_~]{1,3}", "", text)
    text = re.sub(r"^\s*(?:[-*+•] |\d+[.)] )", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip(" \t\n-•")
    return text[:max_chars].strip()


def _extract_list_items(lines: list[str], *, max_items: int = 100) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for raw_line in lines:
        line = re.sub(r"\[REDACTED_[A-Z_]+\]", "", raw_line).strip()
        line = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s*", "", line)
        line = _INLINE_EMPHASIS_RE.sub(r"\2", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = line.strip(" \t,;|•،؛")
        if not line:
            continue
        # Skills/certifications/languages are often separated by English or
        # Arabic commas/semicolons or vertical bars; preserve plus/hash symbols.
        candidates = re.split(r"\s*(?:,|;|،|؛|\||•)\s*", line)
        for candidate in candidates:
            candidate = re.sub(r"\s+", " ", candidate).strip(" .:-")
            if not candidate or len(candidate) > 120:
                continue
            key = candidate.casefold()
            if key not in seen:
                seen.add(key)
                items.append(candidate)
            if len(items) >= max_items:
                return items
    return items


def _completeness_score(fields: dict[str, Any]) -> float:
    weights = {
        "skills": 0.20,
        "experience_summary": 0.25,
        "education_summary": 0.20,
        "projects_summary": 0.15,
        "certifications": 0.10,
        "languages": 0.10,
    }
    score = 0.0
    for field, weight in weights.items():
        value = fields[field]
        if value:
            score += weight
    return round(min(1.0, max(0.0, score)), 2)


def extract_with_heading_fallback(scrubbed_markdown: str) -> ParsedCV:
    """Extract the six ParsedCV fields from recognizable Markdown sections."""

    sections = _split_sections(scrubbed_markdown)
    fields: dict[str, Any] = {
        "skills": _extract_list_items(sections["skills"]),
        "experience_summary": _clean_section(sections["experience"]),
        "education_summary": _clean_section(sections["education"]),
        "projects_summary": _clean_section(sections["projects"]),
        "certifications": _extract_list_items(sections["certifications"]),
        "languages": _extract_list_items(sections["languages"]),
    }
    fields["cv_completeness_score"] = _completeness_score(fields)
    return ParsedCV.model_validate(fields)