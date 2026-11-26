import asyncio
import json

import httpx
import pytest

from app.core.config import settings
from app.schemas.candidate import ParsedCV
from app.services.extractor import (
    OpenAICompatibleExtractionProvider,
    extract_structured_cv,
    extract_with_heading_fallback,
)


MARKDOWN = """## Skills
- Python
- FastAPI, PostgreSQL

## Experience
Backend developer who built APIs and automated integration tests.

## Education
BSc Computer Science, 2022

## Projects
- Candidate matching service

## Certifications
- AWS Cloud Practitioner

## Languages
- Arabic
- English
"""


class FakeProvider:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.received = None

    async def extract(self, scrubbed_markdown: str) -> ParsedCV:
        self.received = scrubbed_markdown
        if self.error:
            raise self.error
        return ParsedCV.model_validate(self.result)


def test_heading_fallback_extracts_all_contract_fields():
    parsed = extract_with_heading_fallback(MARKDOWN)
    assert parsed.skills == ["Python", "FastAPI", "PostgreSQL"]
    assert "Backend developer" in parsed.experience_summary
    assert "BSc Computer Science" in parsed.education_summary
    assert "Candidate matching service" in parsed.projects_summary
    assert parsed.certifications == ["AWS Cloud Practitioner"]
    assert parsed.languages == ["Arabic", "English"]
    assert 0.0 <= parsed.cv_completeness_score <= 1.0


def test_heading_fallback_extracts_bold_and_italic_markdown_headings():
    bold_markdown = """**Skills**
- **Python**, **FastAPI**
- PostgreSQL

**Experience:**
Built production microservices and automated CI/CD pipelines.

__Education__
BSc Software Engineering

*Projects*
- Distributed allocation service

**Certifications**
- CKA

**Languages**
- Arabic, English
"""
    parsed = extract_with_heading_fallback(bold_markdown)
    assert parsed.skills == ["Python", "FastAPI", "PostgreSQL"]
    assert "production microservices" in parsed.experience_summary
    assert "BSc Software Engineering" in parsed.education_summary
    assert "Distributed allocation service" in parsed.projects_summary
    assert parsed.certifications == ["CKA"]
    assert parsed.languages == ["Arabic", "English"]
    assert parsed.cv_completeness_score == 1.0


def test_heading_fallback_handles_title_case_with_conjunctions():
    plain_markdown = """Education and Training
BSc Information Technology, Damascus University

Licenses and Certifications
AWS Solutions Architect
"""
    parsed = extract_with_heading_fallback(plain_markdown)
    assert "Damascus University" in parsed.education_summary
    assert parsed.certifications == ["AWS Solutions Architect"]


def test_heading_fallback_extracts_inline_headings():
    inline_markdown = """**Skills:** Python, Socket.io, ML.NET
Languages: Arabic, English
المشاريع: نظام مطابقة وتوزيع المتدربين
"""
    parsed = extract_with_heading_fallback(inline_markdown)
    assert parsed.skills == ["Python", "Socket.io", "ML.NET"]
    assert parsed.languages == ["Arabic", "English"]
    assert "نظام مطابقة" in parsed.projects_summary


def test_heading_fallback_extracts_arabic_cv_sections():
    arabic_markdown = """المهارات التقنية
بايثون، فاست أي بي آي، قواعد بيانات

**الخبرة العملية:**
تطوير واجهات برمجية خلفية واختبار الأنظمة الموزعة.

التعليم
بكالوريوس هندسة معلوماتية

أبرز المشاريع
نظام مطابقة وتوزيع المتدربين

الشهادات والدورات
شهادة هندسة السحابة

اللغات
العربية، الإنجليزية
"""
    parsed = extract_with_heading_fallback(arabic_markdown)
    assert parsed.skills == ["بايثون", "فاست أي بي آي", "قواعد بيانات"]
    assert "واجهات برمجية" in parsed.experience_summary
    assert "هندسة معلوماتية" in parsed.education_summary
    assert "نظام مطابقة" in parsed.projects_summary
    assert parsed.certifications == ["شهادة هندسة السحابة"]
    assert parsed.languages == ["العربية", "الإنجليزية"]
    assert parsed.cv_completeness_score == 1.0


@pytest.mark.anyio
async def test_llm_provider_result_is_validated_and_tagged_llm():
    provider = FakeProvider(
        result={
            "skills": ["Python"],
            "experience_summary": "Backend API development",
            "education_summary": "BSc Computer Science",
            "projects_summary": "Matching engine",
            "certifications": [],
            "languages": ["Arabic"],
            "cv_completeness_score": 0.8,
        }
    )
    result = await extract_structured_cv(MARKDOWN, provider=provider)
    assert result.extraction_method == "llm"
    assert result.parsed_cv.skills == ["Python"]
    assert provider.received == MARKDOWN.strip()


@pytest.mark.anyio
async def test_provider_failure_uses_deterministic_fallback():
    provider = FakeProvider(error=RuntimeError("provider unavailable"))
    result = await extract_structured_cv(MARKDOWN, provider=provider)
    assert result.extraction_method == "fallback_heuristic"
    assert result.parsed_cv.skills[0] == "Python"
    assert "fallback was used" in result.extraction_error


@pytest.mark.anyio
async def test_provider_timeout_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(settings, "EXTRACTION_LLM_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(settings, "EXTRACTION_LLM_MAX_RETRIES", 0)

    class SlowProvider:
        async def extract(self, scrubbed_markdown: str) -> ParsedCV:
            await asyncio.sleep(0.1)
            return ParsedCV()

    result = await extract_structured_cv(MARKDOWN, provider=SlowProvider())
    assert result.extraction_method == "fallback_heuristic"
    assert result.parsed_cv.skills[0] == "Python"
    assert "timed out" in result.extraction_error


@pytest.mark.anyio
async def test_openai_compatible_provider_retries_rate_limit_then_parses_schema(monkeypatch):
    calls = {"count": 0, "clients_created": 0}
    parsed_json = json.dumps(
        {
            "skills": ["Python"],
            "experience_summary": "",
            "education_summary": "",
            "projects_summary": "",
            "certifications": [],
            "languages": [],
            "cv_completeness_score": 0.2,
        }
    )

    real_async_client = httpx.AsyncClient

    class CountingAsyncClient(real_async_client):
        def __init__(self, *args, **kwargs):
            calls["clients_created"] += 1
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", CountingAsyncClient)

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        request_body = json.loads(request.content)
        strict_schema = request_body["response_format"]["json_schema"]["schema"]
        assert strict_schema["additionalProperties"] is False
        assert set(strict_schema["required"]) == set(strict_schema["properties"])
        for prop in strict_schema["properties"].values():
            assert "default" not in prop
            assert "minimum" not in prop
            assert "maximum" not in prop
        if calls["count"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": parsed_json}}]},
        )

    provider = OpenAICompatibleExtractionProvider(
        base_url="https://llm.example/v1",
        model="fast-test-model",
        max_retries=1,
        retry_backoff_seconds=0,
        transport=httpx.MockTransport(handler),
    )
    result = await provider.extract(MARKDOWN)
    assert result.skills == ["Python"]
    assert calls["count"] == 2
    assert calls["clients_created"] == 1