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
async def test_openai_compatible_provider_retries_rate_limit_then_parses_schema():
    calls = {"count": 0}
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

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        request_body = json.loads(request.content)
        strict_schema = request_body["response_format"]["json_schema"]["schema"]
        assert set(strict_schema["required"]) == set(strict_schema["properties"])
        assert all("default" not in value for value in strict_schema["properties"].values())
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
