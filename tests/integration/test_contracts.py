import hashlib
import io
import json

import pymupdf
import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.api.v1 import candidates as candidate_routes
from app.core.config import settings
from app.main import app
from app.schemas.candidate import ParsedCV
from app.schemas.position import FacetedVectors
from app.services.parser import CVParsingError

AUTH_HEADERS = {"X-API-Key": settings.INTERNAL_API_KEY}


class _FakeEmbeddingService:
    """Deterministic embedding service for integration tests."""

    _DIM = 4

    @property
    def model_version(self) -> str:
        return "fake-test-model"

    @property
    def dimension(self) -> int:
        return self._DIM

    def _vectors(self) -> FacetedVectors:
        return FacetedVectors(
            skills=[0.5] * self._DIM,
            academic=[0.5] * self._DIM,
            practical=[0.5] * self._DIM,
        )

    def embed_position(self, request):
        return self._vectors()

    def embed_candidate(self, metadata, parsed_cv):
        return self._vectors()


@pytest.fixture
def client(monkeypatch):
    # Prevent integration tests from making live LLM requests when local .env
    # defines EXTRACTION_LLM_* settings.
    monkeypatch.setattr(settings, "EXTRACTION_LLM_BASE_URL", None)
    monkeypatch.setattr(settings, "EXTRACTION_LLM_MODEL", None)
    monkeypatch.setattr(settings, "EXTRACTION_LLM_API_KEY", None)

    # Prevent slow model download attempts during lifespan.
    from app.services import embedder as _embedder_mod

    def _skip_load(model_name):
        raise RuntimeError("Skipped in test environment")

    monkeypatch.setattr(_embedder_mod.EmbeddingService, "load", _skip_load)

    with TestClient(app) as test_client:
        # Inject deterministic fake embedder for integration tests.
        app.state.embedding_service = _FakeEmbeddingService()
        app.state.models_loaded = True
        yield test_client


def make_docx(*, text: str | None = None) -> bytes:
    document = Document()
    if text is not None:
        document.add_paragraph(text)
    else:
        document.add_heading("Skills", level=1)
        document.add_paragraph("Python, FastAPI, PostgreSQL")
        document.add_heading("Experience", level=1)
        document.add_paragraph(
            "Built and maintained backend APIs, automated integration tests, and "
            "deployment pipelines. Collaborated with engineering teams on production "
            "services and documented reliable release processes."
        )
        document.add_heading("Education", level=1)
        document.add_paragraph("BSc Computer Science")
    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


def make_pdf_cv() -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    sections = [
        ("Skills", ["Python, FastAPI, Socket.io, ML.NET"]),
        (
            "Experience",
            [
                "Built backend APIs and maintained automated deployment pipelines across production services.",
                "Contact: 0944 123456 | person@example.com | 12345678901",
            ],
        ),
        ("Education", ["BSc Computer Science, 2020-2024"]),
    ]
    y = 72
    for heading, body_lines in sections:
        page.insert_text((72, y), heading, fontsize=15, fontname="hebo")
        y += 22
        for line in body_lines:
            page.insert_text((72, y), line, fontsize=11, fontname="helv")
            y += 18
        y += 18
    content = document.tobytes()
    document.close()
    return content


def _sample_candidate(candidate_id: str = "c1", meta_id: str | None = None) -> dict:
    return {
        "candidate_id": candidate_id,
        "metadata": {"candidate_id": meta_id or candidate_id},
        "parsed_cv": {},
        "vectors": {"skills": [1.0], "academic": [1.0], "practical": [1.0]},
    }


def _sample_position(position_id: str = "p1", min_seats: int = 0, max_seats: int = 2) -> dict:
    return {
        "position_id": position_id,
        "title": "AI Intern",
        "min_seats": min_seats,
        "max_seats": max_seats,
        "vectors": {"skills": [1.0], "academic": [1.0], "practical": [1.0]},
    }


# ---------------------------------------------------------------------------
# Health & readiness
# ---------------------------------------------------------------------------


def test_health_and_ready_probes(client):
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert "X-Request-ID" in health_res.headers

    # Fixture injects the embedding service and sets models_loaded = True.
    assert client.get("/ready").status_code == 200
    app.state.models_loaded = False
    assert client.get("/ready").status_code == 503
    app.state.models_loaded = True


def test_protected_routes_reject_missing_or_wrong_api_key(client):
    res_missing = client.post("/api/v1/positions/embed", json={})
    res_wrong = client.post(
        "/api/v1/positions/embed",
        json={},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res_missing.status_code == 401
    assert res_wrong.status_code == 401


# ---------------------------------------------------------------------------
# Position embedding (Phase 4)
# ---------------------------------------------------------------------------


def test_position_embed_returns_faceted_vectors(client):
    bad = client.post("/api/v1/positions/embed", json={}, headers=AUTH_HEADERS)
    assert bad.status_code == 422

    res = client.post(
        "/api/v1/positions/embed",
        json={
            "position_id": "pos-1",
            "title": "Backend Intern",
            "required_skills": ["PHP", "Laravel", "MySQL"],
            "target_majors": ["Computer Science"],
            "academic_level": "Bachelor",
            "responsibilities": "Build and maintain backend APIs",
            "experience_level": "Junior",
        },
        headers=AUTH_HEADERS,
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["position_id"] == "pos-1"
    assert payload["model_version"] == "fake-test-model"
    vectors = payload["vectors"]
    assert len(vectors["skills"]) == _FakeEmbeddingService._DIM
    assert len(vectors["academic"]) == _FakeEmbeddingService._DIM
    assert len(vectors["practical"]) == _FakeEmbeddingService._DIM


def test_position_embed_returns_503_without_embedding_service(client):
    saved = app.state.embedding_service
    app.state.embedding_service = None
    try:
        res = client.post(
            "/api/v1/positions/embed",
            json={"position_id": "pos-1", "title": "Intern"},
            headers=AUTH_HEADERS,
        )
        assert res.status_code == 503
    finally:
        app.state.embedding_service = saved


# ---------------------------------------------------------------------------
# Candidate ingestion (Phase 3 parsing + Phase 4 embeddings)
# ---------------------------------------------------------------------------


def test_candidate_ingest_multipart_contract(client):
    content = make_docx()
    meta = json.dumps(
        {
            "candidate_id": "cand-1",
            "city": "Damascus",
            "gpa": 85.5,
            "technical_skills": ["Python", "FastAPI"],
        }
    )
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", content, "application/octet-stream")},
        data={"metadata": meta},
    )

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["candidate_id"] == "cand-1"
    assert payload["cv_hash"] == hashlib.sha256(content).hexdigest()
    assert payload["extraction_status"] == "completed"
    assert payload["extraction_method"] == "fallback_heuristic"
    assert payload["parsed_cv"]["skills"] == ["Python", "FastAPI", "PostgreSQL"]
    assert payload["vectors"] is not None
    assert len(payload["vectors"]["skills"]) == _FakeEmbeddingService._DIM
    assert len(payload["vectors"]["academic"]) == _FakeEmbeddingService._DIM
    assert len(payload["vectors"]["practical"]) == _FakeEmbeddingService._DIM
    assert payload["model_version"] == "fake-test-model"


def test_candidate_ingest_real_pdf_end_to_end(client):
    content = make_pdf_cv()
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.pdf", content, "application/pdf")},
        data={"metadata": json.dumps({"candidate_id": "cand-pdf"})},
    )

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["extraction_status"] == "completed"
    assert payload["extraction_method"] == "fallback_heuristic"
    assert payload["cv_hash"] == hashlib.sha256(content).hexdigest()
    assert "Socket.io" in payload["parsed_cv"]["skills"]
    assert "ML.NET" in payload["parsed_cv"]["skills"]
    assert "0944 123456" not in payload["parsed_cv"]["experience_summary"]
    assert "2020-2024" in payload["parsed_cv"]["education_summary"]
    assert payload["vectors"] is not None
    assert payload["model_version"] == "fake-test-model"


def test_candidate_ingest_arabic_docx_end_to_end(client):
    document = Document()
    document.add_paragraph("المهارات التقنية")
    document.add_paragraph("بايثون، فاست أي بي آي، دوكر")
    document.add_paragraph("الخبرة العملية")
    document.add_paragraph(
        "تطوير واجهات برمجية خلفية واختبار الأنظمة الموزعة والعمل مع فرق هندسية لبناء خدمات مستقرة."
    )
    document.add_paragraph("التعليم")
    document.add_paragraph("بكالوريوس هندسة معلوماتية من جامعة دمشق")
    stream = io.BytesIO()
    document.save(stream)

    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("arabic_cv.docx", stream.getvalue(), "application/octet-stream")},
        data={"metadata": json.dumps({"candidate_id": "cand-ar"})},
    )

    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["extraction_status"] == "completed"
    assert payload["extraction_method"] == "fallback_heuristic"
    assert payload["parsed_cv"]["skills"] == ["بايثون", "فاست أي بي آي", "دوكر"]
    assert "جامعة دمشق" in payload["parsed_cv"]["education_summary"]
    assert payload["parsed_cv"]["cv_completeness_score"] > 0.0
    assert payload["vectors"] is not None
    assert payload["model_version"] == "fake-test-model"


def test_candidate_pii_is_scrubbed_before_provider_call(client, monkeypatch):
    class CapturingProvider:
        received_markdown = None

        async def extract(self, scrubbed_markdown: str) -> ParsedCV:
            self.received_markdown = scrubbed_markdown
            return ParsedCV(skills=["Python"], cv_completeness_score=0.2)

    provider = CapturingProvider()
    monkeypatch.setattr(app.state, "cv_extraction_provider", provider, raising=False)
    content = make_docx(
        text=(
            "Experienced developer with Python, Socket.io, and API skills. Contact: "
            "private.person@example.com, https://portfolio.example.com/me, "
            "+963 944 123 456, 0944 123456, national id 12345678901. "
            "Built services, wrote automated tests, and supported deployments."
        )
    )
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", content, "application/octet-stream")},
        data={"metadata": json.dumps({"candidate_id": "cand-pii"})},
    )

    assert res.status_code == 200, res.text
    assert res.json()["extraction_method"] == "llm"
    assert provider.received_markdown is not None
    assert "Socket.io" in provider.received_markdown
    for secret in (
        "private.person@example.com",
        "https://portfolio.example.com/me",
        "+963 944 123 456",
        "0944 123456",
        "12345678901",
    ):
        assert secret not in provider.received_markdown


def test_candidate_ingest_short_document_returns_needs_ocr(client):
    content = make_docx(text="Too short")
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={
            "cv_file": (
                "cv.docx",
                content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
        data={"metadata": json.dumps({"candidate_id": "cand-short"})},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["extraction_status"] == "needs_ocr"
    assert payload["extraction_method"] == "none"
    assert payload["parsed_cv"] is None
    assert payload["vectors"] is None
    assert "OCR" in payload["extraction_error"]


def test_candidate_ingest_surfaces_document_conversion_failure(client, monkeypatch):
    def fail_parse(*args, **kwargs):
        raise CVParsingError(
            "The validated DOCX file could not be converted.",
            cv_hash="a" * 64,
        )

    monkeypatch.setattr(candidate_routes, "parse_cv", fail_parse)
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", make_docx(), "application/octet-stream")},
        data={"metadata": json.dumps({"candidate_id": "cand-broken"})},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["extraction_status"] == "failed"
    assert payload["extraction_method"] == "none"
    assert payload["cv_hash"] == "a" * 64
    assert payload["vectors"] is None
    assert "could not be converted" in payload["extraction_error"]


def test_candidate_ingest_rejects_mime_extension_mismatch(client):
    content = make_docx()
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.pdf", content, "application/pdf")},
        data={"metadata": json.dumps({"candidate_id": "cand-1"})},
    )
    assert res.status_code == 415


def test_candidate_ingest_rejects_oversized_file(client, monkeypatch):
    monkeypatch.setattr(settings, "CV_MAX_FILE_SIZE_BYTES", 512)
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", b"x" * 513, "application/octet-stream")},
        data={"metadata": json.dumps({"candidate_id": "cand-large"})},
    )
    assert res.status_code == 413


def test_candidate_ingest_rejects_malformed_json_metadata(client):
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", make_docx(), "application/octet-stream")},
        data={"metadata": "{not-valid-json"},
    )
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Allocation stubs (Phase 5–6)
# ---------------------------------------------------------------------------


def test_allocation_solve_valid_payload_returns_501(client):
    payload = {"candidates": [_sample_candidate()], "positions": [_sample_position()]}
    res = client.post("/api/v1/allocations/solve", json=payload, headers=AUTH_HEADERS)
    assert res.status_code == 501


def test_allocation_weights_must_sum_to_one(client):
    payload = {
        "candidates": [_sample_candidate()],
        "positions": [_sample_position()],
        "weights": {
            "skills_semantic": 0.9,
            "academic_semantic": 0.9,
            "practical_semantic": 0.0,
            "required_skills_lexical": 0.0,
            "preferred_skills_lexical": 0.0,
        },
    }
    res = client.post("/api/v1/allocations/solve", json=payload, headers=AUTH_HEADERS)
    assert res.status_code == 422


def test_allocation_schema_invariants(client):
    bad_seats = {
        "candidates": [_sample_candidate()],
        "positions": [_sample_position(min_seats=5, max_seats=2)],
    }
    assert (
        client.post("/api/v1/allocations/solve", json=bad_seats, headers=AUTH_HEADERS).status_code
        == 422
    )

    mismatched_id = {
        "candidates": [_sample_candidate(candidate_id="c1", meta_id="c2")],
        "positions": [_sample_position()],
    }
    assert (
        client.post(
            "/api/v1/allocations/solve", json=mismatched_id, headers=AUTH_HEADERS
        ).status_code
        == 422
    )


def test_allocation_explain_contract(client):
    bad = client.post("/api/v1/allocations/explain", json={"items": []}, headers=AUTH_HEADERS)
    assert bad.status_code == 422

    good = client.post(
        "/api/v1/allocations/explain",
        json={"items": [{"candidate": _sample_candidate()}]},
        headers=AUTH_HEADERS,
    )
    assert good.status_code == 501
