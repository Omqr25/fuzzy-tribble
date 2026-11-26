import hashlib
import io
import json

import pytest
from docx import Document
from fastapi.testclient import TestClient

from app.core.config import settings
from app.api.v1 import candidates as candidate_routes
from app.main import app
from app.schemas.candidate import ParsedCV
from app.services.parser import CVParsingError

AUTH_HEADERS = {"X-API-Key": settings.INTERNAL_API_KEY}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
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


def test_health_and_ready_probes(client):
    health_res = client.get("/health")
    assert health_res.status_code == 200
    assert "X-Request-ID" in health_res.headers

    assert client.get("/ready").status_code == 503
    app.state.models_loaded = True
    assert client.get("/ready").status_code == 200
    app.state.models_loaded = False


def test_protected_routes_reject_missing_or_wrong_api_key(client):
    res_missing = client.post("/api/v1/positions/embed", json={})
    res_wrong = client.post(
        "/api/v1/positions/embed",
        json={},
        headers={"X-API-Key": "wrong-key"},
    )
    assert res_missing.status_code == 401
    assert res_wrong.status_code == 401


def test_position_embed_contract_validation_remains_phase_4_stub(client):
    bad = client.post("/api/v1/positions/embed", json={}, headers=AUTH_HEADERS)
    assert bad.status_code == 422

    good = client.post(
        "/api/v1/positions/embed",
        json={
            "position_id": "pos-1",
            "title": "Backend Intern",
            "required_skills": ["PHP", "Laravel", "MySQL"],
        },
        headers=AUTH_HEADERS,
    )
    assert good.status_code == 501


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
    assert payload["vectors"] is None  # Faceted embeddings are Phase 4.
    assert payload["model_version"] == "not_generated_phase_4"




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
            "Experienced developer with Python and API skills. Contact: "
            "private.person@example.com, https://portfolio.example.com/me, "
            "+963 944 123 456, national id 12345678901. "
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
    for secret in (
        "private.person@example.com",
        "https://portfolio.example.com/me",
        "+963 944 123 456",
        "12345678901",
    ):
        assert secret not in provider.received_markdown

def test_candidate_ingest_short_document_returns_needs_ocr(client):
    content = make_docx(text="Too short")
    res = client.post(
        "/api/v1/candidates/ingest",
        headers=AUTH_HEADERS,
        files={"cv_file": ("cv.docx", content, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"metadata": json.dumps({"candidate_id": "cand-short"})},
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    assert payload["extraction_status"] == "needs_ocr"
    assert payload["extraction_method"] == "none"
    assert payload["parsed_cv"] is None
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
    assert client.post("/api/v1/allocations/solve", json=bad_seats, headers=AUTH_HEADERS).status_code == 422

    mismatched_id = {
        "candidates": [_sample_candidate(candidate_id="c1", meta_id="c2")],
        "positions": [_sample_position()],
    }
    assert client.post("/api/v1/allocations/solve", json=mismatched_id, headers=AUTH_HEADERS).status_code == 422


def test_allocation_explain_contract(client):
    bad = client.post("/api/v1/allocations/explain", json={"items": []}, headers=AUTH_HEADERS)
    assert bad.status_code == 422

    good = client.post(
        "/api/v1/allocations/explain",
        json={"items": [{"candidate": _sample_candidate()}]},
        headers=AUTH_HEADERS,
    )
    assert good.status_code == 501
