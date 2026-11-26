import numpy as np
import pytest

from app.schemas.candidate import CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionEmbedRequest
from app.services.embedder import (
    EmbeddingService,
    compose_candidate_academic_text,
    compose_candidate_practical_text,
    compose_candidate_skills_text,
    compose_position_academic_text,
    compose_position_practical_text,
    compose_position_skills_text,
)


# ---------------------------------------------------------------------------
# Text composition
# ---------------------------------------------------------------------------


class TestCandidateTextComposition:
    def _meta(self, **overrides) -> CandidateMetadata:
        defaults = {"candidate_id": "c1"}
        return CandidateMetadata(**(defaults | overrides))

    def _cv(self, **overrides) -> ParsedCV:
        return ParsedCV(**overrides)

    def test_skills_combines_db_and_cv_skills(self):
        meta = self._meta(technical_skills=["Python", "SQL"])
        cv = self._cv(skills=["FastAPI", "Docker"])
        assert compose_candidate_skills_text(meta, cv) == "Python, SQL, FastAPI, Docker"

    def test_skills_empty_when_no_skills(self):
        assert compose_candidate_skills_text(self._meta(), self._cv()) == ""

    def test_academic_combines_all_fields(self):
        meta = self._meta(
            program="BSc",
            field="Engineering",
            major="CS",
            year_of_study=3,
        )
        cv = self._cv(education_summary="Damascus University")
        result = compose_candidate_academic_text(meta, cv)
        assert "BSc" in result
        assert "Engineering" in result
        assert "CS" in result
        assert "year 3" in result
        assert "Damascus University" in result

    def test_academic_empty_when_no_fields(self):
        assert compose_candidate_academic_text(self._meta(), self._cv()) == ""

    def test_practical_combines_experience_projects_additional(self):
        meta = self._meta(additional_info="Available immediately")
        cv = self._cv(
            experience_summary="Built APIs",
            projects_summary="Matching engine",
        )
        result = compose_candidate_practical_text(meta, cv)
        assert "Built APIs" in result
        assert "Matching engine" in result
        assert "Available immediately" in result

    def test_practical_empty_when_no_fields(self):
        assert compose_candidate_practical_text(self._meta(), self._cv()) == ""


class TestPositionTextComposition:
    def _req(self, **overrides) -> PositionEmbedRequest:
        defaults = {"position_id": "p1", "title": "Intern"}
        return PositionEmbedRequest(**(defaults | overrides))

    def test_skills_combines_required_and_preferred(self):
        req = self._req(
            required_skills=["Python", "SQL"],
            preferred_skills=["Docker"],
        )
        assert compose_position_skills_text(req) == "Python, SQL, Docker"

    def test_skills_empty_when_no_skills(self):
        assert compose_position_skills_text(self._req()) == ""

    def test_academic_combines_majors_and_level(self):
        req = self._req(
            target_majors=["CS", "IT"],
            academic_level="Bachelor",
        )
        result = compose_position_academic_text(req)
        assert "CS, IT" in result
        assert "Bachelor" in result

    def test_practical_combines_responsibilities_and_level(self):
        req = self._req(
            responsibilities="Build backend APIs",
            experience_level="Junior",
        )
        result = compose_position_practical_text(req)
        assert "Build backend APIs" in result
        assert "Junior" in result


# ---------------------------------------------------------------------------
# EmbeddingService with a fake model
# ---------------------------------------------------------------------------


class _FakeModel:
    """Mimics the ``sentence_transformers.SentenceTransformer`` interface."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        sentences,
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        seed = hash(tuple(sentences)) % 2**32
        rng = np.random.default_rng(seed)
        vectors = rng.standard_normal((len(sentences), self._dimension))
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1, norms)
        return vectors


class TestEmbeddingService:
    @pytest.fixture()
    def service(self) -> EmbeddingService:
        return EmbeddingService(_FakeModel(dimension=8), model_name="test-model")

    def test_model_version_and_dimension(self, service):
        assert service.model_version == "test-model"
        assert service.dimension == 8

    def test_encode_returns_normalised_vectors(self, service):
        vectors = service.encode(["hello", "world"])
        assert len(vectors) == 2
        for vec in vectors:
            assert len(vec) == 8
            norm = sum(x**2 for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-6

    def test_encode_handles_empty_and_whitespace_text(self, service):
        vectors = service.encode(["", "   ", "valid text"])
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == 8

    def test_embed_position_returns_faceted_vectors(self, service):
        req = PositionEmbedRequest(
            position_id="p1",
            title="Backend Intern",
            required_skills=["Python"],
        )
        result = service.embed_position(req)
        assert isinstance(result, FacetedVectors)
        assert len(result.skills) == 8
        assert len(result.academic) == 8
        assert len(result.practical) == 8

    def test_embed_candidate_returns_faceted_vectors(self, service):
        meta = CandidateMetadata(candidate_id="c1", technical_skills=["Python"])
        cv = ParsedCV(skills=["FastAPI"], experience_summary="Built APIs")
        result = service.embed_candidate(meta, cv)
        assert isinstance(result, FacetedVectors)
        assert len(result.skills) == 8
        assert len(result.academic) == 8
        assert len(result.practical) == 8

    def test_embed_candidate_with_all_empty_fields(self, service):
        meta = CandidateMetadata(candidate_id="c1")
        cv = ParsedCV()
        result = service.embed_candidate(meta, cv)
        assert isinstance(result, FacetedVectors)
        # Every facet should still produce a valid normalised vector.
        for vec in (result.skills, result.academic, result.practical):
            assert len(vec) == 8
            norm = sum(x**2 for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-6

    def test_different_inputs_produce_different_vectors(self, service):
        req_a = PositionEmbedRequest(
            position_id="p1",
            title="Backend Intern",
            required_skills=["Python", "Django"],
        )
        req_b = PositionEmbedRequest(
            position_id="p2",
            title="Frontend Intern",
            required_skills=["JavaScript", "React"],
        )
        vec_a = service.embed_position(req_a)
        vec_b = service.embed_position(req_b)
        # At least the skills facet should differ for different skill sets.
        assert vec_a.skills != vec_b.skills
