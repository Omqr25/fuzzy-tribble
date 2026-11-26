from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pytest

from app.schemas.candidate import CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionEmbedRequest
from app.services.embedder import (
    EmbeddingModel,
    build_candidate_vectors,
    build_position_vectors,
    compose_candidate_academic_text,
    compose_candidate_practical_text,
    compose_candidate_skills_text,
    compose_position_academic_text,
    compose_position_practical_text,
    compose_position_skills_text,
)
from tests.conftest import FakeSentenceTransformer


class TestCandidateTextComposition:
    def _meta(self, **overrides) -> CandidateMetadata:
        defaults = {"candidate_id": "c1"}
        return CandidateMetadata(**(defaults | overrides))

    def _cv(self, **overrides) -> ParsedCV:
        return ParsedCV(**overrides)

    def test_skills_combines_strips_and_deduplicates_case_insensitively(self):
        meta = self._meta(technical_skills=["  Python ", "SQL", "   "])
        cv = self._cv(skills=["python", "FastAPI", "sql", "Docker "])
        assert (
            compose_candidate_skills_text(meta, cv)
            == "Python, SQL, FastAPI, Docker"
        )

    def test_skills_empty_when_no_or_blank_skills(self):
        meta = self._meta(technical_skills=["  ", ""])
        cv = self._cv(skills=["\t"])
        assert compose_candidate_skills_text(meta, cv) == ""

    def test_academic_combines_all_fields_and_ignores_whitespace(self):
        meta = self._meta(
            program=" BSc ",
            field="   ",
            major="CS",
            year_of_study=3,
        )
        cv = self._cv(education_summary=" Damascus University ")
        result = compose_candidate_academic_text(meta, cv)
        assert result == "BSc CS year of study 3 Damascus University"

    def test_academic_empty_when_whitespace_only(self):
        meta = self._meta(program="  ", field="\n", major="")
        cv = self._cv(education_summary="   ")
        assert compose_candidate_academic_text(meta, cv) == ""

    def test_practical_combines_experience_projects_additional(self):
        meta = self._meta(additional_info="Available immediately")
        cv = self._cv(
            experience_summary="Built APIs",
            projects_summary="Matching engine",
        )
        result = compose_candidate_practical_text(meta, cv)
        assert result == "Built APIs Matching engine Available immediately"

    def test_practical_empty_when_whitespace_only(self):
        meta = self._meta(additional_info="  ")
        cv = self._cv(experience_summary="", projects_summary="\t")
        assert compose_candidate_practical_text(meta, cv) == ""


class TestPositionTextComposition:
    def _req(self, **overrides) -> PositionEmbedRequest:
        defaults = {"position_id": "p1", "title": "Intern"}
        return PositionEmbedRequest(**(defaults | overrides))

    def test_skills_combines_strips_and_deduplicates(self):
        req = self._req(
            required_skills=[" Python", "SQL", " "],
            preferred_skills=["sql", "Docker"],
        )
        assert compose_position_skills_text(req) == "Python, SQL, Docker"

    def test_skills_empty_when_no_skills(self):
        assert compose_position_skills_text(self._req()) == ""

    def test_academic_combines_majors_and_level(self):
        req = self._req(
            target_majors=[" CS ", "cs", "IT"],
            academic_level=" Bachelor ",
        )
        assert compose_position_academic_text(req) == "CS, IT Bachelor"

    def test_practical_prefers_responsibilities_and_falls_back_to_description(self):
        with_resp = self._req(
            responsibilities="Build backend APIs",
            description="General team description",
            experience_level="Junior",
        )
        assert compose_position_practical_text(with_resp) == "Build backend APIs Junior"

        fallback_desc = self._req(
            responsibilities="   ",
            description="Develop microservices",
            experience_level="Junior",
        )
        assert compose_position_practical_text(fallback_desc) == "Develop microservices Junior"

        minimal = self._req()
        assert compose_position_practical_text(minimal) == ""


class TestEmbeddingModel:
    @pytest.fixture()
    def backend(self) -> FakeSentenceTransformer:
        return FakeSentenceTransformer(dimension=8)

    @pytest.fixture()
    def model(self, backend: FakeSentenceTransformer) -> EmbeddingModel:
        return EmbeddingModel("BAAI/bge-m3", _model=backend)

    def test_model_version_and_dimension(self, model: EmbeddingModel):
        assert model.model_version == "BAAI/bge-m3"
        assert model.dimension == 8

    def test_encode_passes_normalize_embeddings_true_and_returns_unit_vectors(
        self, model: EmbeddingModel, backend: FakeSentenceTransformer
    ):
        vectors = model.encode_texts(["hello", "world"])
        assert len(vectors) == 2
        assert len(backend.calls) == 1
        assert backend.calls[0]["normalize_embeddings"] is True
        assert backend.calls[0]["show_progress_bar"] is False
        for vec in vectors:
            assert len(vec) == 8
            norm = sum(x**2 for x in vec) ** 0.5
            assert abs(norm - 1.0) < 1e-6

    def test_encode_returns_empty_list_for_empty_input(
        self, model: EmbeddingModel, backend: FakeSentenceTransformer
    ):
        assert model.encode_texts([]) == []
        assert backend.calls == []

    def test_encode_maps_empty_and_whitespace_texts_to_zero_vectors(
        self, model: EmbeddingModel, backend: FakeSentenceTransformer
    ):
        vectors = model.encode_texts(["", "   ", "valid text"])
        assert len(vectors) == 3
        assert vectors[0] == [0.0] * 8
        assert vectors[1] == [0.0] * 8
        assert abs(sum(x**2 for x in vectors[2]) ** 0.5 - 1.0) < 1e-6
        assert backend.calls[0]["sentences"] == ["valid text"]

    def test_encode_skips_backend_when_all_texts_empty(
        self, model: EmbeddingModel, backend: FakeSentenceTransformer
    ):
        vectors = model.encode_texts(["", "  "])
        assert vectors == [[0.0] * 8, [0.0] * 8]
        assert backend.calls == []

    def test_empty_facets_produce_zero_dot_product_similarity(
        self, model: EmbeddingModel
    ):
        cand_vecs = build_candidate_vectors(
            CandidateMetadata(candidate_id="c1", technical_skills=["Python"]),
            ParsedCV(),
            model,
        )
        empty_pos_vecs = build_position_vectors(
            PositionEmbedRequest(position_id="p1", title="Intern"),
            model,
        )
        # Missing facets must evaluate to exact 0.0 in dot products.
        assert empty_pos_vecs.skills == [0.0] * 8
        assert empty_pos_vecs.academic == [0.0] * 8
        assert empty_pos_vecs.practical == [0.0] * 8
        assert cand_vecs.academic == [0.0] * 8
        assert cand_vecs.practical == [0.0] * 8
        assert float(np.dot(cand_vecs.skills, empty_pos_vecs.skills)) == 0.0
        assert float(np.dot(cand_vecs.academic, empty_pos_vecs.academic)) == 0.0

    def test_embed_position_and_candidate_return_faceted_vectors(
        self, model: EmbeddingModel
    ):
        pos = PositionEmbedRequest(
            position_id="p1",
            title="Backend Intern",
            required_skills=["Python"],
            target_majors=["Computer Science"],
            responsibilities="Build APIs",
        )
        pos_vecs = build_position_vectors(pos, model)
        assert isinstance(pos_vecs, FacetedVectors)
        for vec in (pos_vecs.skills, pos_vecs.academic, pos_vecs.practical):
            assert len(vec) == 8
            assert abs(sum(x**2 for x in vec) ** 0.5 - 1.0) < 1e-6

        cand_vecs = build_candidate_vectors(
            CandidateMetadata(
                candidate_id="c1",
                technical_skills=["Python"],
                major="Computer Science",
            ),
            ParsedCV(skills=["FastAPI"], experience_summary="Built APIs"),
            model,
        )
        assert isinstance(cand_vecs, FacetedVectors)
        for vec in (cand_vecs.skills, cand_vecs.academic, cand_vecs.practical):
            assert len(vec) == 8
            assert abs(sum(x**2 for x in vec) ** 0.5 - 1.0) < 1e-6

    def test_e5_model_applies_query_and_passage_prefixes_only_to_non_empty_facets(self):
        backend = FakeSentenceTransformer(dimension=8)
        e5_model = EmbeddingModel("intfloat/multilingual-e5-base", _model=backend)

        build_position_vectors(
            PositionEmbedRequest(
                position_id="p1",
                title="Backend",
                required_skills=["Python"],
            ),
            e5_model,
        )
        build_candidate_vectors(
            CandidateMetadata(candidate_id="c1", technical_skills=["Python"]),
            ParsedCV(experience_summary="Built APIs"),
            e5_model,
        )

        assert backend.calls[0]["sentences"] == ["query: Python"]
        assert backend.calls[1]["sentences"] == ["passage: Python", "passage: Built APIs"]

    def test_different_inputs_produce_different_vectors(self, model: EmbeddingModel):
        vec_a = build_position_vectors(
            PositionEmbedRequest(
                position_id="p1",
                title="Backend Intern",
                required_skills=["Python", "Django"],
            ),
            model,
        )
        vec_b = build_position_vectors(
            PositionEmbedRequest(
                position_id="p2",
                title="Frontend Intern",
                required_skills=["JavaScript", "React"],
            ),
            model,
        )
        assert vec_a.skills != vec_b.skills

    def test_encode_texts_serializes_concurrent_backend_calls(self):
        class OverlapTrackingBackend(FakeSentenceTransformer):
            def __init__(self) -> None:
                super().__init__(dimension=4)
                self._counter_lock = threading.Lock()
                self._active = 0
                self.max_concurrent = 0

            def encode(self, sentences, *, normalize_embeddings=False, show_progress_bar=False):
                with self._counter_lock:
                    self._active += 1
                    self.max_concurrent = max(self.max_concurrent, self._active)
                try:
                    time.sleep(0.01)
                    return super().encode(
                        sentences,
                        normalize_embeddings=normalize_embeddings,
                        show_progress_bar=show_progress_bar,
                    )
                finally:
                    with self._counter_lock:
                        self._active -= 1

        backend = OverlapTrackingBackend()
        model = EmbeddingModel("BAAI/bge-m3", _model=backend)

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(model.encode_texts, [[f"text {i}"] for i in range(6)]))

        assert len(results) == 6
        assert backend.max_concurrent == 1
