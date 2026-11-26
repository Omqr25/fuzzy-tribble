from __future__ import annotations

import math
import unicodedata

import numpy as np
import pytest

from app.schemas.allocation import AlternativePlacement, ScoreBreakdown, ScoringWeights
from app.schemas.candidate import CandidateForAllocation, CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionForAllocation
from app.services import scorer as scorer_module
from app.services.scorer import (
    PrunedGraph,
    ScoringResult,
    SparseEdge,
    prune_sparse_graph,
    score_pairs,
)


def _make_candidate(
    candidate_id: str = "c1",
    *,
    technical_skills: list[str] | None = None,
    additional_info: str = "",
    skills: list[str] | None = None,
    experience_summary: str = "",
    education_summary: str = "",
    projects_summary: str = "",
    certifications: list[str] | None = None,
    languages: list[str] | None = None,
    major: str = "",
    skills_vec: list[float] | None = None,
    academic_vec: list[float] | None = None,
    practical_vec: list[float] | None = None,
) -> CandidateForAllocation:
    return CandidateForAllocation(
        candidate_id=candidate_id,
        metadata=CandidateMetadata(
            candidate_id=candidate_id,
            technical_skills=technical_skills or [],
            additional_info=additional_info,
            major=major,
        ),
        parsed_cv=ParsedCV(
            skills=skills or [],
            experience_summary=experience_summary,
            education_summary=education_summary,
            projects_summary=projects_summary,
            certifications=certifications or [],
            languages=languages or [],
        ),
        vectors=FacetedVectors(
            skills=skills_vec if skills_vec is not None else [1.0, 0.0, 0.0],
            academic=academic_vec if academic_vec is not None else [1.0, 0.0, 0.0],
            practical=practical_vec if practical_vec is not None else [1.0, 0.0, 0.0],
        ),
    )


def _make_position(
    position_id: str = "p1",
    *,
    title: str = "Backend Intern",
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    min_seats: int = 0,
    max_seats: int = 2,
    skills_vec: list[float] | None = None,
    academic_vec: list[float] | None = None,
    practical_vec: list[float] | None = None,
) -> PositionForAllocation:
    return PositionForAllocation(
        position_id=position_id,
        title=title,
        required_skills=required_skills or [],
        preferred_skills=preferred_skills or [],
        min_seats=min_seats,
        max_seats=max_seats,
        vectors=FacetedVectors(
            skills=skills_vec if skills_vec is not None else [1.0, 0.0, 0.0],
            academic=academic_vec if academic_vec is not None else [1.0, 0.0, 0.0],
            practical=practical_vec if practical_vec is not None else [1.0, 0.0, 0.0],
        ),
    )


def _manual_cosine(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    sim = float(np.dot(va / na, vb / nb))
    return min(1.0, max(0.0, sim))


class TestScorePairsValidation:
    def test_raises_on_empty_candidates_or_positions(self):
        cand = _make_candidate("c1")
        pos = _make_position("p1")
        with pytest.raises(ValueError, match="candidates"):
            score_pairs([], [pos])
        with pytest.raises(ValueError, match="positions"):
            score_pairs([cand], [])

    def test_raises_on_duplicate_candidate_or_position_ids(self):
        c1 = _make_candidate("dup-c")
        c2 = _make_candidate("dup-c")
        p1 = _make_position("p1")
        p2 = _make_position("p1")

        with pytest.raises(ValueError, match="Duplicate candidate_id"):
            score_pairs([c1, c2], [p1])
        with pytest.raises(ValueError, match="Duplicate position_id"):
            score_pairs([c1], [p1, p2])

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_raises_on_non_finite_values(self, bad_value: float):
        cand_bad = _make_candidate("c1", skills_vec=[bad_value, 1.0, 0.0])
        pos_ok = _make_position("p1")
        with pytest.raises(ValueError, match="non-finite"):
            score_pairs([cand_bad], [pos_ok])

        cand_ok = _make_candidate("c2")
        pos_bad = _make_position("p2", practical_vec=[1.0, bad_value, 0.0])
        with pytest.raises(ValueError, match="non-finite"):
            score_pairs([cand_ok], [pos_bad])

    def test_raises_on_dimension_mismatch_across_candidates_and_positions(self):
        c1 = _make_candidate("c1", skills_vec=[1.0, 0.0, 0.0])
        c2 = _make_candidate("c2", skills_vec=[1.0, 0.0])
        p1 = _make_position("p1", skills_vec=[1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            score_pairs([c1, c2], [p1])

        p_mismatch = _make_position("p2", academic_vec=[1.0, 0.0])
        with pytest.raises(ValueError, match="dimension"):
            score_pairs([c1], [p_mismatch])

    def test_raises_on_zero_length_or_non_1d_vector(self):
        c_empty = CandidateForAllocation.model_construct(
            candidate_id="c1",
            metadata=CandidateMetadata(candidate_id="c1"),
            parsed_cv=ParsedCV(),
            vectors=FacetedVectors.model_construct(
                skills=[], academic=[1.0], practical=[1.0]
            ),
        )
        p1 = _make_position(
            "p1", skills_vec=[1.0], academic_vec=[1.0], practical_vec=[1.0]
        )
        with pytest.raises(ValueError, match="non-empty"):
            score_pairs([c_empty], [p1])

    def test_raises_on_invalid_weights_type(self):
        cand = _make_candidate("c1")
        pos = _make_position("p1")
        with pytest.raises(ValueError, match="ScoringWeights"):
            score_pairs([cand], [pos], weights={"skills_semantic": 1.0})  # type: ignore[arg-type]

    def test_clips_total_scores_to_one_when_weights_have_tolerance_overshoot(self):
        cand = _make_candidate(
            "c1",
            technical_skills=["Python"],
            skills_vec=[1.0, 0.0],
            academic_vec=[1.0, 0.0],
            practical_vec=[1.0, 0.0],
        )
        pos = _make_position(
            "p1",
            required_skills=["Python"],
            preferred_skills=["Python"],
            skills_vec=[1.0, 0.0],
            academic_vec=[1.0, 0.0],
            practical_vec=[1.0, 0.0],
        )
        # ScoringWeights allows sum within 1e-6 of 1.0; total_scores must still clip to 1.0.
        weights = ScoringWeights(
            skills_semantic=0.35 + 5e-7,
            academic_semantic=0.15,
            practical_semantic=0.20,
            required_skills_lexical=0.20,
            preferred_skills_lexical=0.10,
        )
        result = score_pairs([cand], [pos], weights=weights)
        assert float(result.total_scores[0, 0]) == 1.0


class TestVectorizedSemanticScoring:
    def test_blas_dot_product_accuracy_and_defensive_normalization(self):
        # Intentionally unnormalized vectors with varied scales and negative angles.
        candidates = [
            _make_candidate(
                "c1",
                skills_vec=[3.0, 4.0, 0.0],
                academic_vec=[10.0, 0.0, 0.0],
                practical_vec=[-1.0, 0.0, 0.0],
            ),
            _make_candidate(
                "c2",
                skills_vec=[0.0, 0.0, 5.0],
                academic_vec=[1.0, 1.0, 1.0],
                practical_vec=[2.0, 2.0, 0.0],
            ),
        ]
        positions = [
            _make_position(
                "p1",
                skills_vec=[6.0, 8.0, 0.0],
                academic_vec=[0.0, 2.0, 0.0],
                practical_vec=[1.0, 0.0, 0.0],
            ),
            _make_position(
                "p2",
                skills_vec=[0.0, 1.0, 1.0],
                academic_vec=[3.0, 3.0, 3.0],
                practical_vec=[1.0, 1.0, 0.0],
            ),
        ]

        weights = ScoringWeights(
            skills_semantic=0.5,
            academic_semantic=0.3,
            practical_semantic=0.2,
            required_skills_lexical=0.0,
            preferred_skills_lexical=0.0,
        )
        result = score_pairs(candidates, positions, weights=weights)

        assert isinstance(result, ScoringResult)
        assert result.total_scores.shape == (2, 2)
        assert result.total_scores.dtype == np.float64
        assert result.candidate_ids == ("c1", "c2")
        assert result.position_ids == ("p1", "p2")

        for i, cand in enumerate(candidates):
            for j, pos in enumerate(positions):
                exp_skills = _manual_cosine(cand.vectors.skills, pos.vectors.skills)
                exp_acad = _manual_cosine(cand.vectors.academic, pos.vectors.academic)
                exp_prac = _manual_cosine(cand.vectors.practical, pos.vectors.practical)
                exp_total = 0.5 * exp_skills + 0.3 * exp_acad + 0.2 * exp_prac

                assert result.skills_semantic[i, j] == pytest.approx(exp_skills, abs=1e-12)
                assert result.academic_semantic[i, j] == pytest.approx(exp_acad, abs=1e-12)
                assert result.practical_semantic[i, j] == pytest.approx(exp_prac, abs=1e-12)
                assert result.total_scores[i, j] == pytest.approx(exp_total, abs=1e-12)

        # Negative dot product ([-1, 0, 0] vs [1, 0, 0]) must be clipped to 0.0.
        assert result.practical_semantic[0, 0] == 0.0

    def test_zero_vectors_remain_exact_zeros_and_produce_zero_similarity(self):
        c_zero = _make_candidate(
            "c_zero",
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        c_non_zero = _make_candidate(
            "c_ok",
            skills_vec=[1.0, 2.0, 2.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 3.0, 4.0],
        )
        p_zero = _make_position(
            "p_zero",
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[1.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        p_non_zero = _make_position(
            "p_ok",
            skills_vec=[1.0, 2.0, 2.0],
            academic_vec=[1.0, 0.0, 0.0],
            practical_vec=[0.0, 3.0, 4.0],
        )

        result = score_pairs([c_zero, c_non_zero], [p_zero, p_non_zero])

        assert result.skills_semantic[0, 0] == 0.0
        assert result.skills_semantic[0, 1] == 0.0
        assert result.skills_semantic[1, 0] == 0.0
        assert result.academic_semantic[1, 1] == 0.0
        assert result.practical_semantic[0, 1] == 0.0
        assert result.skills_semantic[1, 1] == pytest.approx(1.0, abs=1e-12)
        assert result.practical_semantic[1, 1] == pytest.approx(1.0, abs=1e-12)
        assert result.total_scores[0, 1] == 0.0


class TestLexicalCoverage:
    def test_empty_or_blank_skill_lists_award_zero_unearned_points(self):
        cand = _make_candidate(
            "c1",
            technical_skills=["Python", "SQL"],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        pos_empty = _make_position(
            "p_empty",
            required_skills=[],
            preferred_skills=["   ", ""],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )

        result = score_pairs([cand], [pos_empty])
        assert result.required_skills_lexical[0, 0] == 0.0
        assert result.preferred_skills_lexical[0, 0] == 0.0
        assert result.total_scores[0, 0] == 0.0

    def test_required_vs_preferred_weighting_and_deduplication(self):
        cand = _make_candidate(
            "c1",
            technical_skills=["Python"],
            skills=["FastAPI"],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        # Required deduplicates to {"python", "sql"} -> 1/2 = 0.5
        # Preferred deduplicates to {"fastapi", "docker"} -> 1/2 = 0.5
        pos = _make_position(
            "p1",
            required_skills=["Python", " python ", "SQL", ""],
            preferred_skills=["FastAPI", "FASTAPI", "Docker"],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )

        result = score_pairs([cand], [pos])
        assert result.required_skills_lexical[0, 0] == pytest.approx(0.5)
        assert result.preferred_skills_lexical[0, 0] == pytest.approx(0.5)
        # Default weights: 0.20 * 0.5 + 0.10 * 0.5 = 0.15
        assert result.total_scores[0, 0] == pytest.approx(0.15)

        # Verify required carries higher default weight than preferred.
        pos_req_only = _make_position(
            "p_req",
            required_skills=["Python"],
            preferred_skills=[],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        pos_pref_only = _make_position(
            "p_pref",
            required_skills=[],
            preferred_skills=["Python"],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        res_cmp = score_pairs([cand], [pos_req_only, pos_pref_only])
        assert res_cmp.total_scores[0, 0] == pytest.approx(0.20)
        assert res_cmp.total_scores[0, 1] == pytest.approx(0.10)
        assert res_cmp.total_scores[0, 0] > res_cmp.total_scores[0, 1]

    def test_boundary_safe_tech_symbols_and_no_false_substring_hits(self):
        cand_tricky = _make_candidate(
            "c_tricky",
            technical_skills=["JavaScript", "Computer Science", "C++", "Socket.io"],
            experience_summary="Built APIs in Node.js. Deployed Docker containers.",
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        cand_exact = _make_candidate(
            "c_exact",
            technical_skills=["C#/.NET"],
            skills=["C/C++"],
            experience_summary="Worked with Java. Proficient in C. Maintained C++.",
            projects_summary="Built backend in Node.js.",
            certifications=["AWS Certified"],
            additional_info="Familiar with R, SQL.",
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )

        # Must NOT false-hit Java in JavaScript, C in Computer/C++, io in Socket.io, Node in Node.js, R in Docker.
        pos_traps = _make_position(
            "p_traps",
            required_skills=["Java", "C", "io", "Node", "R"],
        )
        # Must match exact tech symbols including C++, C#, .NET, Node.js, C, Java, R.
        pos_tech = _make_position(
            "p_tech",
            required_skills=["C++", "C#", ".NET", "Node.js", "C", "Java", "AWS Certified"],
        )

        res = score_pairs([cand_tricky, cand_exact], [pos_traps, pos_tech])
        # cand_tricky matches none of Java, C, io, Node, R.
        assert res.required_skills_lexical[0, 0] == 0.0
        # cand_tricky matches C++ and Node.js -> 2/7.
        assert res.required_skills_lexical[0, 1] == pytest.approx(2 / 7)
        # cand_exact matches Java, C, R (3/5) on pos_traps and all 7/7 on pos_tech.
        assert res.required_skills_lexical[1, 0] == pytest.approx(3 / 5)
        assert res.required_skills_lexical[1, 1] == pytest.approx(1.0)

    def test_arabic_normalization_deduplication_and_corpus_boundaries(self):
        cand_ar = _make_candidate(
            "c_ar",
            technical_skills=["بايثُون", "Data"],
            skills=["Science"],
            experience_summary="خبرة في امن   المعلومات وتحليل البيانات.",
            education_summary="Java and Rust should not come from education_summary",
            languages=["French"],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        pos_ar = _make_position(
            "p_ar",
            required_skills=[
                "بايثون",
                "بـايثون",  # duplicates after tatweel removal
                "أمن المعلومات",
                "امن  المعلومات",  # duplicates after alef/whitespace normalization
                "برمجة",  # must not falsely match if absent
            ],
            preferred_skills=[
                "Data Science",  # must NOT match across separate technical_skills/skills items
                "Rust",  # must NOT match education_summary
                "French",  # must NOT match languages
            ],
        )

        res = score_pairs([cand_ar], [pos_ar])
        # Deduplicated required skills = {"بايثون", "امن المعلومات", "برمجة"} -> 2/3 matched.
        assert res.required_skills_lexical[0, 0] == pytest.approx(2 / 3)
        assert res.preferred_skills_lexical[0, 0] == 0.0

    def test_compound_separator_boundary_traps(self):
        cand = _make_candidate(
            "c_compound",
            technical_skills=["Objective-C", "R&D", "React-Native", "T-SQL"],
            experience_summary="Conducted R&D in Objective-C and React-Native with T-SQL.",
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        pos_false_traps = _make_position(
            "p_traps",
            required_skills=["C", "R", "React", "SQL", "D", "Native", "T"],
        )
        pos_exact = _make_position(
            "p_exact",
            required_skills=["Objective-C", "R&D", "React-Native", "T-SQL"],
        )
        res = score_pairs([cand], [pos_false_traps, pos_exact])
        assert res.required_skills_lexical[0, 0] == 0.0
        assert res.required_skills_lexical[0, 1] == pytest.approx(1.0)

    def test_nfd_vs_nfc_unicode_matching_arabic_and_latin(self):
        nfd_arabic = unicodedata.normalize("NFD", "مؤتمر")
        nfd_latin = unicodedata.normalize("NFD", "Café")
        assert nfd_arabic != unicodedata.normalize("NFC", "مؤتمر")
        assert nfd_latin != unicodedata.normalize("NFC", "Café")

        cand = _make_candidate(
            "c_nfd",
            technical_skills=[nfd_arabic, nfd_latin],
            skills_vec=[0.0, 0.0, 0.0],
            academic_vec=[0.0, 0.0, 0.0],
            practical_vec=[0.0, 0.0, 0.0],
        )
        pos = _make_position(
            "p_nfc",
            required_skills=[
                unicodedata.normalize("NFC", "مؤتمر"),
                unicodedata.normalize("NFC", "Café"),
            ],
            preferred_skills=[nfd_arabic],
        )
        res = score_pairs([cand], [pos])
        assert res.required_skills_lexical[0, 0] == pytest.approx(1.0)
        assert res.preferred_skills_lexical[0, 0] == pytest.approx(1.0)


class TestScoringResultAuditHelpers:
    def test_breakdown_and_top_eligible_alternatives(self):
        c1 = _make_candidate(
            "c1",
            technical_skills=["Python"],
            skills_vec=[1.0, 0.0],
            academic_vec=[1.0, 0.0],
            practical_vec=[0.0, 1.0],
        )
        positions = [
            _make_position(
                "p0",
                required_skills=["Python"],
                skills_vec=[1.0, 0.0],
                academic_vec=[1.0, 0.0],
                practical_vec=[0.0, 1.0],
            ),
            _make_position(
                "p1",
                required_skills=[],
                skills_vec=[0.8, 0.6],
                academic_vec=[1.0, 0.0],
                practical_vec=[0.0, 1.0],
            ),
            _make_position(
                "p2",
                required_skills=[],
                skills_vec=[0.8, 0.6],
                academic_vec=[1.0, 0.0],
                practical_vec=[0.0, 1.0],
            ),
            _make_position(
                "p3",
                required_skills=["Python"],
                skills_vec=[1.0, 0.0],
                academic_vec=[1.0, 0.0],
                practical_vec=[0.0, 1.0],
            ),
        ]

        scoring = score_pairs([c1], positions)
        bd = scoring.breakdown(0, 0)
        assert isinstance(bd, ScoreBreakdown)
        assert bd.skills_semantic == pytest.approx(1.0)
        assert bd.academic_semantic == pytest.approx(1.0)
        assert bd.practical_semantic == pytest.approx(1.0)
        assert bd.required_skills_lexical == pytest.approx(1.0)
        assert bd.preferred_skills_lexical == 0.0

        # p3 is ineligible; p0 excluded (e.g. assigned); p1 and p2 tie -> ordered p1 then p2.
        eligibility = np.array([[True, True, True, False]], dtype=bool)
        alts = scoring.top_eligible_alternatives(
            0, eligibility, exclude_position_idx=0, limit=2
        )
        assert len(alts) == 2
        assert all(isinstance(alt, AlternativePlacement) for alt in alts)
        assert [alt.position_id for alt in alts] == ["p1", "p2"]
        assert alts[0].score == pytest.approx(float(scoring.total_scores[0, 1]))
        assert alts[0].score_breakdown == scoring.breakdown(0, 1)

        with pytest.raises(IndexError):
            scoring.breakdown(-1, 0)
        with pytest.raises(IndexError):
            scoring.breakdown(0, 4)
        with pytest.raises(IndexError):
            scoring.breakdown(np.bool_(False), 0)  # type: ignore[arg-type]

        # NumPy integer scalars (e.g. from np.flatnonzero / np.argmax) must be accepted.
        assert scoring.breakdown(np.int64(0), np.int64(0)) == bd
        alts_np = scoring.top_eligible_alternatives(
            np.int64(0), eligibility, exclude_position_idx=np.int64(0), limit=2
        )
        assert [alt.position_id for alt in alts_np] == ["p1", "p2"]

        assert (
            scoring.top_eligible_alternatives(0, eligibility, exclude_position_idx=0, limit=0)
            == []
        )
        with pytest.raises(ValueError, match="limit"):
            scoring.top_eligible_alternatives(0, eligibility, limit=-1)
        with pytest.raises(IndexError):
            scoring.top_eligible_alternatives(0, eligibility, exclude_position_idx=4)


class TestPruneSparseGraph:
    def _custom_scoring(self, scores: np.ndarray) -> ScoringResult:
        n, m = scores.shape
        zeros = np.zeros((n, m), dtype=np.float64)
        return ScoringResult(
            total_scores=scores.astype(np.float64),
            skills_semantic=scores.astype(np.float64),
            academic_semantic=zeros,
            practical_semantic=zeros,
            required_skills_lexical=zeros,
            preferred_skills_lexical=zeros,
            candidate_ids=tuple(f"c{i}" for i in range(n)),
            position_ids=tuple(f"p{j}" for j in range(m)),
        )

    def test_validates_inputs(self):
        scoring = self._custom_scoring(np.array([[0.5, 0.6]], dtype=np.float64))
        good_elig = np.array([[True, True]], dtype=bool)

        with pytest.raises(ValueError, match="dtype bool"):
            prune_sparse_graph(scoring, np.array([[1, 1]], dtype=int))
        with pytest.raises(ValueError, match="shape"):
            prune_sparse_graph(scoring, np.array([[True]], dtype=bool))
        with pytest.raises(ValueError, match="min_match_threshold"):
            prune_sparse_graph(scoring, good_elig, min_match_threshold=-0.01)
        with pytest.raises(ValueError, match="min_match_threshold"):
            prune_sparse_graph(scoring, good_elig, min_match_threshold=1.01)
        with pytest.raises(ValueError, match="min_match_threshold"):
            prune_sparse_graph(scoring, good_elig, min_match_threshold=float("nan"))
        with pytest.raises(ValueError, match="top_k_per_candidate"):
            prune_sparse_graph(scoring, good_elig, top_k_per_candidate=0)
        with pytest.raises(ValueError, match="top_k_per_candidate"):
            prune_sparse_graph(scoring, good_elig, top_k_per_candidate=True)

    def test_eligibility_masking_threshold_boundary_and_diagnostics(self):
        scores = np.array(
            [
                [0.95, 0.45, 0.4499, 0.81235],  # p0 ineligible, p1 == threshold, p2 < threshold, p3 > threshold
                [0.99, 0.99, 0.99, 0.99],       # all ineligible
                [0.99, 0.20, 0.44, 0.10],       # p0 high score but ineligible; eligible p1..p3 all < 0.45
            ],
            dtype=np.float64,
        )
        elig = np.array(
            [
                [False, True, True, True],
                [False, False, False, False],
                [False, True, True, True],
            ],
            dtype=bool,
        )
        scoring = self._custom_scoring(scores)
        graph = prune_sparse_graph(
            scoring, elig, min_match_threshold=0.45, top_k_per_candidate=12
        )

        assert isinstance(graph, PrunedGraph)
        assert graph.has_eligible_position == (True, False, True)
        assert graph.has_above_threshold_position == (True, False, False)

        # Candidate 0 keeps p3 (0.81235) and p1 (0.45 == threshold), excludes p0 (ineligible) & p2 (< 0.45).
        c0_edges = graph.edges_by_candidate[0]
        assert [e.position_idx for e in c0_edges] == [3, 1]
        assert c0_edges[0] == SparseEdge(
            candidate_idx=0,
            position_idx=3,
            candidate_id="c0",
            position_id="p3",
            score=0.81235,
            scaled_weight=int(round(0.81235 * 10_000)),
            breakdown=scoring.breakdown(0, 3),
        )
        assert c0_edges[1].scaled_weight == 4500

        # Candidates 1 and 2 have no surviving edges.
        assert graph.edges_by_candidate[1] == []
        assert graph.edges_by_candidate[2] == []
        assert graph.edges == c0_edges

    def test_argpartition_top_k_when_m_gt_k_and_m_le_k_with_deterministic_ties(self, monkeypatch):
        # Row 0: 6 surviving > K=3 (with masked 0.99 & 0.98 that must never leak).
        # Row 1: 3 surviving == K=3 (M <= K, ties at 0.80).
        # Row 2: 2 surviving < K=3 (M <= K).
        scores = np.array(
            [
                [0.99, 0.60, 0.85, 0.70, 0.70, 0.70, 0.98, 0.75],
                [0.50, 0.80, 0.80, 0.40, 0.60, 0.55, 0.90, 0.45],
                [0.50, 0.70, 0.95, 0.20, 0.80, 0.80, 0.80, 0.80],
            ],
            dtype=np.float64,
        )
        elig = np.array(
            [
                [False, True, True, True, True, True, False, True],
                [True, True, True, False, False, False, False, False],
                [True, True, False, False, False, False, False, False],
            ],
            dtype=bool,
        )
        scoring = self._custom_scoring(scores)

        argpartition_calls = {"count": 0}
        real_argpartition = np.argpartition

        def spy_argpartition(a, kth, *args, **kwargs):
            argpartition_calls["count"] += 1
            # Masked high-score entries must already be excluded before argpartition.
            assert np.all(np.isfinite(a))
            assert float(np.max(-a)) <= 0.85 + 1e-12
            return real_argpartition(a, kth, *args, **kwargs)

        monkeypatch.setattr(scorer_module.np, "argpartition", spy_argpartition)

        graph = prune_sparse_graph(
            scoring, elig, min_match_threshold=0.45, top_k_per_candidate=3
        )

        # Called only for candidate 0 (6 surviving > 3), NOT for candidates 1 (3 <= 3) or 2 (2 <= 3).
        assert argpartition_calls["count"] == 1

        # Candidate 0 top 3 from {p1:0.60, p2:0.85, p3:0.70, p4:0.70, p5:0.70, p7:0.75}:
        # p2 (0.85), p7 (0.75), and deterministic tie-break picks lowest idx p3 (0.70) over p4/p5.
        assert [e.position_idx for e in graph.edges_by_candidate[0]] == [2, 7, 3]
        assert [e.scaled_weight for e in graph.edges_by_candidate[0]] == [8500, 7500, 7000]

        # Candidate 1 has 3 surviving == top_k=3; tie at 0.80 ordered p1 then p2; p0 (0.50) third.
        assert [e.position_idx for e in graph.edges_by_candidate[1]] == [1, 2, 0]
        assert [e.scaled_weight for e in graph.edges_by_candidate[1]] == [8000, 8000, 5000]

        # Candidate 2 has 2 surviving < top_k=3; ordered p1 (0.70) then p0 (0.50).
        assert [e.position_idx for e in graph.edges_by_candidate[2]] == [1, 0]
        assert [e.scaled_weight for e in graph.edges_by_candidate[2]] == [7000, 5000]

    def test_scaled_weight_rounding_precision(self):
        scores = np.array([[0.12344, 0.12346, 0.99996, 0.45]], dtype=np.float64)
        elig = np.array([[True, True, True, True]], dtype=bool)
        scoring = self._custom_scoring(scores)

        graph = prune_sparse_graph(
            scoring, elig, min_match_threshold=0.10, top_k_per_candidate=10
        )
        by_pos = {e.position_idx: e.scaled_weight for e in graph.edges}
        assert by_pos == {
            0: int(round(0.12344 * 10_000)),
            1: int(round(0.12346 * 10_000)),
            2: int(round(0.99996 * 10_000)),
            3: int(round(0.45 * 10_000)),
        }
        assert by_pos[2] == 10_000
        assert math.isclose(graph.edges[0].score, 0.99996)

    def test_top_k_one_with_tied_surviving_scores_picks_lowest_position_idx(self):
        scores = np.array([[0.90, 0.75, 0.75, 0.75]], dtype=np.float64)
        elig = np.array([[False, True, True, True]], dtype=bool)
        scoring = self._custom_scoring(scores)

        graph = prune_sparse_graph(
            scoring, elig, min_match_threshold=0.45, top_k_per_candidate=1
        )
        assert len(graph.edges_by_candidate[0]) == 1
        assert graph.edges_by_candidate[0][0].position_idx == 1
        assert graph.edges_by_candidate[0][0].position_id == "p1"
        assert graph.edges_by_candidate[0][0].scaled_weight == 7500
