from __future__ import annotations

import math

import pytest

from app.schemas.allocation import (
    AllocationSolveRequest,
    AssignmentResult,
    LockedOverride,
    ScoringWeights,
    UnassignedCandidate,
)
from app.schemas.candidate import CandidateForAllocation, CandidateMetadata, ParsedCV
from app.schemas.position import (
    FacetedVectors,
    PositionConstraints,
    PositionEmbedRequest,
    PositionForAllocation,
)
from app.services.constraints import EligibilityResult, evaluate_eligibility
from app.services.embedder import (
    EmbeddingModel,
    build_candidate_vectors,
    build_position_vectors,
)
from app.services.scorer import (
    SCORE_SCALE_FACTOR,
    PrunedGraph,
    ScoringResult,
    prune_sparse_graph,
    score_pairs,
)
from tests.conftest import FakeSentenceTransformer

_UNIT_4D = [1.0, 0.0, 0.0, 0.0]


def _vec(
    skills: list[float] | None = None,
    academic: list[float] | None = None,
    practical: list[float] | None = None,
) -> FacetedVectors:
    return FacetedVectors(
        skills=list(_UNIT_4D) if skills is None else skills,
        academic=list(_UNIT_4D) if academic is None else academic,
        practical=list(_UNIT_4D) if practical is None else practical,
    )


def _candidate(
    candidate_id: str,
    *,
    city: str | None = "Damascus",
    gpa: float | None = 85.0,
    year_of_study: int | None = 4,
    full_time_commitment: bool = True,
    program: str = "",
    field: str = "",
    major: str = "Computer Science",
    technical_skills: list[str] | None = None,
    additional_info: str = "",
    skills: list[str] | None = None,
    experience_summary: str = "",
    education_summary: str = "",
    projects_summary: str = "",
    certifications: list[str] | None = None,
    languages: list[str] | None = None,
    vectors: FacetedVectors | None = None,
) -> CandidateForAllocation:
    return CandidateForAllocation(
        candidate_id=candidate_id,
        metadata=CandidateMetadata(
            candidate_id=candidate_id,
            city=city,
            gpa=gpa,
            year_of_study=year_of_study,
            full_time_commitment=full_time_commitment,
            program=program,
            field=field,
            major=major,
            technical_skills=[] if technical_skills is None else technical_skills,
            additional_info=additional_info,
        ),
        parsed_cv=ParsedCV(
            skills=[] if skills is None else skills,
            experience_summary=experience_summary,
            education_summary=education_summary,
            projects_summary=projects_summary,
            certifications=[] if certifications is None else certifications,
            languages=[] if languages is None else languages,
        ),
        vectors=_vec() if vectors is None else vectors,
    )


def _position(
    position_id: str,
    *,
    title: str = "Software Engineering Intern",
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    responsibilities: str = "",
    min_seats: int = 0,
    max_seats: int = 2,
    vectors: FacetedVectors | None = None,
    **constraint_overrides,
) -> PositionForAllocation:
    return PositionForAllocation(
        position_id=position_id,
        title=title,
        required_skills=[] if required_skills is None else required_skills,
        preferred_skills=[] if preferred_skills is None else preferred_skills,
        responsibilities=responsibilities,
        min_seats=min_seats,
        max_seats=max_seats,
        constraints=PositionConstraints(**constraint_overrides),
        vectors=_vec() if vectors is None else vectors,
    )


def _run_pipeline(
    request: AllocationSolveRequest,
) -> tuple[EligibilityResult, ScoringResult, PrunedGraph]:
    """Execute Phase 5 Stages 1 -> 2 -> 3 end-to-end from an AllocationSolveRequest."""
    eligibility = evaluate_eligibility(request.candidates, request.positions)
    scoring = score_pairs(
        request.candidates,
        request.positions,
        weights=request.weights,
    )
    graph = prune_sparse_graph(
        scoring,
        eligibility.eligibility_matrix,
        min_match_threshold=request.min_match_threshold,
        top_k_per_candidate=request.top_k_per_candidate,
        locked_overrides=request.locked_overrides,
    )
    return eligibility, scoring, graph


def _embedded_candidate(
    candidate_id: str,
    model: EmbeddingModel,
    *,
    meta_kwargs: dict | None = None,
    cv_kwargs: dict | None = None,
) -> CandidateForAllocation:
    metadata = CandidateMetadata(candidate_id=candidate_id, **(meta_kwargs or {}))
    parsed_cv = ParsedCV(**(cv_kwargs or {}))
    vectors = build_candidate_vectors(metadata, parsed_cv, model)
    return CandidateForAllocation(
        candidate_id=candidate_id,
        metadata=metadata,
        parsed_cv=parsed_cv,
        vectors=vectors,
    )


def _embedded_position(
    position_id: str,
    model: EmbeddingModel,
    *,
    title: str = "Backend Intern",
    department: str | None = None,
    description: str = "",
    responsibilities: str = "",
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    target_majors: list[str] | None = None,
    academic_level: str = "",
    experience_level: str = "",
    min_seats: int = 0,
    max_seats: int = 2,
    constraints_kwargs: dict | None = None,
) -> PositionForAllocation:
    req_skills = [] if required_skills is None else required_skills
    pref_skills = [] if preferred_skills is None else preferred_skills
    embed_req = PositionEmbedRequest(
        position_id=position_id,
        title=title,
        department=department,
        description=description,
        responsibilities=responsibilities,
        required_skills=req_skills,
        preferred_skills=pref_skills,
        target_majors=[] if target_majors is None else target_majors,
        academic_level=academic_level,
        experience_level=experience_level,
    )
    vectors = build_position_vectors(embed_req, model)
    return PositionForAllocation(
        position_id=position_id,
        title=title,
        required_skills=req_skills,
        preferred_skills=pref_skills,
        responsibilities=responsibilities,
        min_seats=min_seats,
        max_seats=max_seats,
        constraints=PositionConstraints(**(constraints_kwargs or {})),
        vectors=vectors,
    )


class TestFullThreeStagePipeline:
    def test_high_scoring_disqualified_pairs_masked_with_exact_summary(self):
        cand = _candidate(
            "c_high_score_filtered",
            city="Homs",
            gpa=74.0,
            year_of_study=2,
            full_time_commitment=False,
            major="Electrical Engineering",
            technical_skills=["Python", "SQL"],
            skills=["Docker"],
            experience_summary="Built production REST APIs.",
        )
        positions = [
            _position(
                "p_city",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                allowed_cities=["Damascus", "Aleppo"],
            ),
            _position(
                "p_gpa",
                required_skills=["Python"],
                preferred_skills=["Docker"],
                min_gpa=80.0,
            ),
            _position(
                "p_major",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                required_majors=["Computer Science", "Software Engineering"],
            ),
            _position(
                "p_kw",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                must_have_keywords=["Kubernetes", "GraphQL"],
            ),
            _position(
                "p_year_ft",
                required_skills=["Python"],
                preferred_skills=["Docker"],
                min_year_of_study=3,
                requires_full_time_commitment=True,
            ),
            _position(
                "p_eligible",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                allowed_cities=["Homs"],
                min_gpa=70.0,
                min_year_of_study=2,
                requires_full_time_commitment=False,
                required_majors=["Electrical Engineering"],
                must_have_keywords=["REST APIs"],
            ),
        ]

        request = AllocationSolveRequest(candidates=[cand], positions=positions)
        eligibility, scoring, graph = _run_pipeline(request)

        # Every position scores 1.0 in Stage 2, proving Stage 1 drives the pruning.
        for pos_idx in range(len(positions)):
            assert scoring.total_scores[0, pos_idx] == pytest.approx(1.0)

        assert eligibility.eligibility_matrix.tolist() == [
            [False, False, False, False, False, True]
        ]
        assert graph.has_eligible_position == (True,)
        assert graph.has_above_threshold_position == (True,)
        assert [edge.position_id for edge in graph.edges] == ["p_eligible"]
        assert [edge.position_id for edge in graph.edges_by_candidate[0]] == [
            "p_eligible"
        ]
        assert graph.edges[0].scaled_weight == SCORE_SCALE_FACTOR

        summary = eligibility.disqualification_summary_for_candidate(0)
        assert summary == {
            "p_city": ["Candidate city 'Homs' is not in allowed cities: Damascus, Aleppo."],
            "p_gpa": ["Candidate GPA 74.0 is below minimum required 80.0."],
            "p_major": [
                "Candidate major 'Electrical Engineering' does not match required "
                "majors: Computer Science, Software Engineering."
            ],
            "p_kw": ["Missing required keywords: Kubernetes, GraphQL."],
            "p_year_ft": [
                "Candidate year of study 2 is below minimum required 3.",
                "Position requires full-time commitment.",
            ],
        }
        assert "p_eligible" not in summary
        for pos_idx, pid in enumerate(eligibility.position_ids[:-1]):
            assert eligibility.reasons_for_pair(0, pos_idx) == summary[pid]

        assert (
            scoring.top_eligible_alternatives(
                0, eligibility.eligibility_matrix, exclude_position_idx=5, limit=2
            )
            == []
        )

    def test_eligible_candidate_below_threshold_masked_with_alternatives_and_flags(self):
        # Default weights: 0.35 skills_sem + 0.15 acad_sem + 0.20 prac_sem + 0.20 req + 0.10 pref
        cand = _candidate(
            "c_below_thresh",
            city="Damascus",
            gpa=78.0,
            major="Computer Science",
            technical_skills=["Git"],
            vectors=_vec(
                skills=[1.0, 0.0, 0.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[1.0, 0.0, 0.0, 0.0],
            ),
        )
        positions = [
            # Eligible, score = 0.35*0.6 + 0.15*0.8 + 0.20*0.0 + 0.20*0.0 + 0.10*0.0 = 0.33 < 0.45
            _position(
                "p_elig_low_best",
                required_skills=["Rust"],
                preferred_skills=["Go"],
                allowed_cities=["Damascus"],
                min_gpa=75.0,
                vectors=_vec(
                    skills=[0.6, 0.8, 0.0, 0.0],
                    academic=[0.8, 0.6, 0.0, 0.0],
                    practical=[0.0, 1.0, 0.0, 0.0],
                ),
            ),
            # Eligible, score = 0.15 < 0.45
            _position(
                "p_elig_low_second",
                required_skills=["Java"],
                preferred_skills=["Spring"],
                allowed_cities=["Damascus"],
                vectors=_vec(
                    skills=[0.0, 1.0, 0.0, 0.0],
                    academic=[1.0, 0.0, 0.0, 0.0],
                    practical=[0.0, 1.0, 0.0, 0.0],
                ),
            ),
            # Ineligible (GPA), yet perfect semantic + lexical match (1.0)
            _position(
                "p_inelig_perfect",
                required_skills=["Git"],
                preferred_skills=["Git"],
                allowed_cities=["Damascus"],
                min_gpa=90.0,
                vectors=_vec(),
            ),
        ]

        request = AllocationSolveRequest(
            candidates=[cand],
            positions=positions,
            min_match_threshold=0.45,
        )
        eligibility, scoring, graph = _run_pipeline(request)

        assert eligibility.eligibility_matrix.tolist() == [[True, True, False]]
        assert scoring.total_scores[0, 0] == pytest.approx(0.33)
        assert scoring.total_scores[0, 1] == pytest.approx(0.15)
        assert scoring.total_scores[0, 2] == pytest.approx(1.0)

        assert graph.edges == []
        assert graph.edges_by_candidate[0] == []
        assert graph.has_eligible_position == (True,)
        assert graph.has_above_threshold_position == (False,)

        alts = scoring.top_eligible_alternatives(
            0, eligibility.eligibility_matrix, limit=2
        )
        assert [alt.position_id for alt in alts] == [
            "p_elig_low_best",
            "p_elig_low_second",
        ]
        assert alts[0].score == pytest.approx(0.33)
        assert alts[1].score == pytest.approx(0.15)

        unassigned = UnassignedCandidate(
            candidate_id=cand.candidate_id,
            reason="below_threshold",
            alternatives=alts,
            disqualification_summary=eligibility.disqualification_summary_for_candidate(0),
        )
        assert unassigned.disqualification_summary == {
            "p_inelig_perfect": ["Candidate GPA 78.0 is below minimum required 90.0."]
        }

    def test_all_ineligible_candidate_has_false_flags_and_empty_alternatives(self):
        cand = _candidate(
            "c_all_ineligible",
            city="Latakia",
            gpa=68.0,
            major="Physics",
            technical_skills=["Python", "SQL"],
            vectors=_vec(),
        )
        positions = [
            _position(
                "p0_city",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                allowed_cities=["Damascus"],
                vectors=_vec(),
            ),
            _position(
                "p1_gpa_major",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                min_gpa=75.0,
                required_majors=["Computer Science"],
                vectors=_vec(),
            ),
            _position(
                "p2_keywords",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                must_have_keywords=["Terraform"],
                vectors=_vec(),
            ),
        ]

        request = AllocationSolveRequest(candidates=[cand], positions=positions)
        eligibility, scoring, graph = _run_pipeline(request)

        for pos_idx in range(len(positions)):
            assert scoring.total_scores[0, pos_idx] == pytest.approx(1.0)

        assert eligibility.eligibility_matrix.tolist() == [[False, False, False]]
        assert graph.edges == []
        assert graph.edges_by_candidate[0] == []
        assert graph.has_eligible_position == (False,)
        assert graph.has_above_threshold_position == (False,)

        alts = scoring.top_eligible_alternatives(
            0, eligibility.eligibility_matrix, limit=2
        )
        assert alts == []

        summary = eligibility.disqualification_summary_for_candidate(0)
        assert summary == {
            "p0_city": ["Candidate city 'Latakia' is not in allowed cities: Damascus."],
            "p1_gpa_major": [
                "Candidate GPA 68.0 is below minimum required 75.0.",
                "Candidate major 'Physics' does not match required majors: Computer Science.",
            ],
            "p2_keywords": ["Missing required keywords: Terraform."],
        }

        unassigned = UnassignedCandidate(
            candidate_id=cand.candidate_id,
            reason="all_ineligible",
            alternatives=alts,
            disqualification_summary=summary,
        )
        assert unassigned.reason == "all_ineligible"
        assert unassigned.alternatives == []


class TestBilingualAndLanguageConsistency:
    def test_arabic_variants_and_languages_consistent_across_constraints_and_scorer(self):
        model = EmbeddingModel("BAAI/bge-m3", _model=FakeSentenceTransformer(dimension=8))

        cand_h_variant = _embedded_candidate(
            "c_ar_h",
            model,
            meta_kwargs={
                "city": "اللاذقيه",
                "gpa": 82.0,
                "major": "هندسه معلوماتيه",
                "technical_skills": ["بايثون", "JavaScript"],
            },
            cv_kwargs={
                "skills": ["FastAPI"],
                "education_summary": "بكالوريوس هندسه معلوماتيه",
                "experience_summary": "تطوير واجهات برمجية في امن المعلومات",
                "languages": ["English", "Arabic", "العربيه"],
            },
        )
        cand_t_variant = _embedded_candidate(
            "c_ar_t",
            model,
            meta_kwargs={
                "city": "اللاذقية",
                "gpa": 85.0,
                "major": "هندسة معلوماتية",
                "technical_skills": ["بايثون", "JavaScript"],
            },
            cv_kwargs={
                "skills": ["FastAPI"],
                "education_summary": "بكالوريوس هندسة معلوماتية",
                "experience_summary": "تطوير واجهات برمجية في أمن المعلومات",
                "languages": ["English", "Arabic", "العربية"],
            },
        )

        positions = [
            _embedded_position(
                "p_bilingual_t",
                model,
                title="مهندس برمجيات",
                required_skills=["English", "Arabic", "هندسة معلوماتية", "بايثون"],
                preferred_skills=["العربية", "أمن المعلومات"],
                target_majors=["هندسة معلوماتية"],
                responsibilities="تطوير واجهات برمجية",
                constraints_kwargs={
                    "allowed_cities": ["اللاذقية"],
                    "required_majors": ["هندسة معلوماتية"],
                    "must_have_keywords": ["English", "Arabic", "هندسة معلوماتية"],
                },
            ),
            _embedded_position(
                "p_bilingual_h",
                model,
                title="مهندس شبكات",
                required_skills=["English", "Arabic", "هندسه معلوماتيه", "بايثون"],
                preferred_skills=["العربيه", "امن المعلومات"],
                target_majors=["هندسه معلوماتيه"],
                responsibilities="تطوير واجهات برمجية",
                constraints_kwargs={
                    "allowed_cities": ["اللاذقيه"],
                    "required_majors": ["هندسه معلوماتيه"],
                    "must_have_keywords": ["English", "Arabic", "العربيه"],
                },
            ),
            # Cross-stage corpus boundary check: city must NOT satisfy must_have_keywords OR lexical skills.
            _embedded_position(
                "p_city_excluded_from_corpus",
                model,
                title="City Keyword Trap",
                required_skills=["اللاذقية"],
                preferred_skills=[],
                target_majors=["هندسة معلوماتية"],
                responsibilities="تطوير واجهات برمجية",
                constraints_kwargs={
                    "allowed_cities": ["اللاذقية"],
                    "must_have_keywords": ["اللاذقية"],
                },
            ),
            # Cross-stage token boundary check: JavaScript must NOT satisfy "Java".
            _embedded_position(
                "p_boundary_trap",
                model,
                title="Java Trap",
                required_skills=["Java"],
                preferred_skills=[],
                target_majors=["هندسة معلوماتية"],
                responsibilities="تطوير واجهات برمجية",
                constraints_kwargs={
                    "allowed_cities": ["اللاذقية"],
                    "must_have_keywords": ["Java"],
                },
            ),
        ]

        request = AllocationSolveRequest(
            candidates=[cand_h_variant, cand_t_variant],
            positions=positions,
        )
        eligibility, scoring, graph = _run_pipeline(request)

        # Both Arabic spelling variants pass both bilingual positions and fail both traps identically.
        assert eligibility.eligibility_matrix.tolist() == [
            [True, True, False, False],
            [True, True, False, False],
        ]

        for cand_idx in (0, 1):
            for pos_idx in (0, 1):
                assert scoring.required_skills_lexical[cand_idx, pos_idx] == pytest.approx(1.0)
                assert scoring.preferred_skills_lexical[cand_idx, pos_idx] == pytest.approx(1.0)
                assert scoring.total_scores[cand_idx, pos_idx] >= 0.45

            # City and partial-token traps consistently fail Stage 1 AND score 0.0 lexical in Stage 2.
            assert scoring.required_skills_lexical[cand_idx, 2] == 0.0
            assert scoring.required_skills_lexical[cand_idx, 3] == 0.0
            assert eligibility.disqualification_summary_for_candidate(cand_idx) == {
                "p_city_excluded_from_corpus": ["Missing required keywords: اللاذقية."],
                "p_boundary_trap": ["Missing required keywords: Java."],
            }
            assert [e.position_id for e in graph.edges_by_candidate[cand_idx]] == [
                "p_bilingual_t",
                "p_bilingual_h",
            ]


class TestCrossPositionActiveWeightRenormalization:
    def test_full_vs_no_preferred_vs_no_skills_zero_vec_from_embedder_achieve_parity(self):
        model = EmbeddingModel("BAAI/bge-m3", _model=FakeSentenceTransformer(dimension=8))

        cand = _embedded_candidate(
            "c_matched",
            model,
            meta_kwargs={
                "city": "Damascus",
                "gpa": 88.0,
                "major": "Computer Science",
                "technical_skills": ["Python", "SQL"],
            },
            cv_kwargs={
                "experience_summary": "Build scalable backend APIs",
            },
        )

        pos_full = _embedded_position(
            "p_full",
            model,
            required_skills=["Python"],
            preferred_skills=["SQL"],
            target_majors=["Computer Science"],
            responsibilities="Build scalable backend APIs",
        )
        pos_no_pref = _embedded_position(
            "p_no_pref",
            model,
            required_skills=["Python", "SQL"],
            preferred_skills=[],
            target_majors=["Computer Science"],
            responsibilities="Build scalable backend APIs",
        )
        pos_no_skills = _embedded_position(
            "p_no_skills",
            model,
            required_skills=[],
            preferred_skills=[],
            target_majors=["Computer Science"],
            responsibilities="Build scalable backend APIs",
        )

        # Embedder emits exact zero vector for empty position skills.
        assert pos_no_skills.vectors.skills == [0.0] * 8

        request = AllocationSolveRequest(
            candidates=[cand],
            positions=[pos_full, pos_no_pref, pos_no_skills],
            min_match_threshold=0.45,
        )
        eligibility, scoring, graph = _run_pipeline(request)

        assert eligibility.eligibility_matrix.tolist() == [[True, True, True]]
        assert scoring.total_scores[0, 0] == pytest.approx(1.0)
        assert scoring.total_scores[0, 1] == pytest.approx(1.0)
        assert scoring.total_scores[0, 2] == pytest.approx(1.0)

        # Raw sub-score breakdowns must stay uninflated.
        bd_no_pref = scoring.breakdown(0, 1)
        assert bd_no_pref.preferred_skills_lexical == 0.0
        assert bd_no_pref.required_skills_lexical == pytest.approx(1.0)

        bd_no_skills = scoring.breakdown(0, 2)
        assert bd_no_skills.skills_semantic == 0.0
        assert bd_no_skills.required_skills_lexical == 0.0
        assert bd_no_skills.preferred_skills_lexical == 0.0
        assert bd_no_skills.academic_semantic == pytest.approx(1.0)
        assert bd_no_skills.practical_semantic == pytest.approx(1.0)

        assert [e.position_id for e in graph.edges] == [
            "p_full",
            "p_no_pref",
            "p_no_skills",
        ]
        assert all(e.scaled_weight == SCORE_SCALE_FACTOR for e in graph.edges)

    def test_partial_match_fair_parity_and_candidate_side_penalty_asymmetry(self):
        # Candidate 0 matches partial facets (~0.60); Candidate 1 lacks skills (penalized on full-spec).
        c_partial = _candidate(
            "c_partial",
            technical_skills=["Python"],
            vectors=_vec(
                skills=[1.0, 0.0, 0.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[1.0, 0.0, 0.0, 0.0],
            ),
        )
        c_no_skills = _candidate(
            "c_no_skills",
            technical_skills=[],
            vectors=_vec(
                skills=[0.0, 0.0, 0.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[1.0, 0.0, 0.0, 0.0],
            ),
        )

        # 0.6 cosine on each active facet -> 0.6*0.35 + 0.6*0.15 + 0.6*0.20 + 0.5*0.20 + 0.0*0.10 = 0.52 (full)
        positions = [
            _position(
                "p_full",
                required_skills=["Python", "Go"],
                preferred_skills=["Rust"],
                vectors=_vec(
                    skills=[0.6, 0.8, 0.0, 0.0],
                    academic=[0.6, 0.8, 0.0, 0.0],
                    practical=[0.6, 0.8, 0.0, 0.0],
                ),
            ),
            _position(
                "p_no_pref",
                required_skills=["Python", "Go"],
                preferred_skills=[],
                vectors=_vec(
                    skills=[0.6, 0.8, 0.0, 0.0],
                    academic=[0.6, 0.8, 0.0, 0.0],
                    practical=[0.6, 0.8, 0.0, 0.0],
                ),
            ),
            _position(
                "p_no_skills",
                required_skills=[],
                preferred_skills=[],
                vectors=_vec(
                    skills=[0.0, 0.0, 0.0, 0.0],
                    academic=[0.6, 0.8, 0.0, 0.0],
                    practical=[0.6, 0.8, 0.0, 0.0],
                ),
            ),
        ]

        request = AllocationSolveRequest(
            candidates=[c_partial, c_no_skills],
            positions=positions,
            min_match_threshold=0.45,
        )
        eligibility, scoring, graph = _run_pipeline(request)

        assert eligibility.eligibility_matrix.tolist() == [
            [True, True, True],
            [True, True, True],
        ]
        # c_partial clears 0.45 across all three position types.
        assert scoring.total_scores[0, 0] == pytest.approx(0.52)
        assert scoring.total_scores[0, 1] == pytest.approx(0.52 / 0.90)
        assert scoring.total_scores[0, 2] == pytest.approx(0.60)
        assert [e.position_id for e in graph.edges_by_candidate[0]] == [
            "p_no_skills",
            "p_no_pref",
            "p_full",
        ]

        # c_no_skills is NOT renormalized for missing candidate skills:
        # p_full: (0.15*0.6 + 0.20*0.6) / 1.0 = 0.21 < 0.45
        # p_no_pref: 0.21 / 0.90 = 0.2333 < 0.45
        # p_no_skills: 0.21 / 0.35 = 0.60 >= 0.45 (position has no skills requirement)
        assert scoring.total_scores[1, 0] == pytest.approx(0.21)
        assert scoring.total_scores[1, 1] == pytest.approx(0.21 / 0.90)
        assert scoring.total_scores[1, 2] == pytest.approx(0.60)
        assert [e.position_id for e in graph.edges_by_candidate[1]] == ["p_no_skills"]


class TestLockedOverridesPipelineIntegration:
    def test_locked_overrides_force_retains_all_lock_categories_with_honest_flags(self):
        # Distinct positions so c2_outside_topk gets descending scores:
        # p0=1.00, p1=0.89, p2=0.62, p3=0.475
        positions = [
            _position(
                "p0",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                allowed_cities=["Damascus"],
                min_gpa=70.0,
                vectors=_vec(
                    skills=[1.0, 0.0, 0.0, 0.0],
                    academic=[1.0, 0.0, 0.0, 0.0],
                    practical=[1.0, 0.0, 0.0, 0.0],
                ),
            ),
            _position(
                "p1",
                required_skills=["Python"],
                preferred_skills=["SQL"],
                allowed_cities=["Damascus", "Aleppo"],
                min_gpa=70.0,
                vectors=_vec(
                    skills=[0.8, 0.6, 0.0, 0.0],
                    academic=[1.0, 0.0, 0.0, 0.0],
                    practical=[0.8, 0.6, 0.0, 0.0],
                ),
            ),
            _position(
                "p2",
                required_skills=["Python"],
                preferred_skills=["Go"],
                allowed_cities=["Damascus"],
                min_gpa=70.0,
                vectors=_vec(
                    skills=[0.6, 0.8, 0.0, 0.0],
                    academic=[0.6, 0.8, 0.0, 0.0],
                    practical=[0.6, 0.8, 0.0, 0.0],
                ),
            ),
            _position(
                "p3",
                required_skills=["Python"],
                preferred_skills=["Go"],
                allowed_cities=["Damascus"],
                min_gpa=70.0,
                must_have_keywords=["Python"],
                vectors=_vec(
                    skills=[0.5, math.sqrt(0.75), 0.0, 0.0],
                    academic=[0.0, 1.0, 0.0, 0.0],
                    practical=[0.5, math.sqrt(0.75), 0.0, 0.0],
                ),
            ),
        ]

        # c0: All ineligible (city="Tartous"), high score on p0 (1.0), locked to p0.
        c0_inelig = _candidate(
            "c0_inelig",
            city="Tartous",
            gpa=90.0,
            technical_skills=["Python", "SQL"],
            vectors=_vec(),
        )
        # c1: Eligible ONLY on p1 (city="Aleppo"), score on p1 = 0.15 < 0.45, locked to p1.
        c1_below = _candidate(
            "c1_below",
            city="Aleppo",
            gpa=90.0,
            technical_skills=["Rust"],
            vectors=_vec(
                skills=[0.0, 0.0, 1.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[0.0, 0.0, 1.0, 0.0],
            ),
        )
        # c2: Eligible & >= 0.45 on all 4 positions; top-2 = [p0, p1]; locked to outside-top-K p3.
        c2_outside_topk = _candidate(
            "c2_outside_topk",
            city="Damascus",
            gpa=90.0,
            technical_skills=["Python", "SQL"],
            vectors=_vec(),
        )
        # c3: Ineligible on p3 (no "Python"), eligible & >= 0.45 on p0, p1, p2; locked to top-1 p0.
        c3_already_topk = _candidate(
            "c3_already_topk",
            city="Damascus",
            gpa=90.0,
            technical_skills=["SQL"],
            vectors=_vec(),
        )
        # c4: Eligible on p1, p2 (score >= 0.45), ineligible on p0 (GPA < 80? Wait: p0 min_gpa=70),
        # so make c4 fail p0? Let's keep p0..p3 as above and verify c0..c3 cleanly,
        # plus add c4 with city="Aleppo" & high score on p1, failing p0/p2/p3, locked to ineligible p0.
        c4_inelig_lock_with_real_elig = _candidate(
            "c4_inelig_with_elig",
            city="Aleppo",
            gpa=90.0,
            technical_skills=["Python", "SQL"],
            vectors=_vec(),
        )

        request = AllocationSolveRequest(
            candidates=[
                c0_inelig,
                c1_below,
                c2_outside_topk,
                c3_already_topk,
                c4_inelig_lock_with_real_elig,
            ],
            positions=positions,
            min_match_threshold=0.45,
            top_k_per_candidate=2,
            locked_overrides=[
                LockedOverride(candidate_id="c0_inelig", position_id="p0"),
                LockedOverride(candidate_id="c1_below", position_id="p1"),
                LockedOverride(candidate_id="c2_outside_topk", position_id="p3"),
                LockedOverride(candidate_id="c3_already_topk", position_id="p0"),
                LockedOverride(candidate_id="c4_inelig_with_elig", position_id="p0"),
            ],
        )

        eligibility, scoring, graph = _run_pipeline(request)

        assert eligibility.eligibility_matrix.tolist() == [
            [False, False, False, False],
            [False, True, False, False],
            [True, True, True, True],
            [True, True, True, False],
            [False, True, False, False],
        ]

        # Honest diagnostic flags reflect true eligibility/threshold status, never faked by locks.
        assert graph.has_eligible_position == (False, True, True, True, True)
        assert graph.has_above_threshold_position == (False, False, True, True, True)

        # c0: ineligible locked edge retained with is_locked=True, no eligible alternatives.
        assert [(e.position_id, e.is_locked) for e in graph.edges_by_candidate[0]] == [
            ("p0", True)
        ]
        assert graph.edges_by_candidate[0][0].scaled_weight == SCORE_SCALE_FACTOR
        assert scoring.top_eligible_alternatives(0, eligibility.eligibility_matrix) == []

        # c1: below-threshold locked edge retained with is_locked=True.
        assert [(e.position_id, e.is_locked) for e in graph.edges_by_candidate[1]] == [
            ("p1", True)
        ]
        assert graph.edges_by_candidate[1][0].score == pytest.approx(0.15)
        assert graph.edges_by_candidate[1][0].scaled_weight == 1500
        assert [
            alt.position_id
            for alt in scoring.top_eligible_alternatives(1, eligibility.eligibility_matrix)
        ] == ["p1"]

        # c2: real top-2 [p0, p1] preserved AND outside-top-K p3 force-retained with is_locked=True.
        assert scoring.total_scores[2, 3] == pytest.approx(0.475)
        assert [(e.position_id, e.is_locked) for e in graph.edges_by_candidate[2]] == [
            ("p0", False),
            ("p1", False),
            ("p3", True),
        ]
        assert [
            alt.position_id
            for alt in scoring.top_eligible_alternatives(
                2, eligibility.eligibility_matrix, exclude_position_idx=3, limit=2
            )
        ] == ["p0", "p1"]

        # c3: p0 already in top-2 -> not duplicated, flagged is_locked=True, p1 retained.
        assert [(e.position_id, e.is_locked) for e in graph.edges_by_candidate[3]] == [
            ("p0", True),
            ("p1", False),
        ]
        assert [
            alt.position_id
            for alt in scoring.top_eligible_alternatives(
                3, eligibility.eligibility_matrix, exclude_position_idx=0, limit=2
            )
        ] == ["p1", "p2"]

        # c4: ineligible lock p0 force-retained alongside real eligible above-threshold p1.
        assert [(e.position_id, e.is_locked) for e in graph.edges_by_candidate[4]] == [
            ("p0", True),
            ("p1", False),
        ]
        assert [
            alt.position_id
            for alt in scoring.top_eligible_alternatives(
                4, eligibility.eligibility_matrix, exclude_position_idx=0, limit=2
            )
        ] == ["p1"]


class TestAllocationSolveRequestIntegration:
    def test_request_fields_control_eligibility_scoring_pruning_and_assignment_contract(self):
        # Construct realistic candidate and positions via Pydantic payload validation.
        cand_a = _candidate(
            "c_lexical_strong",
            city="Damascus",
            gpa=86.0,
            major="Computer Science",
            technical_skills=["Python", "SQL", "Docker"],
            vectors=_vec(
                skills=[0.4, math.sqrt(0.84), 0.0, 0.0],
                academic=[0.4, math.sqrt(0.84), 0.0, 0.0],
                practical=[0.4, math.sqrt(0.84), 0.0, 0.0],
            ),
        )
        cand_b = _candidate(
            "c_semantic_strong",
            city="Damascus",
            gpa=88.0,
            major="Computer Science",
            technical_skills=[],
            vectors=_vec(
                skills=[1.0, 0.0, 0.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[1.0, 0.0, 0.0, 0.0],
            ),
        )

        pos_0 = _position(
            "p0",
            required_skills=["Python", "SQL"],
            preferred_skills=["Docker"],
            allowed_cities=["Damascus"],
            vectors=_vec(
                skills=[1.0, 0.0, 0.0, 0.0],
                academic=[1.0, 0.0, 0.0, 0.0],
                practical=[1.0, 0.0, 0.0, 0.0],
            ),
        )
        pos_1 = _position(
            "p1",
            required_skills=["Python"],
            preferred_skills=["Docker"],
            allowed_cities=["Damascus"],
            vectors=_vec(
                skills=[0.8, 0.6, 0.0, 0.0],
                academic=[0.8, 0.6, 0.0, 0.0],
                practical=[0.8, 0.6, 0.0, 0.0],
            ),
        )
        pos_2 = _position(
            "p2",
            required_skills=["Python"],
            preferred_skills=["Go"],
            allowed_cities=["Damascus"],
            vectors=_vec(
                skills=[0.7, math.sqrt(0.51), 0.0, 0.0],
                academic=[0.7, math.sqrt(0.51), 0.0, 0.0],
                practical=[0.7, math.sqrt(0.51), 0.0, 0.0],
            ),
        )

        # Request 1: lexical-heavy weights, threshold=0.60, top_k=2, locked override for cand_b -> p2.
        req_lexical = AllocationSolveRequest.model_validate(
            {
                "candidates": [cand_a.model_dump(), cand_b.model_dump()],
                "positions": [pos_0.model_dump(), pos_1.model_dump(), pos_2.model_dump()],
                "weights": {
                    "skills_semantic": 0.10,
                    "academic_semantic": 0.10,
                    "practical_semantic": 0.10,
                    "required_skills_lexical": 0.50,
                    "preferred_skills_lexical": 0.20,
                },
                "min_match_threshold": 0.60,
                "top_k_per_candidate": 2,
                "locked_overrides": [{"candidate_id": "c_semantic_strong", "position_id": "p2"}],
            }
        )

        elig_1, score_1, graph_1 = _run_pipeline(req_lexical)

        # c_lexical_strong: p1 (~0.961) > p0 (0.820) > p2 (~0.780); all >= 0.60, capped to top_k=2.
        assert [(e.position_id, e.is_locked) for e in graph_1.edges_by_candidate[0]] == [
            ("p1", False),
            ("p0", False),
        ]
        assert all(edge.score >= 0.60 for edge in graph_1.edges_by_candidate[0])

        # c_semantic_strong has 0.0 lexical -> max possible score = 0.30 < 0.60,
        # so only locked p2 survives with is_locked=True and honest threshold flag False.
        assert float(score_1.total_scores[1].max()) == pytest.approx(0.30)
        assert graph_1.has_eligible_position[1] is True
        assert graph_1.has_above_threshold_position[1] is False
        assert [(e.position_id, e.is_locked) for e in graph_1.edges_by_candidate[1]] == [
            ("p2", True)
        ]

        # Pipeline outputs assemble cleanly into Phase 6 AssignmentResult contract.
        top_edge = graph_1.edges_by_candidate[0][0]
        assignment = AssignmentResult(
            candidate_id=top_edge.candidate_id,
            position_id=top_edge.position_id,
            score=top_edge.score,
            score_breakdown=top_edge.breakdown,
            alternatives=score_1.top_eligible_alternatives(
                top_edge.candidate_idx,
                elig_1.eligibility_matrix,
                exclude_position_idx=top_edge.position_idx,
                limit=2,
            ),
            is_locked=top_edge.is_locked,
        )
        assert len(assignment.alternatives) == 2

        # Request 2: semantic-only weights, threshold=0.75, top_k=1, no locks.
        req_semantic = AllocationSolveRequest(
            candidates=[cand_a, cand_b],
            positions=[pos_0, pos_1, pos_2],
            weights=ScoringWeights(
                skills_semantic=0.50,
                academic_semantic=0.25,
                practical_semantic=0.25,
                required_skills_lexical=0.0,
                preferred_skills_lexical=0.0,
            ),
            min_match_threshold=0.75,
            top_k_per_candidate=1,
            locked_overrides=[],
        )

        _, score_2, graph_2 = _run_pipeline(req_semantic)

        # c_lexical_strong: p0=0.40, p1=0.8699, p2=0.9355 -> top_k=1 retains only p2.
        assert [(e.position_id, e.is_locked) for e in graph_2.edges_by_candidate[0]] == [
            ("p2", False)
        ]
        # c_semantic_strong: p0=1.00, p1=0.80, p2=0.70 -> top_k=1 retains only p0.
        assert score_2.total_scores[1, 0] == pytest.approx(1.0)
        assert [(e.position_id, e.is_locked) for e in graph_2.edges_by_candidate[1]] == [
            ("p0", False)
        ]

    def test_solve_request_invalid_or_conflicting_locked_overrides_rejected(self):
        cand = _candidate("c1")
        pos_0 = _position("p0")
        pos_1 = _position("p1")

        unknown_cand_req = AllocationSolveRequest(
            candidates=[cand],
            positions=[pos_0, pos_1],
            locked_overrides=[LockedOverride(candidate_id="missing", position_id="p0")],
        )
        with pytest.raises(ValueError, match="Unknown candidate_id"):
            _run_pipeline(unknown_cand_req)

        unknown_pos_req = AllocationSolveRequest(
            candidates=[cand],
            positions=[pos_0, pos_1],
            locked_overrides=[LockedOverride(candidate_id="c1", position_id="missing")],
        )
        with pytest.raises(ValueError, match="Unknown position_id"):
            _run_pipeline(unknown_pos_req)

        conflicting_req = AllocationSolveRequest(
            candidates=[cand],
            positions=[pos_0, pos_1],
            locked_overrides=[
                LockedOverride(candidate_id="c1", position_id="p0"),
                LockedOverride(candidate_id="c1", position_id="p1"),
            ],
        )
        with pytest.raises(ValueError, match="Conflicting locked_overrides"):
            _run_pipeline(conflicting_req)

