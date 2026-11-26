from __future__ import annotations

import numpy as np
import pytest

from app.schemas.candidate import CandidateForAllocation, CandidateMetadata, ParsedCV
from app.schemas.position import (
    FacetedVectors,
    PositionConstraints,
    PositionForAllocation,
)
from app.services.constraints import (
    DisqualificationReason,
    EligibilityResult,
    evaluate_eligibility,
)


def _vec() -> FacetedVectors:
    return FacetedVectors(skills=[1.0], academic=[1.0], practical=[1.0])


def _candidate(
    candidate_id: str = "c1",
    *,
    cv: dict | None = None,
    **meta_overrides,
) -> CandidateForAllocation:
    metadata = CandidateMetadata(candidate_id=candidate_id, **meta_overrides)
    parsed_cv = ParsedCV(**(cv or {}))
    return CandidateForAllocation(
        candidate_id=candidate_id,
        metadata=metadata,
        parsed_cv=parsed_cv,
        vectors=_vec(),
    )


def _position(
    position_id: str = "p1",
    **constraint_overrides,
) -> PositionForAllocation:
    return PositionForAllocation(
        position_id=position_id,
        title="Software Engineering Intern",
        max_seats=2,
        constraints=PositionConstraints(**constraint_overrides),
        vectors=_vec(),
    )


class TestAllowedCitiesRule:
    def test_empty_or_blank_allowed_cities_allows_missing_city(self):
        cand = _candidate(city=None)
        pos_empty = _position("p1", allowed_cities=[])
        pos_blanks = _position("p2", allowed_cities=["", "   ", "\t"])
        result = evaluate_eligibility([cand], [pos_empty, pos_blanks])
        assert result.eligibility_matrix.tolist() == [[True, True]]

    def test_missing_or_blank_candidate_city_disqualified(self):
        c_none = _candidate("c1", city=None)
        c_blank = _candidate("c2", city="   ")
        pos = _position("p1", allowed_cities=["Damascus", "   "])
        result = evaluate_eligibility([c_none, c_blank], [pos])
        assert result.eligibility_matrix.tolist() == [[False], [False]]
        assert result.disqualifications[(0, 0)][0].rule == "allowed_cities"
        assert "missing" in result.reasons_for_pair(0, 0)[0].lower()
        assert result.disqualifications[(1, 0)][0].rule == "allowed_cities"

    def test_city_matches_case_stripped_and_arabic_normalized(self):
        c_en = _candidate("c1", city="  dAmAsCuS ")
        c_ar_diacritics = _candidate("c2", city="الإسْكَنْدَرِيَّة")
        c_ar_tatweel = _candidate("c3", city="دِمَـشْق")
        c_mismatch = _candidate("c4", city="Damas")
        pos = _position(
            "p1",
            allowed_cities=["  Damascus ", "الاسكندرية", "دمشق"],
        )
        result = evaluate_eligibility(
            [c_en, c_ar_diacritics, c_ar_tatweel, c_mismatch],
            [pos],
        )
        assert result.eligibility_matrix.tolist() == [[True], [True], [True], [False]]
        assert result.disqualifications[(3, 0)][0].rule == "allowed_cities"


class TestMinGpaRule:
    def test_none_min_gpa_allows_missing_gpa(self):
        result = evaluate_eligibility([_candidate(gpa=None)], [_position(min_gpa=None)])
        assert bool(result.eligibility_matrix[0, 0]) is True

    def test_zero_min_gpa_still_requires_present_gpa(self):
        c_missing = _candidate("c1", gpa=None)
        c_zero = _candidate("c2", gpa=0.0)
        pos = _position(min_gpa=0.0)
        result = evaluate_eligibility([c_missing, c_zero], [pos])
        assert result.eligibility_matrix.tolist() == [[False], [True]]
        assert result.disqualifications[(0, 0)][0].rule == "min_gpa"

    def test_gpa_boundary_equality_and_below(self):
        c_below = _candidate("c1", gpa=74.99)
        c_exact = _candidate("c2", gpa=75.0)
        c_above = _candidate("c3", gpa=88.5)
        pos = _position(min_gpa=75.0)
        result = evaluate_eligibility([c_below, c_exact, c_above], [pos])
        assert result.eligibility_matrix.tolist() == [[False], [True], [True]]
        assert result.disqualifications[(0, 0)][0].rule == "min_gpa"


class TestMinYearOfStudyRule:
    def test_none_min_year_allows_missing_year(self):
        result = evaluate_eligibility(
            [_candidate(year_of_study=None)],
            [_position(min_year_of_study=None)],
        )
        assert bool(result.eligibility_matrix[0, 0]) is True

    def test_missing_below_equal_and_above_year_of_study(self):
        c_missing = _candidate("c1", year_of_study=None)
        c_below = _candidate("c2", year_of_study=2)
        c_exact = _candidate("c3", year_of_study=3)
        c_above = _candidate("c4", year_of_study=5)
        pos = _position(min_year_of_study=3)
        result = evaluate_eligibility(
            [c_missing, c_below, c_exact, c_above],
            [pos],
        )
        assert result.eligibility_matrix.tolist() == [
            [False],
            [False],
            [True],
            [True],
        ]
        assert result.disqualifications[(0, 0)][0].rule == "min_year_of_study"
        assert result.disqualifications[(1, 0)][0].rule == "min_year_of_study"


class TestFullTimeCommitmentRule:
    def test_full_time_commitment_requirement(self):
        c_part_time = _candidate("c1", full_time_commitment=False)
        c_full_time = _candidate("c2", full_time_commitment=True)
        p_flexible = _position("p1", requires_full_time_commitment=False)
        p_strict = _position("p2", requires_full_time_commitment=True)
        result = evaluate_eligibility([c_part_time, c_full_time], [p_flexible, p_strict])
        assert result.eligibility_matrix.tolist() == [
            [True, False],
            [True, True],
        ]
        assert (
            result.disqualifications[(0, 1)][0].rule
            == "requires_full_time_commitment"
        )


class TestRequiredMajorsRule:
    def test_empty_or_blank_required_majors_allows_blank_major_and_field(self):
        cand = _candidate(major="  ", field="")
        pos = _position(required_majors=["", "   "])
        result = evaluate_eligibility([cand], [pos])
        assert bool(result.eligibility_matrix[0, 0]) is True

    def test_exact_and_whole_phrase_match_with_field_fallback_and_arabic(self):
        c_exact = _candidate("c1", major="Computer Science")
        c_phrase = _candidate("c2", major="BSc in Computer Science (Honours)")
        c_fallback = _candidate("c3", major="   ", field="B.Sc. Computer Science")
        c_nonblank_major_no_fallback = _candidate(
            "c4",
            major="Mechanical Engineering",
            field="Computer Science",
        )
        c_missing = _candidate("c5", major="", field="   ")
        c_ar = _candidate("c6", major="بكالوريوس في هَنْدَسَة الحاسوب")

        pos = _position(
            required_majors=["Computer Science", "هندسة الحاسوب", "   "]
        )
        result = evaluate_eligibility(
            [c_exact, c_phrase, c_fallback, c_nonblank_major_no_fallback, c_missing, c_ar],
            [pos],
        )
        assert result.eligibility_matrix.tolist() == [
            [True],
            [True],
            [True],
            [False],
            [False],
            [True],
        ]
        assert result.disqualifications[(3, 0)][0].rule == "required_majors"
        assert result.disqualifications[(4, 0)][0].rule == "required_majors"

    def test_prevents_false_substring_leaks_in_majors(self):
        c_partial = _candidate("c1", major="Computer")
        c_truncated = _candidate("c2", major="Computer Sci")
        c_hospitality = _candidate("c3", major="Hospitality")
        c_literature = _candidate("c4", major="English Literature")
        pos_cs = _position("p1", required_majors=["Computer Science"])
        pos_it = _position("p2", required_majors=["IT"])

        result = evaluate_eligibility(
            [c_partial, c_truncated, c_hospitality, c_literature],
            [pos_cs, pos_it],
        )
        assert result.eligibility_matrix.tolist() == [
            [False, False],
            [False, False],
            [False, False],
            [False, False],
        ]


class TestMustHaveKeywordsRule:
    def test_empty_or_blank_keywords_always_passes(self):
        cand = _candidate()
        pos = _position(must_have_keywords=["", "  "])
        result = evaluate_eligibility([cand], [pos])
        assert bool(result.eligibility_matrix[0, 0]) is True

    def test_searches_all_11_corpus_fields_and_ignores_city(self):
        keywords = [
            "Python",
            "Telecom",
            "Engineering",
            "Software",
            "Available",
            "FastAPI",
            "Kubernetes",
            "Damascus University",
            "Recommender",
            "CKA",
            "Arabic",
        ]
        cand = _candidate(
            "c1",
            city="SpecialCityOnly",
            technical_skills=["Python"],
            program="Telecom",
            field="Engineering",
            major="Software",
            additional_info="Available immediately",
            cv={
                "skills": ["FastAPI"],
                "experience_summary": "Deployed services on Kubernetes.",
                "education_summary": "Graduated from Damascus University",
                "projects_summary": "Built a Recommender engine",
                "certifications": ["CKA"],
                "languages": ["Arabic"],
            },
        )
        pos_all_11 = _position("p1", must_have_keywords=keywords)
        pos_city_not_corpus = _position("p2", must_have_keywords=["SpecialCityOnly"])

        result = evaluate_eligibility([cand], [pos_all_11, pos_city_not_corpus])
        assert result.eligibility_matrix.tolist() == [[True, False]]
        assert "SpecialCityOnly" in result.reasons_for_pair(0, 1)[0]

    def test_requires_every_keyword_and_reports_all_missing(self):
        cand = _candidate("c1", technical_skills=["Python"])
        pos = _position("p1", must_have_keywords=["Python", "Docker", "SQL", "docker"])
        result = evaluate_eligibility([cand], [pos])
        assert bool(result.eligibility_matrix[0, 0]) is False
        reason = result.disqualifications[(0, 0)][0]
        assert reason.rule == "must_have_keywords"
        assert "Docker" in reason.message
        assert "SQL" in reason.message
        assert "Python" not in reason.message

    def test_boundary_safe_tech_symbols_english_and_arabic(self):
        cand_tech = _candidate(
            "c1",
            technical_skills=["C++", "C#", ".NET", "Node.js", "ASP.NET"],
            cv={
                "skills": ["C/C++", "بايثُون"],
                "experience_summary": "Built REST APIs in Python. Used R, SQL, and الإحصاء.",
            },
        )
        pos_pass = _position(
            "p1",
            must_have_keywords=[
                "C++",
                "C#",
                ".NET",
                "Node.js",
                "ASP.NET",
                "C",
                "Python",
                "R",
                "بايثون",
                "الاحصاء",
            ],
        )
        result = evaluate_eligibility([cand_tech], [pos_pass])
        assert bool(result.eligibility_matrix[0, 0]) is True

    def test_prevents_false_substring_matches_for_keywords(self):
        cand = _candidate(
            "c1",
            technical_skills=["JavaScript", "C++", "C#", "Objective-C", "Docker"],
            cv={
                "skills": ["Node.js", "ASP.NET"],
                "experience_summary": "Worked in R&D on Computer Vision.",
            },
        )
        forbid = ["Java", "C", "R", "Node", "js", ".NET"]
        positions = [_position(f"p_{idx}", must_have_keywords=[kw]) for idx, kw in enumerate(forbid)]
        result = evaluate_eligibility([cand], positions)
        assert result.eligibility_matrix.tolist() == [[False] * len(forbid)]

    def test_sentence_ending_periods_still_match_keywords(self):
        cand = _candidate(
            "c1",
            cv={
                "experience_summary": "Developed backend services using Node.js, C++, and .NET."
            },
        )
        pos = _position("p1", must_have_keywords=["Node.js", "C++", ".NET"])
        result = evaluate_eligibility([cand], [pos])
        assert bool(result.eligibility_matrix[0, 0]) is True


class TestEligibilityResultContractAndValidation:
    def test_matrix_shape_dtype_and_multi_rule_disqualifications(self):
        c1 = _candidate(
            "cand-1",
            city="Homs",
            gpa=65.0,
            year_of_study=2,
            full_time_commitment=False,
            major="Art",
            technical_skills=["HTML"],
        )
        c2 = _candidate(
            "cand-2",
            city="Damascus",
            gpa=85.0,
            year_of_study=4,
            full_time_commitment=True,
            major="BSc in Computer Science",
            technical_skills=["Python", "SQL"],
        )
        p1 = _position(
            "pos-1",
            allowed_cities=["Damascus"],
            min_gpa=75.0,
            min_year_of_study=3,
            requires_full_time_commitment=True,
            required_majors=["Computer Science"],
            must_have_keywords=["Python"],
        )
        p2 = _position("pos-2")

        result = evaluate_eligibility([c1, c2], [p1, p2])
        assert isinstance(result, EligibilityResult)
        assert result.eligibility_matrix.shape == (2, 2)
        assert result.eligibility_matrix.dtype == np.dtype(bool)
        assert result.eligibility_matrix.tolist() == [
            [False, True],
            [True, True],
        ]
        assert result.candidate_ids == ("cand-1", "cand-2")
        assert result.position_ids == ("pos-1", "pos-2")

        pair_reasons = result.disqualifications[(0, 0)]
        assert all(isinstance(r, DisqualificationReason) for r in pair_reasons)
        assert [r.rule for r in pair_reasons] == [
            "allowed_cities",
            "min_gpa",
            "min_year_of_study",
            "requires_full_time_commitment",
            "required_majors",
            "must_have_keywords",
        ]
        assert len(result.reasons_for_pair(0, 0)) == 6
        assert result.reasons_for_pair(0, 1) == []
        assert result.reasons_for_pair(1, 0) == []

        assert result.disqualification_summary_for_candidate(0) == {
            "pos-1": result.reasons_for_pair(0, 0)
        }
        assert result.disqualification_summary_for_candidate(1) == {}

    def test_empty_inputs_produce_well_typed_empty_matrices(self):
        res_no_cands = evaluate_eligibility([], [_position("p1"), _position("p2")])
        assert res_no_cands.eligibility_matrix.shape == (0, 2)
        assert res_no_cands.eligibility_matrix.dtype == bool
        assert res_no_cands.candidate_ids == ()
        assert res_no_cands.position_ids == ("p1", "p2")

        res_no_pos = evaluate_eligibility([_candidate("c1")], [])
        assert res_no_pos.eligibility_matrix.shape == (1, 0)
        assert res_no_pos.eligibility_matrix.dtype == bool
        assert res_no_pos.disqualification_summary_for_candidate(0) == {}

    def test_out_of_bounds_index_raises_index_error(self):
        result = evaluate_eligibility([_candidate("c1")], [_position("p1")])
        with pytest.raises(IndexError):
            result.reasons_for_pair(1, 0)
        with pytest.raises(IndexError):
            result.reasons_for_pair(0, -1)
        with pytest.raises(IndexError):
            result.disqualification_summary_for_candidate(-1)

    def test_duplicate_ids_raise_value_error(self):
        with pytest.raises(ValueError, match="candidate_id"):
            evaluate_eligibility([_candidate("c1"), _candidate("c1")], [_position("p1")])

        with pytest.raises(ValueError, match="position_id"):
            evaluate_eligibility([_candidate("c1")], [_position("p1"), _position("p1")])
