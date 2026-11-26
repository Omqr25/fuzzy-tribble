from __future__ import annotations

import unicodedata

import numpy as np
import pytest

from app.schemas.candidate import CandidateForAllocation, CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors
from app.services.text_matching import (
    build_candidate_corpus_segments,
    compile_phrase_pattern,
    contains_phrase,
    dedupe_normalized_items,
    normalize_arabic,
    normalize_text,
    validate_index,
)


def _make_candidate(
    candidate_id: str = "c1",
    *,
    city: str | None = None,
    program: str = "",
    field: str = "",
    major: str = "",
    technical_skills: list[str] | None = None,
    additional_info: str = "",
    skills: list[str] | None = None,
    experience_summary: str = "",
    education_summary: str = "",
    projects_summary: str = "",
    certifications: list[str] | None = None,
    languages: list[str] | None = None,
) -> CandidateForAllocation:
    return CandidateForAllocation(
        candidate_id=candidate_id,
        metadata=CandidateMetadata(
            candidate_id=candidate_id,
            city=city,
            program=program,
            field=field,
            major=major,
            technical_skills=technical_skills or [],
            additional_info=additional_info,
        ),
        parsed_cv=ParsedCV(
            skills=skills or [],
            experience_summary=experience_summary,
            education_summary=education_summary,
            projects_summary=projects_summary,
            certifications=certifications or [],
            languages=languages or [],
        ),
        vectors=FacetedVectors(skills=[1.0], academic=[1.0], practical=[1.0]),
    )


class TestNormalizeArabic:
    def test_strips_diacritics_superscript_alef_and_tatweel(self):
        # Includes fathatan..sukun/hamza marks (\u064b-\u065f), superscript alef (\u0670), and tatweel (\u0640)
        assert normalize_arabic("دِمَـشْق") == "دمشق"
        assert normalize_arabic("بـايثُون") == "بايثون"
        assert normalize_arabic("مُحَمَّد\u0670") == "محمد"

    def test_unifies_all_alef_forms(self):
        assert normalize_arabic("أإآٱ") == "اااا"
        assert normalize_arabic("الإسْكَنْدَرِيَّة") == "الاسكندريه"
        assert normalize_arabic("آفاق أمنية وإدارة ٱقتصاد") == "افاق امنيه واداره اقتصاد"

    def test_unifies_teh_marbuta_and_alef_maksura_for_syrian_variants(self):
        assert normalize_arabic("اللاذقية") == normalize_arabic("اللاذقيه") == "اللاذقيه"
        assert normalize_arabic("حماة") == normalize_arabic("حماه") == "حماه"
        assert normalize_arabic("هندسة") == normalize_arabic("هندسه") == "هندسه"
        assert normalize_arabic("تقنى") == normalize_arabic("تقني") == "تقني"

    def test_preserves_non_arabic_text(self):
        assert normalize_arabic("Python C++ .NET 123") == "Python C++ .NET 123"


class TestNormalizeText:
    def test_handles_none_empty_and_whitespace_only(self):
        assert normalize_text(None) == ""
        assert normalize_text("") == ""
        assert normalize_text("   \t\r\n  ") == ""
        assert normalize_text("\u0640\u064b\u0651") == ""

    def test_nfc_and_nfd_produce_identical_normalized_output(self):
        arabic_nfd = unicodedata.normalize("NFD", "مؤتمر الإسكندرية")
        arabic_nfc = unicodedata.normalize("NFC", "مؤتمر الإسكندرية")
        assert arabic_nfd != arabic_nfc
        assert normalize_text(arabic_nfd) == normalize_text(arabic_nfc) == "مؤتمر الاسكندريه"

        latin_nfd = unicodedata.normalize("NFD", "  Café   Naïve  ")
        latin_nfc = unicodedata.normalize("NFC", "Café Naïve")
        assert latin_nfd != latin_nfc
        assert normalize_text(latin_nfd) == normalize_text(latin_nfc) == "café naïve"

    def test_collapses_whitespace_strips_and_casefolds(self):
        assert normalize_text("  Node.js \t  & \n  ASP.NET  ") == "node.js & asp.net"
        assert normalize_text("Straße") == "strasse"
        assert normalize_text("  هَنْدَسَةُ   المَعْلُومَاتِيَّةِ  ") == "هندسه المعلوماتيه"


class TestDedupeNormalizedItems:
    def test_preserves_first_seen_order_collapses_display_and_ignores_blanks(self):
        items = [
            "",
            "   \t\n ",
            "\u0640\u064b",
            "  Python  ",
            "python",
            "  Node \n  JS ",
            "node js",
            "اللاذقية",
            "اللاذقيه",
            "أمن   المعلومات",
            "امن المعلومات",
        ]
        result = dedupe_normalized_items(items)
        assert result == (
            ("Python", "python"),
            ("Node JS", "node js"),
            ("اللاذقية", "اللاذقيه"),
            ("أمن المعلومات", "امن المعلومات"),
        )

    def test_dedupes_nfd_against_nfc_preserving_first_label(self):
        nfd = unicodedata.normalize("NFD", "Café")
        nfc = unicodedata.normalize("NFC", "café")
        assert dedupe_normalized_items([nfd, nfc]) == ((nfd, "café"),)
        assert dedupe_normalized_items([]) == ()


class TestPhrasePatternMatching:
    @pytest.mark.parametrize(
        ("phrase", "text", "expected"),
        [
            # Empty guards
            ("", "python", False),
            ("python", "", False),
            # Exact and boundary matches for tech symbols
            ("c++", "c++", True),
            ("c++", "experienced in c++ and python.", True),
            ("c#", "built services with c#/.net.", True),
            (".net", "built services with c#/.net.", True),
            (".net", "used .net.", True),
            ("node.js", "developed apis using node.js, c++, and .net.", True),
            ("asp.net", "maintained asp.net core services", True),
            ("c", "proficient in c/c++ and sql.", True),
            ("r", "used r, sql, and python.", True),
            # False substring traps
            ("java", "javascript and typescript", False),
            ("c", "computer science", False),
            ("c", "c++ and c#", False),
            ("c", "objective-c", False),
            ("r", "docker containers", False),
            ("r", "worked in r&d", False),
            ("d", "worked in r&d", False),
            ("react", "react-native mobile app", False),
            ("native", "react-native mobile app", False),
            ("sql", "t-sql queries", False),
            ("t", "t-sql queries", False),
            (".net", "asp.net web forms", False),
            ("node", "node.js backend", False),
            ("js", "node.js backend", False),
            ("io", "socket.io realtime", False),
            # Multi-word & compound boundary traps
            ("computer science", "bsc in computer science (honours)", True),
            ("computer science", "computer", False),
            ("computer science", "computer sci", False),
            ("objective-c", "experienced in objective-c and swift", True),
            ("r&d", "conducted r&d in telecom", True),
            ("react-native", "built react-native apps", True),
            ("t-sql", "wrote t-sql procedures", True),
            # Arabic phrases and boundaries
            ("هندسه الحاسوب", "بكالوريوس في هندسه الحاسوب", True),
            ("بايثون", "خبره في بايثون.", True),
            ("امن المعلومات", "خبره في امن المعلومات وتحليل البيانات", True),
            ("امن", "امنه وتامين البيانات", False),
        ],
    )
    def test_boundary_aware_phrase_matching(
        self,
        phrase: str,
        text: str,
        expected: bool,
    ):
        norm_phrase = normalize_text(phrase)
        norm_text = normalize_text(text)
        pattern = compile_phrase_pattern(norm_phrase)
        assert contains_phrase(norm_text, pattern, norm_phrase) is expected


class TestBuildCandidateCorpusSegments:
    def test_extracts_all_11_fields_excludes_city_and_keeps_segments_separate(self):
        cand = _make_candidate(
            "c1",
            city="SpecialCityOnly",
            technical_skills=["Data", "  Python  ", "بايثُون"],
            program="Telecom",
            field="Engineering",
            major="هندسة البرمجيات",
            additional_info="Available immediately",
            skills=["Science", "python", "FastAPI"],
            experience_summary="Deployed services on Kubernetes.",
            education_summary="Graduated from Damascus University",
            projects_summary="Built a Recommender engine",
            certifications=["CKA", "  "],
            languages=["Arabic", "English"],
        )

        segments = build_candidate_corpus_segments(cand)

        assert segments == (
            "data",
            "python",
            "بايثون",
            "telecom",
            "engineering",
            "هندسه البرمجيات",
            "available immediately",
            "science",
            "fastapi",
            "deployed services on kubernetes.",
            "graduated from damascus university",
            "built a recommender engine",
            "cka",
            "arabic",
            "english",
        )
        assert "specialcityonly" not in segments

        # Phrase spanning separate items ("Data" + "Science") must NOT match.
        phrase = normalize_text("Data Science")
        pattern = compile_phrase_pattern(phrase)
        assert not any(contains_phrase(seg, pattern, phrase) for seg in segments)

        # Arabic variant normalized across segments ("هندسة" vs "هندسه", "بايثُون" vs "بايثون").
        for expected_kw in ("بايثون", "هندسة البرمجيات", "Damascus University", "CKA", "Arabic"):
            kw_norm = normalize_text(expected_kw)
            kw_pattern = compile_phrase_pattern(kw_norm)
            assert any(contains_phrase(seg, kw_pattern, kw_norm) for seg in segments)

    def test_handles_all_empty_fields(self):
        cand = _make_candidate("c_empty", city="Damascus")
        assert build_candidate_corpus_segments(cand) == ()


class TestValidateIndex:
    def test_accepts_python_and_numpy_integers_and_returns_native_int(self):
        assert validate_index(0, 3, "candidate") == 0
        assert validate_index(2, 3, "position") == 2

        for np_idx in (np.int64(1), np.int32(0), np.uint8(2)):
            result = validate_index(np_idx, 3, "candidate")
            assert result == int(np_idx)
            assert type(result) is int

    @pytest.mark.parametrize(
        "bad_index",
        [True, False, np.bool_(True), np.bool_(False), 0.0, np.float64(0.0), "0", None],
    )
    def test_rejects_booleans_and_non_integers(self, bad_index: object):
        with pytest.raises(IndexError, match="candidate index must be an integer"):
            validate_index(bad_index, 3, "candidate")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("idx", "size"),
        [(-1, 3), (3, 3), (4, 3), (0, 0), (-1, 0)],
    )
    def test_rejects_out_of_bounds_indices(self, idx: int, size: int):
        with pytest.raises(IndexError, match="position index .* out of bounds"):
            validate_index(idx, size, "position")
