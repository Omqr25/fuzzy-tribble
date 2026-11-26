"""Phase 5 Stage 1: Deterministic hard eligibility gate.

Evaluates every ``(candidate, position)`` pair against the six hard constraints
in ``PositionConstraints`` before scoring or CP-SAT optimization, producing a
boolean ``(N, M)`` eligibility matrix and structured disqualification reasons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import unicodedata

import numpy as np

from app.schemas.candidate import CandidateForAllocation
from app.schemas.position import PositionForAllocation
from app.services.extractor import _normalize_arabic

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class DisqualificationReason:
    """Single failed eligibility rule and its human-readable explanation."""

    rule: str
    message: str


@dataclass(frozen=True, slots=True)
class EligibilityResult:
    """Result of evaluating hard eligibility across all candidates and positions."""

    eligibility_matrix: np.ndarray
    candidate_ids: tuple[str, ...]
    position_ids: tuple[str, ...]
    disqualifications: dict[tuple[int, int], list[DisqualificationReason]]

    def _validate_candidate_index(self, candidate_idx: int) -> None:
        if not (0 <= candidate_idx < len(self.candidate_ids)):
            raise IndexError(
                f"candidate_idx {candidate_idx} out of range for {len(self.candidate_ids)} candidates"
            )

    def _validate_position_index(self, position_idx: int) -> None:
        if not (0 <= position_idx < len(self.position_ids)):
            raise IndexError(
                f"position_idx {position_idx} out of range for {len(self.position_ids)} positions"
            )

    def reasons_for_pair(self, candidate_idx: int, position_idx: int) -> list[str]:
        """Return disqualification messages for ``(candidate_idx, position_idx)``."""
        self._validate_candidate_index(candidate_idx)
        self._validate_position_index(position_idx)
        return [
            reason.message
            for reason in self.disqualifications.get((candidate_idx, position_idx), ())
        ]

    def disqualification_summary_for_candidate(
        self, candidate_idx: int
    ) -> dict[str, list[str]]:
        """Return ``{position_id: [reasons]}`` for all ineligible positions."""
        self._validate_candidate_index(candidate_idx)
        summary: dict[str, list[str]] = {}
        for pos_idx, position_id in enumerate(self.position_ids):
            if not self.eligibility_matrix[candidate_idx, pos_idx]:
                summary[position_id] = self.reasons_for_pair(candidate_idx, pos_idx)
        return summary


@dataclass(frozen=True, slots=True)
class _PreparedCandidate:
    city_norm: str
    gpa: float | None
    year_of_study: int | None
    full_time_commitment: bool
    effective_major_display: str
    effective_major_norm: str
    corpus_chunks_norm: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PreparedPosition:
    allowed_cities_display: tuple[str, ...]
    allowed_cities_norm: frozenset[str]
    min_gpa: float | None
    min_year_of_study: int | None
    requires_full_time_commitment: bool
    required_majors_display: tuple[str, ...]
    required_major_matchers: tuple[tuple[str, re.Pattern[str]], ...]
    must_have_keywords: tuple[tuple[str, str, re.Pattern[str]], ...]


def _normalize_text(text: str | None) -> str:
    """Normalize text for deterministic Arabic/English comparison."""
    if not text:
        return ""
    composed = unicodedata.normalize("NFC", text)
    arabic_normalized = _normalize_arabic(composed)
    collapsed = _WHITESPACE_RE.sub(" ", arabic_normalized).strip()
    return collapsed.casefold()


def _dedupe_normalized_items(items: Sequence[str]) -> tuple[tuple[str, str], ...]:
    """Return ``(display_label, normalized_value)`` pairs, ignoring blanks."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in items:
        display = _WHITESPACE_RE.sub(" ", (raw or "")).strip()
        norm = _normalize_text(display)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        pairs.append((display, norm))
    return tuple(pairs)


def _compile_phrase_pattern(normalized_phrase: str) -> re.Pattern[str]:
    """Compile a boundary-aware pattern supporting English, Arabic, and tech symbols.

    Prevents partial token leaks (e.g. ``Java`` inside ``JavaScript``, ``C`` inside
    ``Computer`` or ``C++``, ``R`` inside ``Docker`` or ``R&D``, ``.NET`` inside
    ``ASP.NET``) while allowing sentence-ending punctuation such as ``Python.`` and
    slash-separated tokens such as ``C/C++``.
    """
    escaped = re.escape(normalized_phrase)
    return re.compile(
        rf"(?<![\w+#.])(?<![\w+#][&-]){escaped}(?![\w+#]|[.&-][\w+#])"
    )


def _contains_phrase(
    normalized_text: str,
    pattern: re.Pattern[str],
    normalized_phrase: str,
) -> bool:
    """Return True if ``normalized_phrase`` matches exactly or on token boundaries."""
    if not normalized_text or not normalized_phrase:
        return False
    if normalized_text == normalized_phrase:
        return True
    return pattern.search(normalized_text) is not None


def _prepare_candidate(candidate: CandidateForAllocation) -> _PreparedCandidate:
    meta = candidate.metadata
    cv = candidate.parsed_cv

    major_norm = _normalize_text(meta.major)
    field_norm = _normalize_text(meta.field)
    if major_norm:
        effective_major_display = _WHITESPACE_RE.sub(" ", meta.major).strip()
        effective_major_norm = major_norm
    elif field_norm:
        effective_major_display = _WHITESPACE_RE.sub(" ", meta.field).strip()
        effective_major_norm = field_norm
    else:
        effective_major_display = ""
        effective_major_norm = ""

    raw_corpus_fields: list[str] = [
        *meta.technical_skills,
        meta.program,
        meta.field,
        meta.major,
        meta.additional_info,
        *cv.skills,
        cv.experience_summary,
        cv.education_summary,
        cv.projects_summary,
        *cv.certifications,
        *cv.languages,
    ]
    corpus_chunks_norm = tuple(
        norm for _, norm in _dedupe_normalized_items(raw_corpus_fields)
    )

    return _PreparedCandidate(
        city_norm=_normalize_text(meta.city),
        gpa=meta.gpa,
        year_of_study=meta.year_of_study,
        full_time_commitment=meta.full_time_commitment is True,
        effective_major_display=effective_major_display,
        effective_major_norm=effective_major_norm,
        corpus_chunks_norm=corpus_chunks_norm,
    )


def _prepare_position(position: PositionForAllocation) -> _PreparedPosition:
    rules = position.constraints
    city_items = _dedupe_normalized_items(rules.allowed_cities)
    major_items = _dedupe_normalized_items(rules.required_majors)
    keyword_items = _dedupe_normalized_items(rules.must_have_keywords)

    return _PreparedPosition(
        allowed_cities_display=tuple(display for display, _ in city_items),
        allowed_cities_norm=frozenset(norm for _, norm in city_items),
        min_gpa=rules.min_gpa,
        min_year_of_study=rules.min_year_of_study,
        requires_full_time_commitment=rules.requires_full_time_commitment,
        required_majors_display=tuple(display for display, _ in major_items),
        required_major_matchers=tuple(
            (norm, _compile_phrase_pattern(norm)) for _, norm in major_items
        ),
        must_have_keywords=tuple(
            (display, norm, _compile_phrase_pattern(norm))
            for display, norm in keyword_items
        ),
    )


def _check_allowed_cities(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
    raw_city: str | None,
) -> DisqualificationReason | None:
    if not pos.allowed_cities_norm:
        return None
    allowed = ", ".join(pos.allowed_cities_display)
    if not cand.city_norm:
        return DisqualificationReason(
            rule="allowed_cities",
            message=f"Candidate city is missing; allowed cities: {allowed}.",
        )
    if cand.city_norm not in pos.allowed_cities_norm:
        city_label = _WHITESPACE_RE.sub(" ", (raw_city or "")).strip()
        return DisqualificationReason(
            rule="allowed_cities",
            message=f"Candidate city '{city_label}' is not in allowed cities: {allowed}.",
        )
    return None


def _check_min_gpa(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> DisqualificationReason | None:
    if pos.min_gpa is None:
        return None
    if cand.gpa is None:
        return DisqualificationReason(
            rule="min_gpa",
            message=f"Candidate GPA is missing; minimum required is {pos.min_gpa}.",
        )
    if cand.gpa < pos.min_gpa:
        return DisqualificationReason(
            rule="min_gpa",
            message=f"Candidate GPA {cand.gpa} is below minimum required {pos.min_gpa}.",
        )
    return None


def _check_min_year_of_study(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> DisqualificationReason | None:
    if pos.min_year_of_study is None:
        return None
    if cand.year_of_study is None:
        return DisqualificationReason(
            rule="min_year_of_study",
            message=(
                "Candidate year of study is missing; minimum required is "
                f"{pos.min_year_of_study}."
            ),
        )
    if cand.year_of_study < pos.min_year_of_study:
        return DisqualificationReason(
            rule="min_year_of_study",
            message=(
                f"Candidate year of study {cand.year_of_study} is below minimum "
                f"required {pos.min_year_of_study}."
            ),
        )
    return None


def _check_full_time_commitment(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> DisqualificationReason | None:
    if not pos.requires_full_time_commitment:
        return None
    if not cand.full_time_commitment:
        return DisqualificationReason(
            rule="requires_full_time_commitment",
            message="Position requires full-time commitment.",
        )
    return None


def _check_required_majors(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> DisqualificationReason | None:
    if not pos.required_major_matchers:
        return None
    allowed = ", ".join(pos.required_majors_display)
    if not cand.effective_major_norm:
        return DisqualificationReason(
            rule="required_majors",
            message=f"Candidate major/field is missing; required majors: {allowed}.",
        )
    if any(
        _contains_phrase(cand.effective_major_norm, pattern, req_norm)
        for req_norm, pattern in pos.required_major_matchers
    ):
        return None
    return DisqualificationReason(
        rule="required_majors",
        message=(
            f"Candidate major '{cand.effective_major_display}' does not match "
            f"required majors: {allowed}."
        ),
    )


def _check_must_have_keywords(
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> DisqualificationReason | None:
    if not pos.must_have_keywords:
        return None
    missing = [
        display
        for display, norm, pattern in pos.must_have_keywords
        if not any(
            _contains_phrase(chunk, pattern, norm)
            for chunk in cand.corpus_chunks_norm
        )
    ]
    if not missing:
        return None
    return DisqualificationReason(
        rule="must_have_keywords",
        message=f"Missing required keywords: {', '.join(missing)}.",
    )


def _evaluate_pair(
    raw_candidate: CandidateForAllocation,
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
) -> list[DisqualificationReason]:
    checks = (
        _check_allowed_cities(cand, pos, raw_candidate.metadata.city),
        _check_min_gpa(cand, pos),
        _check_min_year_of_study(cand, pos),
        _check_full_time_commitment(cand, pos),
        _check_required_majors(cand, pos),
        _check_must_have_keywords(cand, pos),
    )
    return [reason for reason in checks if reason is not None]


def _validate_unique_ids(
    items: Sequence[CandidateForAllocation] | Sequence[PositionForAllocation],
    attr: str,
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        item_id = getattr(item, attr)
        if item_id in seen:
            raise ValueError(f"Duplicate {attr}: {item_id!r}")
        seen.add(item_id)
        ids.append(item_id)
    return tuple(ids)


def evaluate_eligibility(
    candidates: Sequence[CandidateForAllocation],
    positions: Sequence[PositionForAllocation],
) -> EligibilityResult:
    """Evaluate all ``(candidate, position)`` pairs against hard constraints."""
    candidate_ids = _validate_unique_ids(candidates, "candidate_id")
    position_ids = _validate_unique_ids(positions, "position_id")

    prepared_candidates = [_prepare_candidate(cand) for cand in candidates]
    prepared_positions = [_prepare_position(pos) for pos in positions]

    eligibility_matrix = np.ones(
        (len(candidates), len(positions)),
        dtype=bool,
    )
    disqualifications: dict[tuple[int, int], list[DisqualificationReason]] = {}

    for cand_idx, (raw_cand, prep_cand) in enumerate(
        zip(candidates, prepared_candidates, strict=True)
    ):
        for pos_idx, prep_pos in enumerate(prepared_positions):
            reasons = _evaluate_pair(raw_cand, prep_cand, prep_pos)
            if reasons:
                eligibility_matrix[cand_idx, pos_idx] = False
                disqualifications[(cand_idx, pos_idx)] = reasons

    return EligibilityResult(
        eligibility_matrix=eligibility_matrix,
        candidate_ids=candidate_ids,
        position_ids=position_ids,
        disqualifications=disqualifications,
    )
