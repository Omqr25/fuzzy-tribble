"""Phase 5 Stage 1: Deterministic hard eligibility gate.

Evaluates every ``(candidate, position)`` pair against the six hard constraints
in ``PositionConstraints`` before scoring or CP-SAT optimization, producing a
boolean ``(N, M)`` eligibility matrix and structured disqualification reasons.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

import numpy as np

from app.schemas.candidate import CandidateForAllocation
from app.schemas.position import PositionForAllocation
from app.services.text_matching import (
    build_candidate_corpus_segments,
    compile_phrase_pattern,
    contains_phrase,
    dedupe_normalized_items,
    normalize_text,
    validate_index,
)


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

    def reasons_for_pair(
        self,
        candidate_idx: int | np.integer,
        position_idx: int | np.integer,
    ) -> list[str]:
        """Return disqualification messages for ``(candidate_idx, position_idx)``."""
        cand_idx = validate_index(candidate_idx, len(self.candidate_ids), "candidate")
        pos_idx = validate_index(position_idx, len(self.position_ids), "position")
        return [
            reason.message
            for reason in self.disqualifications.get((cand_idx, pos_idx), ())
        ]

    def disqualification_summary_for_candidate(
        self, candidate_idx: int | np.integer
    ) -> dict[str, list[str]]:
        """Return ``{position_id: [reasons]}`` for all ineligible positions."""
        cand_idx = validate_index(candidate_idx, len(self.candidate_ids), "candidate")
        summary: dict[str, list[str]] = {}
        for pos_idx, position_id in enumerate(self.position_ids):
            if not self.eligibility_matrix[cand_idx, pos_idx]:
                summary[position_id] = self.reasons_for_pair(cand_idx, pos_idx)
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


def _collapse_whitespace(text: str | None) -> str:
    return " ".join((text or "").split())


def _get_or_compile_pattern(
    normalized_phrase: str,
    pattern_cache: dict[str, re.Pattern[str]],
) -> re.Pattern[str]:
    pattern = pattern_cache.get(normalized_phrase)
    if pattern is None:
        pattern = compile_phrase_pattern(normalized_phrase)
        pattern_cache[normalized_phrase] = pattern
    return pattern


def _prepare_candidate(candidate: CandidateForAllocation) -> _PreparedCandidate:
    meta = candidate.metadata

    major_norm = normalize_text(meta.major)
    field_norm = normalize_text(meta.field)
    if major_norm:
        effective_major_display = _collapse_whitespace(meta.major)
        effective_major_norm = major_norm
    elif field_norm:
        effective_major_display = _collapse_whitespace(meta.field)
        effective_major_norm = field_norm
    else:
        effective_major_display = ""
        effective_major_norm = ""

    return _PreparedCandidate(
        city_norm=normalize_text(meta.city),
        gpa=meta.gpa,
        year_of_study=meta.year_of_study,
        full_time_commitment=meta.full_time_commitment is True,
        effective_major_display=effective_major_display,
        effective_major_norm=effective_major_norm,
        corpus_chunks_norm=build_candidate_corpus_segments(candidate),
    )


def _prepare_position(
    position: PositionForAllocation,
    pattern_cache: dict[str, re.Pattern[str]],
) -> _PreparedPosition:
    rules = position.constraints
    city_items = dedupe_normalized_items(rules.allowed_cities)
    major_items = dedupe_normalized_items(rules.required_majors)
    keyword_items = dedupe_normalized_items(rules.must_have_keywords)

    return _PreparedPosition(
        allowed_cities_display=tuple(display for display, _ in city_items),
        allowed_cities_norm=frozenset(norm for _, norm in city_items),
        min_gpa=rules.min_gpa,
        min_year_of_study=rules.min_year_of_study,
        requires_full_time_commitment=rules.requires_full_time_commitment,
        required_majors_display=tuple(display for display, _ in major_items),
        required_major_matchers=tuple(
            (norm, _get_or_compile_pattern(norm, pattern_cache))
            for _, norm in major_items
        ),
        must_have_keywords=tuple(
            (display, norm, _get_or_compile_pattern(norm, pattern_cache))
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
        city_label = _collapse_whitespace(raw_city)
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
        contains_phrase(cand.effective_major_norm, pattern, req_norm)
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


def _candidate_has_keyword(
    cand_idx: int,
    cand: _PreparedCandidate,
    norm: str,
    pattern: re.Pattern[str],
    keyword_cache: dict[tuple[int, str], bool],
) -> bool:
    cache_key = (cand_idx, norm)
    cached = keyword_cache.get(cache_key)
    if cached is not None:
        return cached
    matched = any(
        contains_phrase(chunk, pattern, norm)
        for chunk in cand.corpus_chunks_norm
    )
    keyword_cache[cache_key] = matched
    return matched


def _check_must_have_keywords(
    cand_idx: int,
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
    keyword_cache: dict[tuple[int, str], bool],
) -> DisqualificationReason | None:
    if not pos.must_have_keywords:
        return None
    missing = [
        display
        for display, norm, pattern in pos.must_have_keywords
        if not _candidate_has_keyword(cand_idx, cand, norm, pattern, keyword_cache)
    ]
    if not missing:
        return None
    return DisqualificationReason(
        rule="must_have_keywords",
        message=f"Missing required keywords: {', '.join(missing)}.",
    )


def _evaluate_pair(
    cand_idx: int,
    raw_candidate: CandidateForAllocation,
    cand: _PreparedCandidate,
    pos: _PreparedPosition,
    keyword_cache: dict[tuple[int, str], bool],
) -> list[DisqualificationReason]:
    checks = (
        _check_allowed_cities(cand, pos, raw_candidate.metadata.city),
        _check_min_gpa(cand, pos),
        _check_min_year_of_study(cand, pos),
        _check_full_time_commitment(cand, pos),
        _check_required_majors(cand, pos),
        _check_must_have_keywords(cand_idx, cand, pos, keyword_cache),
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

    pattern_cache: dict[str, re.Pattern[str]] = {}
    prepared_candidates = [_prepare_candidate(cand) for cand in candidates]
    prepared_positions = [_prepare_position(pos, pattern_cache) for pos in positions]

    eligibility_matrix = np.ones(
        (len(candidates), len(positions)),
        dtype=bool,
    )
    disqualifications: dict[tuple[int, int], list[DisqualificationReason]] = {}
    keyword_cache: dict[tuple[int, str], bool] = {}

    for cand_idx, (raw_cand, prep_cand) in enumerate(
        zip(candidates, prepared_candidates, strict=True)
    ):
        for pos_idx, prep_pos in enumerate(prepared_positions):
            reasons = _evaluate_pair(
                cand_idx,
                raw_cand,
                prep_cand,
                prep_pos,
                keyword_cache,
            )
            if reasons:
                eligibility_matrix[cand_idx, pos_idx] = False
                disqualifications[(cand_idx, pos_idx)] = reasons

    return EligibilityResult(
        eligibility_matrix=eligibility_matrix,
        candidate_ids=candidate_ids,
        position_ids=position_ids,
        disqualifications=disqualifications,
    )
