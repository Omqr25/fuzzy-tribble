"""Vectorized hybrid scoring matrix and sparse graph pruning (Phase 5 Stages 2-3).

Computes per-facet semantic cosine similarities via BLAS matrix multiplication
combined with boundary-aware exact lexical skill coverage and per-position
active-weight renormalization, then prunes the dense bipartite score matrix
into a deterministic sparse graph (preserving locked overrides) for CP-SAT.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import TypeVar

import numpy as np

from app.schemas.allocation import (
    AlternativePlacement,
    LockedOverride,
    ScoreBreakdown,
    ScoringWeights,
)
from app.schemas.candidate import CandidateForAllocation
from app.schemas.position import PositionForAllocation
from app.services.text_matching import (
    build_candidate_corpus_segments,
    compile_phrase_pattern,
    contains_phrase,
    dedupe_normalized_items,
    validate_index,
)

SCORE_SCALE_FACTOR = 10_000
_FACETS = ("skills", "academic", "practical")
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ScoringResult:
    """Dense N x M hybrid score matrices and audit helpers."""

    total_scores: np.ndarray
    skills_semantic: np.ndarray
    academic_semantic: np.ndarray
    practical_semantic: np.ndarray
    required_skills_lexical: np.ndarray
    preferred_skills_lexical: np.ndarray
    candidate_ids: tuple[str, ...]
    position_ids: tuple[str, ...]

    def breakdown(self, i: int | np.integer, j: int | np.integer) -> ScoreBreakdown:
        """Return the sub-score breakdown for candidate ``i`` and position ``j``."""
        cand_idx = validate_index(i, len(self.candidate_ids), "candidate")
        pos_idx = validate_index(j, len(self.position_ids), "position")
        return ScoreBreakdown(
            skills_semantic=float(self.skills_semantic[cand_idx, pos_idx]),
            academic_semantic=float(self.academic_semantic[cand_idx, pos_idx]),
            practical_semantic=float(self.practical_semantic[cand_idx, pos_idx]),
            required_skills_lexical=float(
                self.required_skills_lexical[cand_idx, pos_idx]
            ),
            preferred_skills_lexical=float(
                self.preferred_skills_lexical[cand_idx, pos_idx]
            ),
        )

    def top_eligible_alternatives(
        self,
        i: int | np.integer,
        eligibility_matrix: np.ndarray,
        *,
        exclude_position_idx: int | np.integer | None = None,
        limit: int = 2,
    ) -> list[AlternativePlacement]:
        """Return up to ``limit`` highest-scoring eligible positions for candidate ``i``."""
        cand_idx = validate_index(i, len(self.candidate_ids), "candidate")
        eligible = _validate_eligibility_matrix(
            eligibility_matrix, self.total_scores.shape
        )
        excluded_idx: int | None = None
        if exclude_position_idx is not None:
            excluded_idx = validate_index(
                exclude_position_idx, len(self.position_ids), "position"
            )
        if (
            isinstance(limit, (bool, np.bool_))
            or not isinstance(limit, (int, np.integer))
            or int(limit) < 0
        ):
            raise ValueError(f"limit must be a non-negative integer (got {limit!r}).")
        resolved_limit = int(limit)
        if resolved_limit == 0:
            return []

        eligible_indices = np.flatnonzero(eligible[cand_idx])
        candidates_j = [
            int(j) for j in eligible_indices if int(j) != excluded_idx
        ]
        if not candidates_j:
            return []

        candidates_j.sort(
            key=lambda j: (-float(self.total_scores[cand_idx, j]), j)
        )
        return [
            AlternativePlacement(
                position_id=self.position_ids[j],
                score=float(self.total_scores[cand_idx, j]),
                score_breakdown=self.breakdown(cand_idx, j),
            )
            for j in candidates_j[:resolved_limit]
        ]


@dataclass(frozen=True, slots=True)
class SparseEdge:
    """A surviving candidate-position edge after eligibility and threshold pruning."""

    candidate_idx: int
    position_idx: int
    candidate_id: str
    position_id: str
    score: float
    scaled_weight: int
    breakdown: ScoreBreakdown
    is_locked: bool = False


@dataclass(frozen=True, slots=True)
class PrunedGraph:
    """Sparse bipartite graph and candidate-level diagnostic masks for Phase 6."""

    edges: list[SparseEdge]
    edges_by_candidate: dict[int, list[SparseEdge]]
    has_eligible_position: tuple[bool, ...]
    has_above_threshold_position: tuple[bool, ...]


# ── Validation & vector helpers ────────────────────────────────────────────


def _validate_unique_ids(
    items: Sequence[_T],
    id_getter: Callable[[_T], str],
    label: str,
) -> tuple[str, ...]:
    ids: list[str] = []
    seen: set[str] = set()
    for item in items:
        item_id = id_getter(item)
        if item_id in seen:
            raise ValueError(f"Duplicate {label}_id: {item_id!r}.")
        seen.add(item_id)
        ids.append(item_id)
    return tuple(ids)


def _stack_and_normalize_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    facet_name: str,
    entity_kind: str,
    entity_ids: Sequence[str],
    expected_dim: int | None = None,
) -> tuple[np.ndarray, int]:
    """Validate finite 1D vectors and L2-normalize non-zero rows."""
    dim = expected_dim
    rows: list[np.ndarray] = []

    for entity_id, raw_vec in zip(entity_ids, vectors, strict=True):
        row = np.asarray(raw_vec, dtype=np.float64)
        if row.ndim != 1 or row.size == 0:
            raise ValueError(
                f"{entity_kind} {entity_id!r} facet {facet_name!r} vector must be "
                "a non-empty 1D sequence."
            )
        if dim is None:
            dim = int(row.size)
        elif row.size != dim:
            raise ValueError(
                f"{entity_kind} {entity_id!r} facet {facet_name!r} has dimension "
                f"{row.size}, expected {dim}."
            )
        if not np.all(np.isfinite(row)):
            raise ValueError(
                f"{entity_kind} {entity_id!r} facet {facet_name!r} contains "
                "non-finite values (NaN or Inf)."
            )
        rows.append(row)

    if dim is None:
        raise ValueError(f"{entity_kind} vectors for facet {facet_name!r} must not be empty.")
    matrix = np.vstack(rows).astype(np.float64, copy=False)
    return _l2_normalize_rows(matrix), dim


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize non-zero rows safely while leaving zero rows as exact zeros."""
    # Scale by row max-abs before squaring so large/small magnitudes never overflow/underflow.
    max_abs = np.max(np.abs(matrix), axis=1, keepdims=True)
    is_zero = max_abs == 0.0
    safe_scale = np.where(is_zero, 1.0, max_abs)
    scaled = np.where(is_zero, 0.0, matrix / safe_scale)
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    safe_norms = np.where(is_zero, 1.0, norms)
    return np.where(is_zero, 0.0, scaled / safe_norms)


def _validate_eligibility_matrix(
    eligibility_matrix: np.ndarray,
    expected_shape: tuple[int, int],
) -> np.ndarray:
    arr = np.asarray(eligibility_matrix)
    if arr.dtype != np.bool_:
        raise ValueError(
            f"eligibility_matrix must have dtype bool (got {arr.dtype})."
        )
    if arr.ndim != 2 or arr.shape != expected_shape:
        raise ValueError(
            f"eligibility_matrix must have shape {expected_shape} (got {arr.shape})."
        )
    return arr


# ── Boundary-aware lexical skill helpers ───────────────────────────────────


def _compute_coverage_matrix(
    n_candidates: int,
    skill_lists: Sequence[tuple[str, ...]],
    skill_matches: dict[str, np.ndarray],
) -> np.ndarray:
    matrix = np.zeros((n_candidates, len(skill_lists)), dtype=np.float64)
    for pos_idx, skills in enumerate(skill_lists):
        if not skills:
            continue
        matched_counts = np.zeros(n_candidates, dtype=np.float64)
        for skill in skills:
            matched_counts += skill_matches[skill]
        matrix[:, pos_idx] = matched_counts / len(skills)
    return matrix


def _compute_lexical_matrices(
    candidates: Sequence[CandidateForAllocation],
    positions: Sequence[PositionForAllocation],
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[str, ...]],
    list[tuple[str, ...]],
]:
    corpora = [build_candidate_corpus_segments(c) for c in candidates]
    req_lists = [
        tuple(norm for _, norm in dedupe_normalized_items(p.required_skills))
        for p in positions
    ]
    pref_lists = [
        tuple(norm for _, norm in dedupe_normalized_items(p.preferred_skills))
        for p in positions
    ]

    unique_skills = {skill for group in (*req_lists, *pref_lists) for skill in group}
    skill_matches: dict[str, np.ndarray] = {}
    for skill in unique_skills:
        pattern = compile_phrase_pattern(skill)
        skill_matches[skill] = np.fromiter(
            (
                any(contains_phrase(segment, pattern, skill) for segment in segments)
                for segments in corpora
            ),
            dtype=np.float64,
            count=len(corpora),
        )

    n_candidates = len(candidates)
    return (
        _compute_coverage_matrix(n_candidates, req_lists, skill_matches),
        _compute_coverage_matrix(n_candidates, pref_lists, skill_matches),
        req_lists,
        pref_lists,
    )


# ── Public Stage 2 & Stage 3 APIs ──────────────────────────────────────────


def score_pairs(
    candidates: Sequence[CandidateForAllocation],
    positions: Sequence[PositionForAllocation],
    weights: ScoringWeights | None = None,
) -> ScoringResult:
    """Compute the Stage 2 N x M vectorized hybrid scoring matrix."""
    if not candidates:
        raise ValueError("candidates must not be empty.")
    if not positions:
        raise ValueError("positions must not be empty.")

    resolved_weights = ScoringWeights() if weights is None else weights
    if not isinstance(resolved_weights, ScoringWeights):
        raise ValueError("weights must be a ScoringWeights instance or None.")

    candidate_ids = _validate_unique_ids(
        candidates, lambda c: c.candidate_id, "candidate"
    )
    position_ids = _validate_unique_ids(
        positions, lambda p: p.position_id, "position"
    )

    semantic_matrices: dict[str, np.ndarray] = {}
    pos_active_facets: dict[str, np.ndarray] = {}
    for facet in _FACETS:
        cand_matrix, dim = _stack_and_normalize_vectors(
            [getattr(c.vectors, facet) for c in candidates],
            facet_name=facet,
            entity_kind="candidate",
            entity_ids=candidate_ids,
        )
        pos_matrix, _ = _stack_and_normalize_vectors(
            [getattr(p.vectors, facet) for p in positions],
            facet_name=facet,
            entity_kind="position",
            entity_ids=position_ids,
            expected_dim=dim,
        )
        pos_active_facets[facet] = np.any(pos_matrix != 0.0, axis=1)
        semantic_matrices[facet] = np.clip(cand_matrix @ pos_matrix.T, 0.0, 1.0)

    req_lexical, pref_lexical, req_lists, pref_lists = _compute_lexical_matrices(
        candidates, positions
    )

    skills_sim = semantic_matrices["skills"]
    academic_sim = semantic_matrices["academic"]
    practical_sim = semantic_matrices["practical"]

    blended = (
        resolved_weights.skills_semantic * skills_sim
        + resolved_weights.academic_semantic * academic_sim
        + resolved_weights.practical_semantic * practical_sim
        + resolved_weights.required_skills_lexical * req_lexical
        + resolved_weights.preferred_skills_lexical * pref_lexical
    )

    req_active = np.fromiter(
        (len(skills) > 0 for skills in req_lists),
        dtype=bool,
        count=len(positions),
    )
    pref_active = np.fromiter(
        (len(skills) > 0 for skills in pref_lists),
        dtype=bool,
        count=len(positions),
    )

    # Renormalize per position column across active position-side components only,
    # keeping sub-score matrices raw while preventing empty position fields from
    # capping candidate scores below 1.0 (never renormalizing for empty candidate fields).
    active_weight_sums = (
        resolved_weights.skills_semantic
        * pos_active_facets["skills"].astype(np.float64)
        + resolved_weights.academic_semantic
        * pos_active_facets["academic"].astype(np.float64)
        + resolved_weights.practical_semantic
        * pos_active_facets["practical"].astype(np.float64)
        + resolved_weights.required_skills_lexical * req_active.astype(np.float64)
        + resolved_weights.preferred_skills_lexical * pref_active.astype(np.float64)
    )
    safe_denom = np.where(active_weight_sums > 0.0, active_weight_sums, 1.0)
    renormalized = np.where(active_weight_sums > 0.0, blended / safe_denom, 0.0)
    total_scores = np.clip(renormalized, 0.0, 1.0)

    return ScoringResult(
        total_scores=total_scores,
        skills_semantic=skills_sim,
        academic_semantic=academic_sim,
        practical_semantic=practical_sim,
        required_skills_lexical=req_lexical,
        preferred_skills_lexical=pref_lexical,
        candidate_ids=candidate_ids,
        position_ids=position_ids,
    )


def _select_top_k_indices(
    scores_row: np.ndarray,
    surviving_indices: np.ndarray,
    top_k: int,
) -> list[int]:
    """Select up to ``top_k`` surviving position indices ordered by (-score, position_idx)."""
    surviving_count = int(surviving_indices.size)
    if surviving_count == 0:
        return []

    if surviving_count <= top_k:
        chosen = [int(idx) for idx in surviving_indices]
        chosen.sort(key=lambda pos_idx: (-float(scores_row[pos_idx]), pos_idx))
        return chosen

    # Partition strictly among surviving entries so masked/-inf values can never leak.
    surviving_scores = scores_row[surviving_indices]
    part_idx = np.argpartition(-surviving_scores, kth=top_k - 1)
    cutoff_score = float(surviving_scores[part_idx[top_k - 1]])

    if np.any(surviving_scores[part_idx[top_k:]] == cutoff_score):
        # Resolve cutoff ties deterministically by lowest position_idx (surviving_indices is ascending).
        strictly_above = surviving_indices[
            part_idx[:top_k][surviving_scores[part_idx[:top_k]] > cutoff_score]
        ]
        needed_ties = top_k - int(strictly_above.size)
        at_cutoff = surviving_indices[surviving_scores == cutoff_score][:needed_ties]
        selected_indices = np.concatenate((strictly_above, at_cutoff))
    else:
        selected_indices = surviving_indices[part_idx[:top_k]]

    chosen = [int(idx) for idx in selected_indices]
    chosen.sort(key=lambda pos_idx: (-float(scores_row[pos_idx]), pos_idx))
    return chosen


def _resolve_locked_overrides(
    locked_overrides: Sequence[LockedOverride] | None,
    candidate_ids: tuple[str, ...],
    position_ids: tuple[str, ...],
) -> dict[int, int]:
    """Validate and deduplicate locked overrides into ``{cand_idx: pos_idx}``."""
    if locked_overrides is None:
        return {}
    if isinstance(locked_overrides, (str, bytes)) or not isinstance(
        locked_overrides, Sequence
    ):
        raise ValueError(
            "locked_overrides must be a sequence of LockedOverride or None."
        )
    if not locked_overrides:
        return {}

    cand_idx_by_id = {cid: idx for idx, cid in enumerate(candidate_ids)}
    pos_idx_by_id = {pid: idx for idx, pid in enumerate(position_ids)}
    locked_by_cand: dict[int, int] = {}

    for override in locked_overrides:
        if not isinstance(override, LockedOverride):
            raise ValueError(
                "locked_overrides entries must be LockedOverride instances "
                f"(got {type(override).__name__})."
            )
        if override.candidate_id not in cand_idx_by_id:
            raise ValueError(
                f"Unknown candidate_id in locked_overrides: {override.candidate_id!r}."
            )
        if override.position_id not in pos_idx_by_id:
            raise ValueError(
                f"Unknown position_id in locked_overrides: {override.position_id!r}."
            )
        cand_idx = cand_idx_by_id[override.candidate_id]
        pos_idx = pos_idx_by_id[override.position_id]
        existing_pos_idx = locked_by_cand.get(cand_idx)
        if existing_pos_idx is not None and existing_pos_idx != pos_idx:
            raise ValueError(
                f"Conflicting locked_overrides for candidate_id "
                f"{override.candidate_id!r}: {position_ids[existing_pos_idx]!r} "
                f"vs {override.position_id!r}."
            )
        locked_by_cand[cand_idx] = pos_idx

    return locked_by_cand


def prune_sparse_graph(
    scoring: ScoringResult,
    eligibility_matrix: np.ndarray,
    *,
    min_match_threshold: float = 0.45,
    top_k_per_candidate: int = 12,
    locked_overrides: Sequence[LockedOverride] | None = None,
) -> PrunedGraph:
    """Prune ineligible/below-threshold pairs and retain top-K edges plus locks."""
    eligible = _validate_eligibility_matrix(
        eligibility_matrix, scoring.total_scores.shape
    )
    if (
        isinstance(min_match_threshold, bool)
        or not isinstance(min_match_threshold, (int, float))
        or not math.isfinite(float(min_match_threshold))
        or not (0.0 <= float(min_match_threshold) <= 1.0)
    ):
        raise ValueError(
            f"min_match_threshold must be in [0.0, 1.0] (got {min_match_threshold!r})."
        )
    if (
        isinstance(top_k_per_candidate, bool)
        or not isinstance(top_k_per_candidate, int)
        or top_k_per_candidate < 1
    ):
        raise ValueError(
            f"top_k_per_candidate must be an integer >= 1 (got {top_k_per_candidate!r})."
        )

    locked_by_cand = _resolve_locked_overrides(
        locked_overrides, scoring.candidate_ids, scoring.position_ids
    )

    threshold = float(min_match_threshold)
    surviving_mask = eligible & (scoring.total_scores >= threshold)

    has_eligible_position = tuple(bool(val) for val in np.any(eligible, axis=1))
    has_above_threshold_position = tuple(
        bool(val) for val in np.any(surviving_mask, axis=1)
    )

    edges: list[SparseEdge] = []
    edges_by_candidate: dict[int, list[SparseEdge]] = {}

    for cand_idx, cand_id in enumerate(scoring.candidate_ids):
        surviving_indices = np.flatnonzero(surviving_mask[cand_idx])
        selected_positions = _select_top_k_indices(
            scoring.total_scores[cand_idx],
            surviving_indices,
            top_k_per_candidate,
        )
        locked_pos_idx = locked_by_cand.get(cand_idx)
        if locked_pos_idx is not None and locked_pos_idx not in selected_positions:
            selected_positions.append(locked_pos_idx)
            selected_positions.sort(
                key=lambda pos_idx: (
                    -float(scoring.total_scores[cand_idx, pos_idx]),
                    pos_idx,
                )
            )

        candidate_edges: list[SparseEdge] = []
        for pos_idx in selected_positions:
            score = float(scoring.total_scores[cand_idx, pos_idx])
            edge = SparseEdge(
                candidate_idx=cand_idx,
                position_idx=pos_idx,
                candidate_id=cand_id,
                position_id=scoring.position_ids[pos_idx],
                score=score,
                scaled_weight=int(round(score * SCORE_SCALE_FACTOR)),
                breakdown=scoring.breakdown(cand_idx, pos_idx),
                is_locked=(pos_idx == locked_pos_idx),
            )
            candidate_edges.append(edge)
        edges_by_candidate[cand_idx] = candidate_edges
        edges.extend(candidate_edges)

    return PrunedGraph(
        edges=edges,
        edges_by_candidate=edges_by_candidate,
        has_eligible_position=has_eligible_position,
        has_above_threshold_position=has_above_threshold_position,
    )
