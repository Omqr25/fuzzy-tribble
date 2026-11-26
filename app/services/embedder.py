"""Faceted embedding generation for candidates and positions.

Loads a sentence-transformer model once at startup and produces three
L2-normalized vectors per entity (skills, academic, practical) so cosine
similarity reduces to a dot product. Empty facets produce exact zero vectors
so missing information contributes 0.0 to similarity scores.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Literal

from app.schemas.candidate import CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionEmbedRequest

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

TextRole = Literal["query", "passage"]
_E5_MODEL_RE = re.compile(r"(?:^|[/-])e5(?:[/-]|$)", re.IGNORECASE)


def _requires_e5_prefix(model_name: str) -> bool:
    """Return True for E5-family encoders that require query/passage prefixes."""
    return bool(_E5_MODEL_RE.search(model_name or ""))


class EmbeddingModel:
    """Thread-safe wrapper around a SentenceTransformer loaded once at startup.

    ``normalize_embeddings=True`` is always passed so that cosine similarity
    reduces to a simple dot product downstream.
    """

    def __init__(self, model_name: str, *, _model: Any | None = None) -> None:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer as _ST
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding support requires sentence-transformers; install it "
                    "with: pip install sentence-transformers"
                ) from exc

            logger.info("Loading embedding model: %s …", model_name)
            self._model: SentenceTransformer | Any = _ST(model_name)
        else:
            self._model = _model

        self.model_version: str = model_name
        self._uses_e5_prefix: bool = _requires_e5_prefix(model_name)
        self.dimension: int = int(self._model.get_sentence_embedding_dimension())
        if _model is None:
            logger.info(
                "Embedding model ready: %s (dimension=%d)", model_name, self.dimension
            )

    def encode_texts(
        self,
        texts: list[str],
        *,
        role: TextRole | None = None,
    ) -> list[list[float]]:
        """Encode texts into L2-normalized vectors; empty texts map to zero vectors."""
        if not texts:
            return []

        prefix = f"{role}: " if role and self._uses_e5_prefix else ""
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []

        for idx, text in enumerate(texts):
            cleaned = (text or "").strip()
            if cleaned:
                non_empty_indices.append(idx)
                non_empty_texts.append(f"{prefix}{cleaned}")

        results: list[list[float]] = [[0.0] * self.dimension for _ in texts]
        if not non_empty_texts:
            return results

        embeddings = self._model.encode(
            non_empty_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        encoded_rows = (
            embeddings.tolist() if hasattr(embeddings, "tolist") else list(embeddings)
        )
        for idx, vector in zip(non_empty_indices, encoded_rows, strict=True):
            results[idx] = [float(value) for value in vector]
        return results


# ── Text assembly helpers ──────────────────────────────────────────────────


def _join_non_empty(*parts: str) -> str:
    """Concatenate non-blank strings with a single space separator."""
    return " ".join(p.strip() for p in parts if p and p.strip())


def _dedupe_clean_items(items: list[str]) -> list[str]:
    """Strip items, drop blanks, and deduplicate case-insensitively in order."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in items:
        token = (raw or "").strip()
        if not token:
            continue
        key = token.casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(token)
    return cleaned


def compose_candidate_skills_text(
    metadata: CandidateMetadata,
    parsed_cv: ParsedCV,
) -> str:
    """Combine DB technical_skills with CV-extracted skills."""
    skills = _dedupe_clean_items([*metadata.technical_skills, *parsed_cv.skills])
    return ", ".join(skills)


def compose_candidate_academic_text(
    metadata: CandidateMetadata,
    parsed_cv: ParsedCV,
) -> str:
    """Combine program, field, major, year of study, and education summary."""
    parts = [metadata.program, metadata.field, metadata.major]
    if metadata.year_of_study is not None:
        parts.append(f"year of study {metadata.year_of_study}")
    parts.append(parsed_cv.education_summary)
    return _join_non_empty(*parts)


def compose_candidate_practical_text(
    metadata: CandidateMetadata,
    parsed_cv: ParsedCV,
) -> str:
    """Combine experience summary, projects summary, and additional info."""
    return _join_non_empty(
        parsed_cv.experience_summary,
        parsed_cv.projects_summary,
        metadata.additional_info,
    )


def compose_position_skills_text(position: PositionEmbedRequest) -> str:
    """Combine required and preferred skills."""
    skills = _dedupe_clean_items(
        [*position.required_skills, *position.preferred_skills]
    )
    return ", ".join(skills)


def compose_position_academic_text(position: PositionEmbedRequest) -> str:
    """Combine target majors and academic level."""
    majors = _dedupe_clean_items(position.target_majors)
    return _join_non_empty(", ".join(majors), position.academic_level)


def compose_position_practical_text(position: PositionEmbedRequest) -> str:
    """Combine responsibilities (or description fallback) and experience level."""
    body = (
        position.responsibilities
        if position.responsibilities.strip()
        else position.description
    )
    return _join_non_empty(body, position.experience_level)


# ── Faceted vector builders ────────────────────────────────────────────────


def build_candidate_vectors(
    metadata: CandidateMetadata,
    parsed_cv: ParsedCV,
    model: EmbeddingModel,
) -> FacetedVectors:
    """Generate three faceted vectors for a candidate."""
    vectors = model.encode_texts(
        [
            compose_candidate_skills_text(metadata, parsed_cv),
            compose_candidate_academic_text(metadata, parsed_cv),
            compose_candidate_practical_text(metadata, parsed_cv),
        ],
        role="passage",
    )
    return FacetedVectors(
        skills=vectors[0],
        academic=vectors[1],
        practical=vectors[2],
    )


def build_position_vectors(
    position: PositionEmbedRequest,
    model: EmbeddingModel,
) -> FacetedVectors:
    """Generate three faceted vectors for a position."""
    vectors = model.encode_texts(
        [
            compose_position_skills_text(position),
            compose_position_academic_text(position),
            compose_position_practical_text(position),
        ],
        role="query",
    )
    return FacetedVectors(
        skills=vectors[0],
        academic=vectors[1],
        practical=vectors[2],
    )
