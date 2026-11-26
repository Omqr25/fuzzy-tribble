"""Faceted embedding generation for candidates and positions.

Loads a sentence-transformer model once (via ``EmbeddingService.load``) and
generates three L2-normalised vectors per entity — skills, academic, and
practical — so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from app.schemas.candidate import CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionEmbedRequest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Text composition — pure functions, one per (facet × side)
# ---------------------------------------------------------------------------

_EMPTY_TEXT_PLACEHOLDER = " "


def compose_candidate_skills_text(
    metadata: CandidateMetadata, parsed_cv: ParsedCV
) -> str:
    """Combine DB technical_skills with CV-extracted skills."""
    parts = list(metadata.technical_skills) + list(parsed_cv.skills)
    return ", ".join(parts)


def compose_candidate_academic_text(
    metadata: CandidateMetadata, parsed_cv: ParsedCV
) -> str:
    """Combine program, field, major, year-of-study, and education summary."""
    fragments: list[str] = []
    for value in (metadata.program, metadata.field, metadata.major):
        if value:
            fragments.append(value)
    if metadata.year_of_study is not None:
        fragments.append(f"year {metadata.year_of_study}")
    if parsed_cv.education_summary:
        fragments.append(parsed_cv.education_summary)
    return " ".join(fragments)


def compose_candidate_practical_text(
    metadata: CandidateMetadata, parsed_cv: ParsedCV
) -> str:
    """Combine experience summary, projects summary, and additional info."""
    fragments: list[str] = []
    if parsed_cv.experience_summary:
        fragments.append(parsed_cv.experience_summary)
    if parsed_cv.projects_summary:
        fragments.append(parsed_cv.projects_summary)
    if metadata.additional_info:
        fragments.append(metadata.additional_info)
    return " ".join(fragments)


def compose_position_skills_text(request: PositionEmbedRequest) -> str:
    """Combine required and preferred skills."""
    parts = list(request.required_skills) + list(request.preferred_skills)
    return ", ".join(parts)


def compose_position_academic_text(request: PositionEmbedRequest) -> str:
    """Combine target majors and academic level."""
    fragments: list[str] = []
    if request.target_majors:
        fragments.append(", ".join(request.target_majors))
    if request.academic_level:
        fragments.append(request.academic_level)
    return " ".join(fragments)


def compose_position_practical_text(request: PositionEmbedRequest) -> str:
    """Combine responsibilities and experience level."""
    fragments: list[str] = []
    if request.responsibilities:
        fragments.append(request.responsibilities)
    if request.experience_level:
        fragments.append(request.experience_level)
    return " ".join(fragments)


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------


class EmbeddingService:
    """Thin wrapper around a sentence-transformer model.

    Use the ``load`` classmethod in production.  In tests, construct directly
    with any object that exposes ``.encode(sentences, ...)`` and
    ``.get_sentence_embedding_dimension() -> int``.
    """

    def __init__(self, model: Any, *, model_name: str) -> None:
        self._model = model
        self._model_name = model_name

    @classmethod
    def load(cls, model_name: str) -> EmbeddingService:
        """Download / load the model from the HuggingFace cache."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading sentence-transformer model '%s' …", model_name)
        model = SentenceTransformer(model_name)
        return cls(model=model, model_name=model_name)

    # -- properties ----------------------------------------------------------

    @property
    def model_version(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    # -- low-level encode ----------------------------------------------------

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode *texts* into L2-normalised vectors (list-of-lists)."""
        safe = [t if t.strip() else _EMPTY_TEXT_PLACEHOLDER for t in texts]
        embeddings: np.ndarray = self._model.encode(
            safe,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    # -- high-level faceted embed --------------------------------------------

    def embed_position(self, request: PositionEmbedRequest) -> FacetedVectors:
        """Generate three aligned vectors for a position."""
        texts = [
            compose_position_skills_text(request),
            compose_position_academic_text(request),
            compose_position_practical_text(request),
        ]
        vectors = self.encode(texts)
        return FacetedVectors(
            skills=vectors[0], academic=vectors[1], practical=vectors[2]
        )

    def embed_candidate(
        self, metadata: CandidateMetadata, parsed_cv: ParsedCV
    ) -> FacetedVectors:
        """Generate three aligned vectors for a candidate."""
        texts = [
            compose_candidate_skills_text(metadata, parsed_cv),
            compose_candidate_academic_text(metadata, parsed_cv),
            compose_candidate_practical_text(metadata, parsed_cv),
        ]
        vectors = self.encode(texts)
        return FacetedVectors(
            skills=vectors[0], academic=vectors[1], practical=vectors[2]
        )
