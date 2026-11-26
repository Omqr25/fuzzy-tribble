from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.schemas.candidate import CandidateMetadata, ParsedCV
from app.schemas.position import FacetedVectors, PositionEmbedRequest

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_PLACEHOLDER = "not specified"


class EmbeddingModel:
    """Thread-safe wrapper around a SentenceTransformer loaded once at startup.

    ``normalize_embeddings=True`` is always passed so that cosine similarity
    reduces to a simple dot product downstream.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError as exc:
            raise RuntimeError(
                "Embedding support requires sentence-transformers; install it "
                "with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading embedding model: %s …", model_name)
        self._model: SentenceTransformer = _ST(model_name)
        self.model_version: str = model_name
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model ready: %s (dimension=%d)", model_name, self.dimension
        )

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into L2-normalized vectors."""

        if not texts:
            return []
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()


# ── Text assembly helpers ──────────────────────────────────────────────────


def _join_non_empty(*parts: str) -> str:
    """Concatenate non-blank strings with a single space separator."""

    return " ".join(p.strip() for p in parts if p and p.strip())


# ── Faceted vector builders ────────────────────────────────────────────────


def build_candidate_vectors(
    metadata: CandidateMetadata,
    parsed_cv: ParsedCV,
    model: EmbeddingModel,
) -> FacetedVectors:
    """Generate three faceted vectors for a candidate.

    Facet 1 — Skills:     db.technical_skills + cv.skills
    Facet 2 — Academic:   program + field + major + year_of_study + cv.education
    Facet 3 — Practical:  cv.experience + cv.projects + additional_info
    """

    # Facet 1
    skills_text = _join_non_empty(
        ", ".join(metadata.technical_skills),
        ", ".join(parsed_cv.skills),
    )

    # Facet 2
    academic_parts = [metadata.program, metadata.field, metadata.major]
    if metadata.year_of_study is not None:
        academic_parts.append(f"year of study {metadata.year_of_study}")
    academic_parts.append(parsed_cv.education_summary)
    academic_text = _join_non_empty(*academic_parts)

    # Facet 3
    practical_text = _join_non_empty(
        parsed_cv.experience_summary,
        parsed_cv.projects_summary,
        metadata.additional_info,
    )

    vectors = model.encode_texts([
        skills_text or _PLACEHOLDER,
        academic_text or _PLACEHOLDER,
        practical_text or _PLACEHOLDER,
    ])

    return FacetedVectors(
        skills=vectors[0],
        academic=vectors[1],
        practical=vectors[2],
    )


def build_position_vectors(
    position: PositionEmbedRequest,
    model: EmbeddingModel,
) -> FacetedVectors:
    """Generate three faceted vectors for a position.

    Facet 1 — Skills:     required_skills + preferred_skills
    Facet 2 — Academic:   target_majors + academic_level
    Facet 3 — Practical:  responsibilities + experience_level
    """

    skills_text = _join_non_empty(
        ", ".join(position.required_skills),
        ", ".join(position.preferred_skills),
    )

    academic_text = _join_non_empty(
        ", ".join(position.target_majors),
        position.academic_level,
    )

    practical_text = _join_non_empty(
        position.responsibilities,
        position.experience_level,
    )

    vectors = model.encode_texts([
        skills_text or _PLACEHOLDER,
        academic_text or _PLACEHOLDER,
        practical_text or _PLACEHOLDER,
    ])

    return FacetedVectors(
        skills=vectors[0],
        academic=vectors[1],
        practical=vectors[2],
    )
