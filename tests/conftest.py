from __future__ import annotations

import numpy as np

from app.services.embedder import EmbeddingModel


class FakeSentenceTransformer:
    """Deterministic SentenceTransformer stub for unit and integration tests.

    ``normalize_embeddings`` intentionally defaults to ``False`` so tests
    immediately catch regressions if ``EmbeddingModel`` stops requesting
    normalized vectors.
    """

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.calls: list[dict[str, object]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dimension

    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.calls.append(
            {
                "sentences": list(sentences),
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        rows: list[list[float]] = []
        for text in sentences:
            seed = sum(ord(ch) for ch in text) + 1
            raw = [
                ((seed * (idx + 1) * 7919 + idx * 104729) % 10000) / 10000.0 + 0.5
                for idx in range(self._dimension)
            ]
            rows.append(raw)

        vectors = np.asarray(rows, dtype=np.float64)
        if normalize_embeddings:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.where(norms == 0, 1.0, norms)
        return vectors


def make_fake_embedding_model(
    dimension: int = 4,
    model_name: str = "fake-test-model",
) -> EmbeddingModel:
    """Build an ``EmbeddingModel`` backed by ``FakeSentenceTransformer``."""
    return EmbeddingModel(
        model_name,
        _model=FakeSentenceTransformer(dimension=dimension),
    )
