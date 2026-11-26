import math


class FakeEmbeddingModel:
    """Deterministic embedding model for tests; produces normalized vectors.

    Every call returns vectors whose values are determined solely by the input
    text, so assertions on equality or inequality are stable across runs.
    """

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.model_version = "fake-test-model"

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            seed = sum(ord(c) for c in text) + 1
            raw = [
                ((seed * (i + 1) * 7919 + i * 104729) % 10000) / 10000.0 + 0.001
                for i in range(self.dimension)
            ]
            norm = math.sqrt(sum(v * v for v in raw))
            results.append([v / norm for v in raw])
        return results
