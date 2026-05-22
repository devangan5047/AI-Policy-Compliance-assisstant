import hashlib
import math

from app.config.settings import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.settings.embedding_model)
            except Exception:
                self._model = False
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self.model
        if model:
            vectors = model.encode(texts, normalize_embeddings=True)
            return [vector.tolist() for vector in vectors]
        return [self._hash_embedding(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def _hash_embedding(self, text: str) -> list[float]:
        dimension = self.settings.embedding_dimension
        vector = [0.0] * dimension
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % dimension
            sign = 1 if digest[4] % 2 == 0 else -1
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
