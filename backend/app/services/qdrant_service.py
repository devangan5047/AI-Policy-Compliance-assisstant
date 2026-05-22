import math

from app.config.settings import get_settings
from app.models.openai_models import PolicyChunk, QueryFilters


class QdrantService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, PolicyChunk] = {}

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if self.settings.qdrant_url:
                self._client = QdrantClient(
                    url=self.settings.qdrant_url,
                    api_key=self.settings.qdrant_api_key,
                )
                existing = [collection.name for collection in self._client.get_collections().collections]
                if self.settings.qdrant_collection not in existing:
                    self._client.create_collection(
                        collection_name=self.settings.qdrant_collection,
                        vectors_config=VectorParams(
                            size=self.settings.embedding_dimension,
                            distance=Distance.COSINE,
                        ),
                    )
        except Exception:
            self._client = None

    def upsert(self, chunks: list[PolicyChunk], vectors: list[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = vector

        if not self._client:
            return

        from qdrant_client.models import PointStruct

        self._client.upsert(
            collection_name=self.settings.qdrant_collection,
            points=[
                PointStruct(
                    id=abs(hash(chunk.id)),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata.model_dump(),
                    },
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    def search(self, vector: list[float], filters: QueryFilters, top_k: int) -> list[PolicyChunk]:
        if self._client:
            qdrant_filter = self._build_filter(filters)
            hits = self._client.search(
                collection_name=self.settings.qdrant_collection,
                query_vector=vector,
                query_filter=qdrant_filter,
                limit=top_k,
            )
            return [
                PolicyChunk(
                    id=hit.payload["chunk_id"],
                    text=hit.payload["text"],
                    metadata=hit.payload["metadata"],
                    score=hit.score,
                )
                for hit in hits
            ]

        scored = []
        for chunk in self._filtered_chunks(filters):
            score = self._cosine(vector, self._vectors.get(chunk.id, []))
            if score > 0:
                scored.append(chunk.model_copy(update={"score": score}))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _build_filter(self, filters: QueryFilters):
        values = filters.model_dump(exclude_none=True)
        if not values:
            return None
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return Filter(
            must=[
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                for key, value in values.items()
            ]
        )

    def _filtered_chunks(self, filters: QueryFilters) -> list[PolicyChunk]:
        values = filters.model_dump(exclude_none=True)
        if not values:
            return list(self._chunks.values())
        return [
            chunk
            for chunk in self._chunks.values()
            if all(getattr(chunk.metadata, key) == value for key, value in values.items())
        ]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if not left_norm or not right_norm:
            return 0.0
        return numerator / (left_norm * right_norm)
