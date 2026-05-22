import re
from collections import Counter

from app.config.settings import get_settings
from app.models.openai_models import PolicyChunk, QueryFilters


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class ElasticsearchService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._client = None
        self._chunks: dict[str, PolicyChunk] = {}

        if self.settings.elasticsearch_url:
            try:
                from elasticsearch import Elasticsearch

                self._client = Elasticsearch(self.settings.elasticsearch_url)
            except Exception:
                self._client = None

    def index_chunks(self, chunks: list[PolicyChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk
            if self._client:
                self._client.index(
                    index=self.settings.elasticsearch_index,
                    id=chunk.id,
                    document={
                        "text": chunk.text,
                        "metadata": chunk.metadata.model_dump(),
                    },
                )

    def search(self, query: str, filters: QueryFilters, top_k: int) -> list[PolicyChunk]:
        if self._client:
            must = [{"match": {"text": query}}]
            filter_terms = []
            for key, value in filters.model_dump(exclude_none=True).items():
                filter_terms.append({"term": {f"metadata.{key}.keyword": value}})
            response = self._client.search(
                index=self.settings.elasticsearch_index,
                query={"bool": {"must": must, "filter": filter_terms}},
                size=top_k,
            )
            return [
                PolicyChunk(
                    id=hit["_id"],
                    text=hit["_source"]["text"],
                    metadata=hit["_source"]["metadata"],
                    score=hit["_score"],
                )
                for hit in response["hits"]["hits"]
            ]

        query_counts = Counter(TOKEN_RE.findall(query.lower()))
        scored: list[PolicyChunk] = []
        for chunk in self._filtered_chunks(filters):
            doc_counts = Counter(TOKEN_RE.findall(chunk.text.lower()))
            score = sum(doc_counts[token] * weight for token, weight in query_counts.items())
            if score:
                scored.append(chunk.model_copy(update={"score": float(score)}))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _filtered_chunks(self, filters: QueryFilters) -> list[PolicyChunk]:
        values = filters.model_dump(exclude_none=True)
        if not values:
            return list(self._chunks.values())
        return [
            chunk
            for chunk in self._chunks.values()
            if all(getattr(chunk.metadata, key) == value for key, value in values.items())
        ]
