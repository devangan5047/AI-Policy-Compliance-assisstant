from app.models.openai_models import PolicyChunk, QueryFilters
from app.rag.reranker import AuthorityReranker
from app.services.elasticsearch_service import ElasticsearchService
from app.services.embedding_service import EmbeddingService
from app.services.qdrant_service import QdrantService


class HybridSearch:
    def __init__(
        self,
        embeddings: EmbeddingService,
        qdrant: QdrantService,
        elasticsearch: ElasticsearchService,
    ) -> None:
        self.embeddings = embeddings
        self.qdrant = qdrant
        self.elasticsearch = elasticsearch
        self.reranker = AuthorityReranker()

    def search(self, query: str, filters: QueryFilters, top_k: int) -> list[PolicyChunk]:
        query_vector = self.embeddings.embed_one(query)
        vector_hits = self.qdrant.search(query_vector, filters, top_k * 2)
        keyword_hits = self.elasticsearch.search(query, filters, top_k * 2)

        merged: dict[str, PolicyChunk] = {}
        for rank, chunk in enumerate(vector_hits, start=1):
            merged[chunk.id] = chunk.model_copy(update={"score": chunk.score + 1 / rank})
        for rank, chunk in enumerate(keyword_hits, start=1):
            existing = merged.get(chunk.id)
            score = chunk.score + 1 / rank
            if existing:
                merged[chunk.id] = existing.model_copy(update={"score": existing.score + score})
            else:
                merged[chunk.id] = chunk.model_copy(update={"score": score})

        return self.reranker.rerank(list(merged.values()), top_k)
