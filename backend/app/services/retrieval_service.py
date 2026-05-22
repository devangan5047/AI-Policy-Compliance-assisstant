from uuid import uuid4

from app.models.openai_models import PolicyChunk, PolicyMetadata, QueryFilters
from app.rag.hybrid_search import HybridSearch
from app.services.chunking_service import ChunkingService
from app.services.elasticsearch_service import ElasticsearchService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PDFService
from app.services.qdrant_service import QdrantService


class RetrievalService:
    def __init__(self) -> None:
        self.embeddings = EmbeddingService()
        self.qdrant = QdrantService()
        self.elasticsearch = ElasticsearchService()
        self.chunker = ChunkingService()
        self.pdf = PDFService()
        self.hybrid = HybridSearch(self.embeddings, self.qdrant, self.elasticsearch)

    def ingest_text(self, text: str, metadata: PolicyMetadata) -> tuple[str, list[PolicyChunk]]:
        document_id = uuid4().hex[:12]
        chunks = self.chunker.chunk_text(text, metadata, document_id)
        vectors = self.embeddings.embed([chunk.text for chunk in chunks]) if chunks else []
        self.qdrant.upsert(chunks, vectors)
        self.elasticsearch.index_chunks(chunks)
        return document_id, chunks

    def ingest_pdf(self, content: bytes, metadata: PolicyMetadata) -> tuple[str, list[PolicyChunk]]:
        return self.ingest_text(self.pdf.extract_text_from_bytes(content), metadata)

    def retrieve(self, question: str, filters: QueryFilters, top_k: int) -> list[PolicyChunk]:
        return self.hybrid.search(question, filters, top_k)


retrieval_service = RetrievalService()
