from app.models.openai_models import PolicyChunk, PolicyMetadata
from app.config.settings import get_settings


class ChunkingService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def chunk_text(self, text: str, metadata: PolicyMetadata, document_id: str) -> list[PolicyChunk]:
        clean = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        if not clean:
            return []

        size = self.settings.chunk_size
        overlap = min(self.settings.chunk_overlap, max(size // 3, 1))
        chunks: list[PolicyChunk] = []
        start = 0
        index = 0

        while start < len(clean):
            end = min(start + size, len(clean))
            if end < len(clean):
                boundary = max(clean.rfind(". ", start, end), clean.rfind("\n", start, end))
                if boundary > start + size // 2:
                    end = boundary + 1

            chunk_text = clean[start:end].strip()
            if chunk_text:
                chunk_meta = metadata.model_copy()
                chunk_meta.clause_id = chunk_meta.clause_id or f"{document_id}-{index + 1}"
                chunks.append(
                    PolicyChunk(
                        id=f"{document_id}:{index}",
                        text=chunk_text,
                        metadata=chunk_meta,
                    )
                )
                index += 1

            start = max(end - overlap, end) if end == len(clean) else max(end - overlap, 0)
            if end == len(clean):
                break

        return chunks
