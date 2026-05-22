from app.models.openai_models import Citation, PolicyChunk


def build_citations(chunks: list[PolicyChunk]) -> list[Citation]:
    citations: list[Citation] = []
    seen = set()
    for chunk in chunks:
        key = (chunk.metadata.source, chunk.metadata.clause_id, chunk.metadata.page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                source=chunk.metadata.source,
                title=chunk.metadata.title,
                framework=chunk.metadata.framework,
                category=chunk.metadata.category,
                clause_id=chunk.metadata.clause_id,
                page=chunk.metadata.page,
                snippet=chunk.text[:320],
                score=round(chunk.score, 4),
            )
        )
    return citations
