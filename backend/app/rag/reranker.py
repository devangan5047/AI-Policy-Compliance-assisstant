from app.models.openai_models import PolicyChunk


class AuthorityReranker:
    def rerank(self, chunks: list[PolicyChunk], top_k: int) -> list[PolicyChunk]:
        reranked = []
        for chunk in chunks:
            authority_bonus = chunk.metadata.authority / 100
            framework_bonus = 0.05 if chunk.metadata.framework.lower() in {"gdpr", "nist"} else 0
            reranked.append(chunk.model_copy(update={"score": chunk.score + authority_bonus + framework_bonus}))
        return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
