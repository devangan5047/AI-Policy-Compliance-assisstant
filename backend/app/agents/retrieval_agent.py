from app.models.openai_models import AgentTrace, PolicyChunk


class PolicyRetrievalAgent:
    name = "Policy Retrieval Agent"

    def summarize(self, chunks: list[PolicyChunk]) -> AgentTrace:
        frameworks = sorted({chunk.metadata.framework for chunk in chunks})
        return AgentTrace(
            agent=self.name,
            summary=f"Retrieved {len(chunks)} relevant policy sections across {', '.join(frameworks) or 'no frameworks'}.",
            confidence=0.85 if chunks else 0.15,
        )
