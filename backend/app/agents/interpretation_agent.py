from app.models.openai_models import AgentTrace, PolicyChunk, PolicyQueryRequest


class PolicyInterpretationAgent:
    name = "Policy Interpretation Agent"

    def interpret(self, request: PolicyQueryRequest, chunks: list[PolicyChunk]) -> tuple[str, AgentTrace]:
        if not chunks:
            return (
                "No authoritative policy context was found for this question.",
                AgentTrace(agent=self.name, summary="Could not interpret policy because retrieval returned no context.", confidence=0.2),
            )

        clause_list = ", ".join(chunk.metadata.clause_id or chunk.id for chunk in chunks[:4])
        interpretation = (
            f"The most relevant clauses are {clause_list}. They should be read together because the question "
            "may touch operational policy, privacy, security, and reporting obligations."
        )
        return (
            interpretation,
            AgentTrace(agent=self.name, summary="Mapped the question to retrieved clauses and cross-framework obligations.", confidence=0.78),
        )
