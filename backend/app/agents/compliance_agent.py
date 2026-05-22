from app.models.openai_models import AgentTrace, PolicyChunk, PolicyQueryRequest


class ComplianceCheckerAgent:
    name = "Compliance Checker Agent"

    def check(self, request: PolicyQueryRequest, chunks: list[PolicyChunk]) -> tuple[str, int, AgentTrace]:
        if not chunks:
            return "insufficient_context", 25, AgentTrace(agent=self.name, summary="Insufficient source material for a compliance decision.", confidence=0.25)

        combined = " ".join(chunk.text.lower() for chunk in chunks)
        question = request.question.lower()
        prohibited_terms = ("must not", "prohibited", "forbidden", "not allowed", "unauthorized", "breach")
        conditional_terms = ("approval", "manager", "legal", "dpo", "security", "report", "notify", "assessment")

        if any(term in combined for term in prohibited_terms):
            status = "non_compliant"
            score = 35
        elif any(term in combined for term in conditional_terms) or any(term in question for term in ("accidentally", "another country", "different regions")):
            status = "needs_review"
            score = 62
        else:
            status = "compliant"
            score = 82

        return (
            status,
            score,
            AgentTrace(agent=self.name, summary=f"Estimated action status as {status} with score {score}/100.", confidence=0.68),
        )
