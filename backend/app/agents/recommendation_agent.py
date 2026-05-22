from app.models.openai_models import AgentTrace, PolicyChunk, PolicyQueryRequest


class RecommendationAgent:
    name = "Recommendation Agent"

    def recommend(self, request: PolicyQueryRequest, status: str, risk_level: str, chunks: list[PolicyChunk]) -> tuple[list[str], AgentTrace]:
        recommendations = []
        if status in {"needs_review", "non_compliant", "insufficient_context"}:
            recommendations.append("Pause the action until the relevant owner confirms the policy interpretation.")
        if risk_level in {"high", "critical"}:
            recommendations.append("Escalate to Legal, Privacy, Security, or HR according to the policy owner named in the source document.")
        if "data" in request.question.lower() or any("data" in chunk.text.lower() for chunk in chunks):
            recommendations.append("Share the minimum necessary data and verify region, purpose, access, and retention restrictions first.")
        if "remote" in request.question.lower() or "travel" in request.question.lower():
            recommendations.append("Confirm work location approval, tax or immigration constraints, and secure access requirements before traveling.")
        if not recommendations:
            recommendations.append("Proceed only within the cited policy boundaries and keep a record of the decision.")

        return (
            recommendations,
            AgentTrace(agent=self.name, summary="Generated compliant alternatives and escalation guidance.", confidence=0.76),
        )
