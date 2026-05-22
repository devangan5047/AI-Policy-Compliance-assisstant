from app.models.openai_models import AgentTrace, PolicyChunk


class RiskAssessmentAgent:
    name = "Risk Assessment Agent"

    def assess(self, status: str, chunks: list[PolicyChunk]) -> tuple[str, bool, AgentTrace]:
        text = " ".join(chunk.text.lower() for chunk in chunks)
        high_risk_terms = ("personal data", "customer data", "cross-border", "gdpr", "credentials", "incident", "breach")
        critical_terms = ("regulator", "special category", "health data", "financial data", "unauthorized disclosure")

        if status == "insufficient_context":
            risk, escalation = "medium", True
        elif any(term in text for term in critical_terms):
            risk, escalation = "critical", True
        elif status == "non_compliant" or any(term in text for term in high_risk_terms):
            risk, escalation = "high", True
        elif status == "needs_review":
            risk, escalation = "medium", True
        else:
            risk, escalation = "low", False

        return (
            risk,
            escalation,
            AgentTrace(agent=self.name, summary=f"Risk evaluated as {risk}; escalation required: {escalation}.", confidence=0.72),
        )
