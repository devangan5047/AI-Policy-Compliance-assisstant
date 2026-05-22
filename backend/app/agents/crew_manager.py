import json
import re

from app.agents.compliance_agent import ComplianceCheckerAgent
from app.agents.interpretation_agent import PolicyInterpretationAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.retrieval_agent import PolicyRetrievalAgent
from app.agents.risk_assessment_agent import RiskAssessmentAgent
from app.models.llm_config import OpenRouterClient
from app.models.openai_models import AgentTrace, ComplianceAnswer, PolicyChunk, PolicyQueryRequest
from app.rag.prompt_templates import ANSWER_PROMPT, COMPLIANCE_SYSTEM_PROMPT, STRUCTURED_ANSWER_PROMPT


class ComplianceCrewManager:
    def __init__(self) -> None:
        self.retrieval_agent = PolicyRetrievalAgent()
        self.interpretation_agent = PolicyInterpretationAgent()
        self.compliance_agent = ComplianceCheckerAgent()
        self.risk_agent = RiskAssessmentAgent()
        self.recommendation_agent = RecommendationAgent()
        self.llm = OpenRouterClient()

    def run(self, request: PolicyQueryRequest, chunks: list[PolicyChunk]) -> ComplianceAnswer:
        traces: list[AgentTrace] = [self.retrieval_agent.summarize(chunks)]
        interpretation, trace = self.interpretation_agent.interpret(request, chunks)
        traces.append(trace)
        status, score, trace = self.compliance_agent.check(request, chunks)
        traces.append(trace)
        risk_level, escalation_required, trace = self.risk_agent.assess(status, chunks)
        traces.append(trace)
        recommendations, trace = self.recommendation_agent.recommend(request, status, risk_level, chunks)
        traces.append(trace)

        conflicts = self._detect_conflicts(chunks)
        answer, token_usage, llm_assessment = self._compose_answer(
            request=request,
            chunks=chunks,
            interpretation=interpretation,
            status=status,
            score=score,
            risk_level=risk_level,
            escalation_required=escalation_required,
            recommendations=recommendations,
            conflicts=conflicts,
        )
        if llm_assessment:
            status = llm_assessment["compliance_status"]
            score = llm_assessment["compliance_score"]
            risk_level = llm_assessment["risk_level"]
            escalation_required = llm_assessment["escalation_required"]

        return ComplianceAnswer(
            answer=answer,
            compliance_status=status,
            compliance_score=score,
            risk_level=risk_level,
            escalation_required=escalation_required,
            citations=[],
            conflicts=conflicts,
            recommendations=recommendations,
            agent_trace=traces,
            token_usage=token_usage,
        )

    def stream_answer_text(self, request: PolicyQueryRequest, chunks: list[PolicyChunk]):
        prompt = self._answer_prompt(request, chunks)
        answer, _ = self.llm.generate(prompt)
        yield answer

    def _compose_answer(
        self,
        request: PolicyQueryRequest,
        chunks: list[PolicyChunk],
        interpretation: str,
        status: str,
        score: int,
        risk_level: str,
        escalation_required: bool,
        recommendations: list[str],
        conflicts: list[str],
    ) -> tuple[str, dict[str, int], dict | None]:
        prompt = (
            COMPLIANCE_SYSTEM_PROMPT
            + "\n\n"
            + STRUCTURED_ANSWER_PROMPT.format(
                question=request.question,
                employee_context=request.employee_context or "Not provided",
                context=self._format_context(chunks) or "No context retrieved",
            )
        )
        llm_text, usage = self.llm.generate(prompt)
        parsed = self._parse_llm_answer(llm_text)
        if parsed:
            recommendations.clear()
            recommendations.extend(parsed.get("recommendations", []))
            conflicts.clear()
            conflicts.extend(parsed.get("conflicts", []))
            return parsed["answer"], usage, parsed

        return llm_text, usage, None

    def _answer_prompt(self, request: PolicyQueryRequest, chunks: list[PolicyChunk]) -> str:
        return (
            COMPLIANCE_SYSTEM_PROMPT
            + "\n\n"
            + ANSWER_PROMPT.format(
                question=request.question,
                employee_context=request.employee_context or "Not provided",
                context=self._format_context(chunks) or "No context retrieved",
            )
        )

    def _format_context(self, chunks: list[PolicyChunk]) -> str:
        return "\n\n".join(
            (
                f"[{chunk.metadata.title or chunk.metadata.source} | "
                f"{chunk.metadata.framework} | {chunk.metadata.clause_id or chunk.id}] {chunk.text}"
            )
            for chunk in chunks[:10]
        )

    def _parse_llm_answer(self, llm_text: str | None) -> dict | None:
        if not llm_text:
            return None
        cleaned = llm_text.strip()
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        statuses = {"compliant", "non_compliant", "needs_review", "insufficient_context"}
        risks = {"low", "medium", "high", "critical"}
        if parsed.get("compliance_status") not in statuses or parsed.get("risk_level") not in risks:
            return None
        parsed["answer"] = str(parsed.get("answer") or "").strip()
        if not parsed["answer"]:
            return None
        parsed["recommendations"] = [str(item) for item in parsed.get("recommendations", []) if str(item).strip()]
        parsed["conflicts"] = [str(item) for item in parsed.get("conflicts", []) if str(item).strip()]
        try:
            parsed["compliance_score"] = max(0, min(100, int(parsed.get("compliance_score", 0))))
        except (TypeError, ValueError):
            return None
        parsed["escalation_required"] = bool(parsed.get("escalation_required"))
        return parsed

    def _detect_conflicts(self, chunks: list[PolicyChunk]) -> list[str]:
        conflicts = []
        allow_chunks = [chunk for chunk in chunks if any(term in chunk.text.lower() for term in ("allowed", "permitted", "may"))]
        deny_chunks = [chunk for chunk in chunks if any(term in chunk.text.lower() for term in ("must not", "prohibited", "not allowed"))]
        if allow_chunks and deny_chunks:
            conflicts.append(
                "Retrieved sources include both permissive and restrictive language; a policy owner should resolve precedence."
            )
        frameworks = {chunk.metadata.framework.lower() for chunk in chunks}
        if "gdpr" in frameworks and "internal" in frameworks:
            conflicts.append("Internal policy should be checked against GDPR obligations where personal data is involved.")
        return conflicts
