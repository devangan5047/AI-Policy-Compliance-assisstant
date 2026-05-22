import json
import time
from collections.abc import Iterator

from app.agents.crew_manager import ComplianceCrewManager
from app.models.openai_models import AgentTrace, AnalyticsEvent, ComplianceAnswer, PolicyQueryRequest
from app.rag.citation_builder import build_citations
from app.services.retrieval_service import retrieval_service


class ResponseService:
    def __init__(self) -> None:
        self.crew = ComplianceCrewManager()
        self.analytics: list[AnalyticsEvent] = []

    def answer(self, request: PolicyQueryRequest) -> ComplianceAnswer:
        started = time.perf_counter()
        chunks = retrieval_service.retrieve(request.question, request.filters, request.top_k)
        result = self.crew.run(request, chunks)
        citations = build_citations(chunks)
        result.citations = citations
        result.retrieved_context_count = len(chunks)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result.token_usage.setdefault("latency_ms", elapsed_ms)

        self.analytics.append(
            AnalyticsEvent(
                question=request.question,
                status=result.compliance_status,
                risk_level=result.risk_level,
                framework=request.filters.framework,
                category=request.filters.category,
            )
        )
        return result

    def answer_stream(self, request: PolicyQueryRequest) -> Iterator[str]:
        started = time.perf_counter()
        chunks = retrieval_service.retrieve(request.question, request.filters, request.top_k)
        answer_parts: list[str] = []

        for text in self.crew.stream_answer_text(request, chunks):
            answer_parts.append(text)
            yield self._json_line({"type": "chunk", "text": text})

        answer_text = "".join(answer_parts).strip()
        status, score, status_trace = self.crew.compliance_agent.check(request, chunks)
        risk_level, escalation_required, risk_trace = self.crew.risk_agent.assess(status, chunks)
        recommendations, recommendation_trace = self.crew.recommendation_agent.recommend(request, status, risk_level, chunks)
        conflicts = self.crew._detect_conflicts(chunks)
        citations = build_citations(chunks)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        result = ComplianceAnswer(
            answer=answer_text,
            compliance_status=status,
            compliance_score=score,
            risk_level=risk_level,
            escalation_required=escalation_required,
            citations=citations,
            conflicts=conflicts,
            recommendations=recommendations,
            agent_trace=[
                self.crew.retrieval_agent.summarize(chunks),
                AgentTrace(agent="Streaming LLM", summary="Generated the response directly from retrieved document context.", confidence=0.9),
                status_trace,
                risk_trace,
                recommendation_trace,
            ],
            token_usage={"latency_ms": elapsed_ms},
            retrieved_context_count=len(chunks),
        )

        self.analytics.append(
            AnalyticsEvent(
                question=request.question,
                status=result.compliance_status,
                risk_level=result.risk_level,
                framework=request.filters.framework,
                category=request.filters.category,
            )
        )
        yield self._json_line({"type": "final", "answer": result.model_dump(mode="json")})

    def dashboard(self) -> dict:
        by_risk: dict[str, int] = {}
        by_status: dict[str, int] = {}
        common_terms: dict[str, int] = {}
        for event in self.analytics:
            by_risk[event.risk_level] = by_risk.get(event.risk_level, 0) + 1
            by_status[event.status] = by_status.get(event.status, 0) + 1
            for token in event.question.lower().split():
                if len(token) > 4:
                    common_terms[token.strip(".,?!")] = common_terms.get(token.strip(".,?!"), 0) + 1
        return {
            "total_queries": len(self.analytics),
            "risk_distribution": by_risk,
            "status_distribution": by_status,
            "common_query_terms": sorted(common_terms.items(), key=lambda item: item[1], reverse=True)[:10],
        }

    @staticmethod
    def _json_line(payload: dict) -> str:
        return json.dumps(payload) + "\n"


response_service = ResponseService()
