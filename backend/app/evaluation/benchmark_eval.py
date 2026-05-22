import time

from app.models.openai_models import BenchmarkResult, PolicyQueryRequest, QueryFilters
from app.services.response_service import response_service


def benchmark_framework(framework: str, questions: list[str]) -> BenchmarkResult:
    latencies = []
    context_counts = []
    for question in questions:
        started = time.perf_counter()
        result = response_service.answer(
            PolicyQueryRequest(question=question, filters=QueryFilters(framework=framework))
        )
        latencies.append((time.perf_counter() - started) * 1000)
        context_counts.append(result.retrieved_context_count)
    return BenchmarkResult(
        framework=framework,
        query_count=len(questions),
        average_latency_ms=sum(latencies) / max(len(latencies), 1),
        average_context_count=sum(context_counts) / max(len(context_counts), 1),
    )
