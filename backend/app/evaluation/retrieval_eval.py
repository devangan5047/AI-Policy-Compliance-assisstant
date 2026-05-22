from app.models.openai_models import PolicyQueryRequest
from app.services.retrieval_service import retrieval_service


def evaluate_retrieval(samples: list[dict]) -> dict:
    scores = []
    for sample in samples:
        request = PolicyQueryRequest(question=sample["question"])
        chunks = retrieval_service.retrieve(request.question, request.filters, request.top_k)
        expected = set(sample.get("expected_sources", []))
        retrieved = {chunk.metadata.source for chunk in chunks}
        scores.append(len(expected & retrieved) / max(len(expected), 1))
    return {"metric": "source_recall", "score": sum(scores) / max(len(scores), 1), "samples": len(samples)}
