def context_precision(answer: str, contexts: list[str]) -> dict:
    if not contexts:
        return {"metric": "context_precision", "score": 0.0}
    answer_terms = set(answer.lower().split())
    overlaps = []
    for context in contexts:
        terms = set(context.lower().split())
        overlaps.append(len(answer_terms & terms) / max(len(answer_terms), 1))
    return {"metric": "context_precision", "score": sum(overlaps) / len(overlaps)}
