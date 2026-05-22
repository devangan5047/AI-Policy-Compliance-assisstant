def llm_as_judge_placeholder(answer: str, citations: list[dict]) -> dict:
    cited = bool(citations)
    hedged = any(term in answer.lower() for term in ("policy", "clause", "source", "context", "escalate"))
    return {
        "metric": "compliance_interpretation_quality",
        "score": 0.8 if cited and hedged else 0.45,
        "notes": "Install/configure DeepEval to replace this heuristic with an LLM-as-judge metric.",
    }
