from fastapi import HTTPException


BLOCKED_PATTERNS = (
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "jailbreak",
)

def validate_policy_query(question: str) -> None:
    normalized = " ".join(question.lower().split())
    if len(normalized) < 8:
        raise HTTPException(status_code=400, detail="Ask a complete question about the uploaded documents.")
    if any(pattern in normalized for pattern in BLOCKED_PATTERNS):
        raise HTTPException(status_code=400, detail="Prompt-injection style requests are not allowed.")
