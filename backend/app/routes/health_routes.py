from fastapi import APIRouter


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict:
    return {"status": "ok", "service": "policy-compliance-assistant"}
