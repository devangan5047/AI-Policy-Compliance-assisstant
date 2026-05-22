from fastapi import APIRouter

from app.models.openai_models import ComplianceAnswer, PolicyQueryRequest
from app.services.response_service import response_service
from app.utils.guardrails import validate_policy_query


router = APIRouter(prefix="/compliance", tags=["compliance"])


@router.post("/check", response_model=ComplianceAnswer)
def check_compliance(request: PolicyQueryRequest) -> ComplianceAnswer:
    validate_policy_query(request.question)
    return response_service.answer(request)


@router.get("/analytics")
def compliance_analytics() -> dict:
    return response_service.dashboard()
