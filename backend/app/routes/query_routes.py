from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.openai_models import ComplianceAnswer, PolicyQueryRequest
from app.services.response_service import response_service
from app.utils.guardrails import validate_policy_query


router = APIRouter(prefix="/query", tags=["policy-query"])


@router.post("", response_model=ComplianceAnswer)
def ask_policy_question(request: PolicyQueryRequest) -> ComplianceAnswer:
    validate_policy_query(request.question)
    return response_service.answer(request)


@router.post("/stream")
def stream_policy_answer(request: PolicyQueryRequest) -> StreamingResponse:
    validate_policy_query(request.question)
    return StreamingResponse(
        response_service.answer_stream(request),
        media_type="application/x-ndjson",
    )
