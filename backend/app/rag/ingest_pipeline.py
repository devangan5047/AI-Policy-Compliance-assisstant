from app.models.openai_models import PolicyMetadata
from app.services.retrieval_service import retrieval_service


def ingest_policy_text(text: str, metadata: PolicyMetadata):
    return retrieval_service.ingest_text(text, metadata)
