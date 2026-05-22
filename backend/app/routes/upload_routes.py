from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.openai_models import BatchUploadResponse, PolicyMetadata, UploadResponse
from app.services.retrieval_service import retrieval_service


router = APIRouter(prefix="/policies", tags=["policy-ingestion"])


@router.post("/upload", response_model=UploadResponse)
async def upload_policy(
    file: UploadFile = File(...),
    category: str = Form("internal"),
    framework: str = Form("internal"),
    source: str | None = Form(None),
    title: str | None = Form(None),
    authority: int = Form(5),
) -> UploadResponse:
    content = await file.read()
    metadata = PolicyMetadata(
        source=source or file.filename or "uploaded-policy",
        title=title or file.filename,
        category=category,
        framework=framework,
        authority=authority,
    )
    return _ingest_uploaded_content(file.filename or "uploaded-policy", content, metadata)


@router.post("/upload-batch", response_model=BatchUploadResponse)
async def upload_policies(
    files: list[UploadFile] = File(...),
    category: str = Form("internal"),
    framework: str = Form("internal"),
    authority: int = Form(5),
) -> BatchUploadResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one PDF, TXT, or Markdown document.")

    documents: list[UploadResponse] = []
    for file in files:
        filename = file.filename or "uploaded-document"
        metadata = PolicyMetadata(
            source=filename,
            title=filename,
            category=category,
            framework=framework,
            authority=authority,
        )
        documents.append(_ingest_uploaded_content(filename, await file.read(), metadata))

    return BatchUploadResponse(
        documents=documents,
        total_documents=len(documents),
        total_chunks_indexed=sum(document.chunks_indexed for document in documents),
    )


@router.post("/ingest-text", response_model=UploadResponse)
def ingest_policy_text(
    text: str = Form(...),
    category: str = Form("internal"),
    framework: str = Form("internal"),
    source: str = Form("manual-entry"),
    title: str | None = Form(None),
    authority: int = Form(5),
) -> UploadResponse:
    metadata = PolicyMetadata(
        source=source,
        title=title,
        category=category,
        framework=framework,
        authority=authority,
    )
    document_id, chunks = retrieval_service.ingest_text(text, metadata)
    return UploadResponse(document_id=document_id, chunks_indexed=len(chunks), metadata=metadata)


def _ingest_uploaded_content(filename: str, content: bytes, metadata: PolicyMetadata) -> UploadResponse:
    normalized_filename = filename.lower()
    if normalized_filename.endswith(".pdf"):
        document_id, chunks = retrieval_service.ingest_pdf(content, metadata)
    elif normalized_filename.endswith((".txt", ".md")):
        document_id, chunks = retrieval_service.ingest_text(content.decode("utf-8", errors="ignore"), metadata)
    else:
        raise HTTPException(status_code=400, detail=f"{filename} is not supported. Upload PDF, TXT, or Markdown files.")

    return UploadResponse(document_id=document_id, chunks_indexed=len(chunks), metadata=metadata)
