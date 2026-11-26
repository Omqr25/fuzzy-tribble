import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.core.security import verify_api_key
from app.schemas.candidate import CandidateIngestResponse, CandidateMetadata
from app.services.embedder import build_candidate_vectors
from app.services.extractor import extract_structured_cv
from app.services.parser import (
    CVParsingError,
    CVValidationError,
    count_non_pii_text_characters,
    parse_cv,
    scrub_pii,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/candidates",
    tags=["Candidates"],
    dependencies=[Depends(verify_api_key)],
)

NO_EMBEDDING_MODEL_VERSION = "no_model_loaded"


@router.post("/ingest", response_model=CandidateIngestResponse)
async def ingest_candidate(
    request: Request,
    cv_file: UploadFile = File(..., description="Candidate CV (.pdf or .docx)"),
    metadata: str = Form(..., description="JSON string matching CandidateMetadata"),
) -> CandidateIngestResponse:
    try:
        parsed_metadata = CandidateMetadata.model_validate_json(metadata)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(),
        ) from exc

    try:
        content = await cv_file.read(settings.CV_MAX_FILE_SIZE_BYTES + 1)
    finally:
        await cv_file.close()

    try:
        parsed_document = await run_in_threadpool(
            parse_cv,
            content,
            cv_file.filename or "",
        )
    except CVValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    except CVParsingError as exc:
        # A correctly identified and hashed file that fails during document
        # conversion is represented explicitly in the ingestion contract.
        logger.error("CV parsing failed for hash %s: %s", exc.cv_hash, str(exc))
        return CandidateIngestResponse(
            candidate_id=parsed_metadata.candidate_id,
            cv_hash=exc.cv_hash,
            extraction_status="failed",
            extraction_method="none",
            extraction_error=str(exc),
            parsed_cv=None,
            vectors=None,
            model_version=NO_EMBEDDING_MODEL_VERSION,
        )

    scrubbed_markdown = scrub_pii(parsed_document.markdown)
    non_whitespace_chars = count_non_pii_text_characters(scrubbed_markdown)
    if non_whitespace_chars < settings.CV_MIN_EXTRACTED_TEXT_CHARS:
        return CandidateIngestResponse(
            candidate_id=parsed_metadata.candidate_id,
            cv_hash=parsed_document.cv_hash,
            extraction_status="needs_ocr",
            extraction_method="none",
            extraction_error=(
                "Extracted text contains fewer than "
                f"{settings.CV_MIN_EXTRACTED_TEXT_CHARS} non-whitespace characters; OCR is required."
            ),
            parsed_cv=None,
            vectors=None,
            model_version=NO_EMBEDDING_MODEL_VERSION,
        )

    provider = getattr(request.app.state, "cv_extraction_provider", None)
    try:
        extraction = await extract_structured_cv(
            scrubbed_markdown,
            provider=provider,
        )
    except Exception:
        # Do not silently hide a failure in both the configured extractor and
        # deterministic fallback. Avoid echoing document contents in errors.
        logger.exception(
            "CV structured extraction and fallback both failed for hash %s",
            parsed_document.cv_hash,
        )
        return CandidateIngestResponse(
            candidate_id=parsed_metadata.candidate_id,
            cv_hash=parsed_document.cv_hash,
            extraction_status="failed",
            extraction_method="none",
            extraction_error="Structured extraction and deterministic fallback both failed.",
            parsed_cv=None,
            vectors=None,
            model_version=NO_EMBEDDING_MODEL_VERSION,
        )

    # Phase 4: generate faceted embeddings when the model is available.
    embedding_model = getattr(request.app.state, "embedding_model", None)
    vectors = None
    model_version = NO_EMBEDDING_MODEL_VERSION

    if embedding_model is not None:
        try:
            vectors = await run_in_threadpool(
                build_candidate_vectors,
                parsed_metadata,
                extraction.parsed_cv,
                embedding_model,
            )
            model_version = embedding_model.model_version
        except Exception:
            logger.exception(
                "Embedding generation failed for candidate %s",
                parsed_metadata.candidate_id,
            )

    return CandidateIngestResponse(
        candidate_id=parsed_metadata.candidate_id,
        cv_hash=parsed_document.cv_hash,
        extraction_status="completed",
        extraction_method=extraction.extraction_method,
        extraction_error=extraction.extraction_error,
        parsed_cv=extraction.parsed_cv,
        vectors=vectors,
        model_version=model_version,
    )
