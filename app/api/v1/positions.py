import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.core.security import verify_api_key
from app.schemas.position import PositionEmbedRequest, PositionEmbedResponse
from app.services.embedder import build_position_vectors

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/positions",
    tags=["Positions"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/embed", response_model=PositionEmbedResponse)
async def embed_position(
    payload: PositionEmbedRequest, request: Request
) -> PositionEmbedResponse:
    model = getattr(request.app.state, "embedding_model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not loaded; the service is not ready.",
        )

    try:
        vectors = await run_in_threadpool(build_position_vectors, payload, model)
    except Exception as exc:
        logger.exception(
            "Embedding generation failed for position %s",
            payload.position_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Position embedding generation failed.",
        ) from exc

    return PositionEmbedResponse(
        position_id=payload.position_id,
        vectors=vectors,
        model_version=model.model_version,
    )
