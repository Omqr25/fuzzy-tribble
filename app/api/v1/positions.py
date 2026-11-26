import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.core.security import verify_api_key
from app.schemas.position import PositionEmbedRequest, PositionEmbedResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/positions",
    tags=["Positions"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/embed", response_model=PositionEmbedResponse)
async def embed_position(
    payload: PositionEmbedRequest,
    request: Request,
) -> PositionEmbedResponse:
    embedding_service = getattr(request.app.state, "embedding_service", None)
    if embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model is not loaded.",
        )

    vectors = await run_in_threadpool(embedding_service.embed_position, payload)
    return PositionEmbedResponse(
        position_id=payload.position_id,
        vectors=vectors,
        model_version=embedding_service.model_version,
    )
