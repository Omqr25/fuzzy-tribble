from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import verify_api_key
from app.schemas.position import PositionEmbedRequest, PositionEmbedResponse

router = APIRouter(
    prefix="/api/v1/positions",
    tags=["Positions"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/embed", response_model=PositionEmbedResponse)
async def embed_position(payload: PositionEmbedRequest) -> PositionEmbedResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Position embedding service will be implemented in Phase 4.",
    )