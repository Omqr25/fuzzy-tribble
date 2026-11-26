from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness_check(request: Request) -> JSONResponse:
    models_loaded = bool(getattr(request.app.state, "models_loaded", False))
    if not models_loaded:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "booting", "models_loaded": False},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready", "models_loaded": True},
    )