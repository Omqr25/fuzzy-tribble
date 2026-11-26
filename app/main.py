import logging
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from app.api.v1 import allocations, candidates, health, positions
from app.core.config import settings
from app.core.logging import request_id_ctx, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_loaded = False
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_ctx.set(req_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info("%s %s (%.2fms)", request.method, request.url.path, duration_ms)
        request_id_ctx.reset(token)

    response.headers["X-Request-ID"] = req_id
    return response

app.include_router(health.router)
app.include_router(positions.router)
app.include_router(candidates.router)
app.include_router(allocations.router)