import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import time
import uuid

from fastapi import FastAPI, Request
from starlette.concurrency import run_in_threadpool

from app.api.v1 import allocations, candidates, health, positions
from app.core.config import settings
from app.core.logging import request_id_ctx, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def _load_embedding_model_into_state(app: FastAPI, model_name: str) -> None:
    """Load the embedding model without blocking server startup (/ready stays 503 while booting)."""
    try:
        from app.services.embedder import EmbeddingModel

        model = await run_in_threadpool(EmbeddingModel, model_name)
        app.state.embedding_model = model
        app.state.embedding_model_version = model.model_version
        app.state.models_loaded = True
        logger.info(
            "Embedding model loaded: %s (dim=%d)",
            model.model_version,
            model.dimension,
        )
    except asyncio.CancelledError:
        logger.info("Embedding model loading cancelled during shutdown")
        raise
    except Exception:
        logger.exception(
            "Failed to load embedding model '%s'; the service will run "
            "without embedding capabilities",
            model_name,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models_loaded = False
    app.state.embedding_model = None
    app.state.embedding_model_version = "none"
    load_task: asyncio.Task[None] | None = None

    if settings.EMBEDDING_MODEL:
        load_task = asyncio.create_task(
            _load_embedding_model_into_state(app, settings.EMBEDDING_MODEL)
        )
        app.state.model_load_task = load_task

    try:
        yield
    finally:
        if load_task is not None and not load_task.done():
            load_task.cancel()
            with suppress(asyncio.CancelledError):
                await load_task


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
