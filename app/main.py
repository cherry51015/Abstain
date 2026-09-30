"""
FastAPI application factory.

`create_app()` wires settings -> catalog -> win model -> (optional) LLM
client -> service -> database. Everything is injectable so tests can run the
real HTTP stack against SQLite with a fake LLM.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.routes import router
from app.catalog import UnknownReferenceError, load_catalog
from app.config import Settings
from app.db import Base, build_engine, build_session_factory
from app.engine.decision_engine import DecisionEngine
from app.extraction.cache import ResponseCache
from app.extraction.llm_client import OpenAICompatibleClient
from app.extraction.llm_extractor import LLMExtractor
from app.observability import RequestContextMiddleware, configure_logging, request_id_var
from app.repository import ConflictError, NotFoundError
from app.scoring.win_model import WinModel
from app.service import DisputeService

logger = logging.getLogger("abstain.api")


def build_service(settings: Settings) -> tuple[DisputeService, OpenAICompatibleClient | None]:
    client = None
    extractor = None
    if settings.llm_api_key:
        client = OpenAICompatibleClient(
            api_key=settings.llm_api_key, model=settings.llm_model, base_url=settings.llm_base_url,
            max_concurrency=settings.llm_max_concurrency, max_requests_per_minute=settings.llm_requests_per_minute,
            extra_body=settings.llm_extra_body,
            cache=ResponseCache(settings.llm_cache_path) if settings.llm_cache_path else None,
        )
        extractor = LLMExtractor(client, n_samples=settings.llm_samples, max_tokens=settings.llm_max_tokens)
    service = DisputeService(catalog=load_catalog(), model=WinModel.load(), engine=DecisionEngine(),
                             llm=extractor, mode=settings.extraction_mode, llm_budget_s=settings.llm_budget_s)
    return service, client


def create_app(settings: Settings | None = None, *, service: DisputeService | None = None,
               create_schema: bool = False) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        client = None
        if service is None:
            app.state.service, client = build_service(settings)
        else:
            app.state.service = service
        engine = build_engine(settings.database_url)
        if create_schema:  # tests and throwaway local runs; deployments use `alembic upgrade head`
            Base.metadata.create_all(engine)
        app.state.session_factory = build_session_factory(engine)
        logger.info("startup", extra={"mode": app.state.service.mode, "model_version": app.state.service.model.version})
        yield
        if client:
            await client.aclose()
        engine.dispose()

    app = FastAPI(title="Abstain", version="1.0.0", lifespan=lifespan,
                  description="Chargeback decision engine: LLM fact extraction, calibrated P(win), "
                              "value-of-information escalation.")
    app.state.settings = settings
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-API-Key", "Idempotency-Key", "X-Request-ID"],
                       expose_headers=["X-Request-ID"])
    app.include_router(router)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    def error(status: int, detail: str) -> JSONResponse:
        return JSONResponse(status_code=status, content={"detail": detail, "request_id": request_id_var.get()})

    @app.exception_handler(UnknownReferenceError)
    async def _unknown_ref(_: Request, exc: UnknownReferenceError):
        return error(422, str(exc))

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError):
        return error(404, str(exc))

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError):
        return error(409, str(exc))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled error")
        return error(500, "Internal error")

    return app


app = create_app()
