"""FastAPI application.

Deliberately small. There is one interesting endpoint, and everything it does is
delegated to ``tyretread.inspect_image`` - the same function the CLI and the tests
call - so the API cannot drift from the evaluated pipeline.

The model is loaded once at start-up and held in application state. Loading it per
request would add unnecessary latency and, worse, would let a partially-written
artifact be picked up mid-deploy.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..evidence import render_evidence
from ..imaging.io import decode_image_bytes
from ..inspect import inspect_image
from ..models.artifact import ModelArtifact, list_artifacts, load_artifact
from .schemas import ErrorOut, HealthOut, InspectionOut, ModelMetricsOut, ModelOut
from .settings import ApiSettings, load_settings

logger = logging.getLogger("tyretread.api")

__all__ = ["create_app"]


class ApiError(Exception):
    """An error with a stable code, so the frontend can branch on it."""

    def __init__(self, status: int, code: str, message: str, detail: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing an oversized body without buffering all of it.

    ``UploadFile.read()`` with no argument would happily materialise a 2 GB body in
    memory before anyone checked its size. Reading in chunks and stopping at the limit
    means a hostile or accidental large upload costs one chunk over the limit.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1 << 20):
        total += len(chunk)
        if total > limit:
            raise ApiError(
                413, "file_too_large",
                f"Image exceeds the {limit // (1024 * 1024)} MB limit.",
                {"limit_bytes": limit},
            )
        chunks.append(chunk)

    if total == 0:
        raise ApiError(400, "empty_upload", "No image data was received.")
    return b"".join(chunks)


def get_artifact(request: Request) -> ModelArtifact:
    artifact = getattr(request.app.state, "artifact", None)
    if artifact is None:
        raise ApiError(
            503, "model_unavailable",
            "The analysis model is not loaded, so no inspection can be performed.",
            {"available": list_artifacts()},
        )
    return artifact


def get_settings(request: Request) -> ApiSettings:
    return request.app.state.settings


def _summarise_metrics(metrics: dict | None) -> ModelMetricsOut | None:
    """Reduce the artifact's full metrics record to the fields a client needs."""
    if not metrics:
        return None
    winner = (metrics.get("winner") or {}).get("metrics") or {}
    calibration = metrics.get("calibration") or {}
    threshold = metrics.get("threshold_choice") or {}

    def mean_of(key: str) -> float | None:
        entry = winner.get(key)
        return entry.get("mean") if isinstance(entry, dict) else None

    def std_of(key: str) -> float | None:
        entry = winner.get(key)
        return entry.get("std") if isinstance(entry, dict) else None

    return ModelMetricsOut(
        balanced_accuracy=mean_of("balanced_accuracy"),
        balanced_accuracy_std=std_of("balanced_accuracy"),
        roc_auc=mean_of("roc_auc"),
        brier_score=calibration.get("brier_score"),
        expected_calibration_error=calibration.get("expected_calibration_error"),
        abstention_rate=threshold.get("abstention_rate"),
        balanced_accuracy_on_decided=threshold.get("balanced_accuracy_on_decided"),
        quality_gate_pass_rate=metrics.get("quality_gate_pass_rate"),
    )


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(
        level=settings.log_level,
        format='{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.artifact = None
        app.state.model_error = None
        try:
            app.state.artifact = load_artifact(settings.model_id)
            logger.info("loaded model %s", settings.model_id)
        except Exception as exc:
            # Start anyway, reporting degraded health. A backend that refuses to boot
            # without a model is harder to diagnose than one that says what is wrong.
            app.state.model_error = str(exc)
            logger.error("could not load model %s: %s", settings.model_id, exc)
        yield

    app = FastAPI(
        title="Tyre tread screening API",
        version=__version__,
        description=(
            "Assistive visual screening of tyre tread condition from a photograph. "
            "Not a certified inspection: see the disclaimer on every response."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=False,  # no cookies or sessions, so none are needed
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        """Tag every request, so a user-reported failure is findable in the logs."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %s in %.0fms (request_id=%s)",
            request.method, request.url.path, response.status_code, elapsed, request_id,
        )
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status,
            content=ErrorOut(error=exc.code, message=exc.message, detail=exc.detail).model_dump(),
        )

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    async def health(request: Request) -> HealthOut:
        """Liveness and readiness in one response.

        Reports ``degraded`` rather than failing when the model is missing, so a
        platform health check distinguishes "process is up but cannot serve" from
        "process is down".
        """
        artifact = getattr(request.app.state, "artifact", None)
        return HealthOut(
            status="ok" if artifact else "degraded",
            version=__version__,
            model_loaded=artifact is not None,
            model_id=artifact.metadata.model_id if artifact else None,
            detail=getattr(request.app.state, "model_error", None),
        )

    @app.get("/v1/model", response_model=ModelOut, tags=["meta"])
    async def model_info(
        artifact: Annotated[ModelArtifact, Depends(get_artifact)],
    ) -> ModelOut:
        """What is actually being served, including how it was validated."""
        meta = artifact.metadata
        return ModelOut(
            id=meta.model_id,
            artifact_format_version=meta.format_version,
            trained_on=meta.training_dataset,
            n_training_samples=meta.n_training_samples,
            validation=meta.validation,
            metrics=_summarise_metrics(meta.metrics),
        )

    @app.post("/v1/inspect", response_model=InspectionOut, tags=["inspection"])
    async def inspect(
        request: Request,
        artifact: Annotated[ModelArtifact, Depends(get_artifact)],
        settings: Annotated[ApiSettings, Depends(get_settings)],
        image: Annotated[UploadFile, File(description="A photograph of the tyre tread.")],
        include_evidence: Annotated[bool, Query(
            description="Return base64 images of the intermediate analysis stages."
        )] = True,
        include_features: Annotated[bool, Query(
            description="Return every raw feature value. For debugging and research."
        )] = False,
    ) -> InspectionOut:
        """Assess one photograph.

        A refused image is a **200 with an ``unable_to_assess`` verdict**, not an
        error status. The request succeeded and the answer is "this photograph cannot
        be assessed, here is why and here is what to do" - which is a result the
        client must render, not an exception it should handle.
        """
        if image.content_type and image.content_type not in settings.allowed_content_types:
            raise ApiError(
                415, "unsupported_media_type",
                f"Content type {image.content_type} is not supported.",
                {"supported": list(settings.allowed_content_types)},
            )

        payload = await _read_upload(image, settings.max_upload_bytes)

        try:
            loaded = decode_image_bytes(payload)
        except ValueError as exc:
            raise ApiError(
                400, "undecodable_image",
                "The uploaded file could not be read as an image.",
                {"reason": str(exc)},
            ) from exc

        started = time.perf_counter()
        try:
            # Feature extraction is CPU-bound, so it runs in a worker thread to avoid
            # blocking the event loop, with a timeout so one pathological image cannot
            # occupy a worker indefinitely.
            inspection = await asyncio.wait_for(
                asyncio.to_thread(inspect_image, loaded.bgr, artifact),
                timeout=settings.inspection_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            raise ApiError(
                504, "inspection_timeout",
                "Analysis took too long and was abandoned. Please try again.",
                {"timeout_seconds": settings.inspection_timeout_seconds},
            ) from exc
        except Exception as exc:
            logger.exception("inspection failed")
            raise ApiError(
                500, "inspection_failed",
                "Analysis failed unexpectedly.",
                {"reason": type(exc).__name__},
            ) from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        payload_out = inspection.as_dict(include_features=include_features)
        payload_out["elapsed_ms"] = round(elapsed_ms, 1)

        meta = artifact.metadata
        payload_out["model"] = {
            "id": meta.model_id,
            "artifact_format_version": meta.format_version,
            "trained_on": meta.training_dataset,
            "n_training_samples": meta.n_training_samples,
            "validation": meta.validation,
            "metrics": _summarise_metrics(meta.metrics),
        }

        if include_evidence:
            payload_out["evidence"] = render_evidence(loaded.bgr, inspection.extraction)

        return InspectionOut(**payload_out)

    return app
