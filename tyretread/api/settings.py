"""API settings, resolved from the environment.

Nothing here has a secret default and nothing is read from a committed file. The
deployment targets differ (a laptop, Render, a container) and the only thing they
reliably share is an environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

__all__ = ["ApiSettings", "load_settings"]


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class ApiSettings:
    #: Identifier of the artifact to serve. Pinned by environment rather than
    #: "latest", so a deploy serves a known model and a rollback is a config change.
    model_id: str = "current"
    #: Hard cap on an upload, enforced while reading the stream rather than after, so
    #: an oversized body is rejected without ever being buffered in full.
    max_upload_bytes: int = 12 * 1024 * 1024
    #: Browsers send a phone photo as image/jpeg or image/heic; the decoder is the
    #: real authority, so this list is an early filter rather than a security control.
    allowed_content_types: tuple[str, ...] = (
        "image/jpeg", "image/png", "image/webp", "image/heic", "image/heif",
        "application/octet-stream",
    )
    #: Explicit origins. A wildcard is refused in production because the API is
    #: called from a known frontend and nothing else.
    cors_allow_origins: list[str] = field(default_factory=lambda: ["http://localhost:5173"])
    #: Seconds a single inspection may take before the request is abandoned.
    inspection_timeout_seconds: float = 20.0
    log_level: str = "INFO"
    environment: str = "development"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


def load_settings() -> ApiSettings:
    """Build settings from the environment, failing loudly on a bad configuration.

    A misconfigured CORS policy or upload limit should stop the process at start-up
    rather than surface as a confusing browser error later.
    """
    settings = ApiSettings(
        model_id=os.environ.get("TYRETREAD_MODEL_ID", "current"),
        max_upload_bytes=_env_int("TYRETREAD_MAX_UPLOAD_BYTES", 12 * 1024 * 1024),
        cors_allow_origins=_env_list("TYRETREAD_CORS_ORIGINS", ["http://localhost:5173"]),
        inspection_timeout_seconds=float(
            os.environ.get("TYRETREAD_INSPECTION_TIMEOUT", "20")
        ),
        log_level=os.environ.get("TYRETREAD_LOG_LEVEL", "INFO").upper(),
        environment=os.environ.get("TYRETREAD_ENV", "development"),
    )

    if settings.max_upload_bytes < 64 * 1024:
        raise ValueError("TYRETREAD_MAX_UPLOAD_BYTES is too small to hold a photograph")

    if settings.is_production and "*" in settings.cors_allow_origins:
        raise ValueError(
            "TYRETREAD_CORS_ORIGINS must list explicit origins in production; "
            "'*' would let any site call this API with a user's upload"
        )

    return settings
