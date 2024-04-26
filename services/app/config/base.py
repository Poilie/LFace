from __future__ import annotations

import binascii
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from litestar.serialization import decode_json, encode_json
from redis.asyncio import Redis
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.utils import module_to_os_path, slugify

if TYPE_CHECKING:
    from litestar.data_extractors import RequestExtractorField, ResponseExtractorField

DEFAULT_MODULE_NAME = "app"
BASE_DIR: Final[Path] = module_to_os_path(DEFAULT_MODULE_NAME)

TRUE_VALUES = {"True", "true", "1", "yes", "Y", "T"}


@dataclass
class ServerSettings:
    """Server configurations."""

    APP_LOC: str = "app.asgi:app"
    """Path to app executable, or factory."""
    APP_LOC_IS_FACTORY: bool = False
    """Indicate if APP_LOC points to an executable or factory."""
    HOST: str = field(default_factory=lambda: os.getenv("LITESTAR_HOST", "0.0.0.0"))  # noqa: S104
    """Server network host."""
    PORT: int = field(default_factory=lambda: int(
        os.getenv("LITESTAR_PORT", "8000")))
    """Server port."""
    KEEPALIVE: int = field(default_factory=lambda: int(
        os.getenv("LITESTAR_KEEPALIVE", "65")))
    """Seconds to hold connections open (65 is > AWS lb idle timeout)."""
    RELOAD: bool = field(
        default_factory=lambda: os.getenv(
            "LITESTAR_RELOAD", "False") in TRUE_VALUES,
    )
    """Turn on hot reloading."""
    RELOAD_DIRS: list[str] = field(default_factory=lambda: [f"{BASE_DIR}"])
    """Directories to watch for reloading."""
    HTTP_WORKERS: int | None = field(
        default_factory=lambda: int(os.getenv("WEB_CONCURRENCY")) if os.getenv(
            "WEB_CONCURRENCY") is not None else None,  # type: ignore[arg-type]
    )
    """Number of HTTP Worker processes to be spawned by Uvicorn."""


@dataclass
class SaqSettings:
    """Server configurations."""

    PROCESSES: int = field(default_factory=lambda: int(
        os.getenv("SAQ_PROCESSES", "1")))
    """The number of worker processes to start.

    Default is set to 1.
    """
    CONCURRENCY: int = field(default_factory=lambda: int(
        os.getenv("SAQ_CONCURRENCY", "10")))
    """The number of concurrent jobs allowed to execute per worker process.

    Default is set to 10.
    """
    WEB_ENABLED: bool = field(
        default_factory=lambda: os.getenv(
            "SAQ_WEB_ENABLED", "True") in TRUE_VALUES,
    )
    """If true, the worker admin UI is hosted on worker startup."""
    USE_SERVER_LIFESPAN: bool = field(
        default_factory=lambda: os.getenv(
            "SAQ_USE_SERVER_LIFESPAN", "False") in TRUE_VALUES,
    )
    """Auto start and stop `saq` processes when starting the Litestar application."""


@dataclass
class LogSettings:
    """Logger configuration"""

    # https://stackoverflow.com/a/1845097/6560549
    EXCLUDE_PATHS: str = r"\A(?!x)x"
    """Regex to exclude paths from logging."""
    HTTP_EVENT: str = "HTTP"
    """Log event name for logs from Litestar handlers."""
    INCLUDE_COMPRESSED_BODY: bool = False
    """Include 'body' of compressed responses in log output."""
    LEVEL: int = field(default_factory=lambda: int(
        os.getenv("LOG_LEVEL", "10")))
    """Stdlib log levels.

    Only emit logs at this level, or higher.
    """
    OBFUSCATE_COOKIES: set[str] = field(default_factory=lambda: {"session"})
    """Request cookie keys to obfuscate."""
    OBFUSCATE_HEADERS: set[str] = field(
        default_factory=lambda: {"Authorization", "X-API-KEY"})
    """Request header keys to obfuscate."""
    JOB_FIELDS: list[str] = field(
        default_factory=lambda: [
            "function",
            "kwargs",
            "key",
            "scheduled",
            "attempts",
            "completed",
            "queued",
            "started",
            "result",
            "error",
        ],
    )
    """Attributes of the SAQ.

    [`Job`](https://github.com/tobymao/saq/blob/master/saq/job.py) to be
    logged.
    """
    REQUEST_FIELDS: list[RequestExtractorField] = field(
        default_factory=lambda: [
            "path",
            "method",
            "headers",
            "cookies",
            "query",
            "path_params",
            "body",
        ],
    )
    """Attributes of the [Request][litestar.connection.request.Request] to be
    logged."""
    RESPONSE_FIELDS: list[ResponseExtractorField] = field(
        default_factory=lambda: [
            "status_code",
            "cookies",
            "headers",
            "body",
        ],
    )
    """Attributes of the [Response][litestar.response.Response] to be
    logged."""
    WORKER_EVENT: str = "Worker"
    """Log event name for logs from SAQ worker."""
    SAQ_LEVEL: int = 20
    """Level to log SAQ logs."""
    UVICORN_ACCESS_LEVEL: int = 20
    """Level to log uvicorn access logs."""
    UVICORN_ERROR_LEVEL: int = 20
    """Level to log uvicorn error logs."""
    GRANIAN_ACCESS_LEVEL: int = 30
    """Level to log uvicorn access logs."""
    GRANIAN_ERROR_LEVEL: int = 20
    """Level to log uvicorn error logs."""


@dataclass
class RedisSettings:
    URL: str = field(default_factory=lambda: os.getenv(
        "REDIS_URL", "redis://localhost:6379/0"))
    """A Redis connection URL."""
    SOCKET_CONNECT_TIMEOUT: int = field(
        default_factory=lambda: int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")))
    """Length of time to wait (in seconds) for a connection to become
    active."""
    HEALTH_CHECK_INTERVAL: int = field(default_factory=lambda: int(
        os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "5")))
    """Length of time to wait (in seconds) before testing connection health."""
    SOCKET_KEEPALIVE: bool = field(
        default_factory=lambda: os.getenv(
            "REDIS_SOCKET_KEEPALIVE", "True") in TRUE_VALUES,
    )
    """Length of time to wait (in seconds) between keepalive commands."""
    _redis_instance: Redis | None = None
    """Redis instance generated from settings."""

    @property
    def client(self) -> Redis:
        return self.get_client()

    def get_client(self) -> Redis:
        if self._redis_instance is not None:
            return self._redis_instance
        self._redis_instance = Redis.from_url(
            url=self.URL,
            encoding="utf-8",
            decode_responses=False,
            socket_connect_timeout=self.SOCKET_CONNECT_TIMEOUT,
            socket_keepalive=self.SOCKET_KEEPALIVE,
            health_check_interval=self.HEALTH_CHECK_INTERVAL,
        )
        return self._redis_instance


@dataclass
class AppSettings:
    """Application configuration"""

    URL: str = field(default_factory=lambda: os.getenv(
        "APP_URL", "http://localhost:8000"))
    """The frontend base URL"""
    DEBUG: bool = field(default_factory=lambda: os.getenv(
        "LITESTAR_DEBUG", "False") in TRUE_VALUES)
    """Run `Litestar` with `debug=True`."""
    SECRET_KEY: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", binascii.hexlify(
            os.urandom(32)).decode(encoding="utf-8")),
    )
    """Application secret key."""
    NAME: str = field(default_factory=lambda: "app")
    """Application name."""
    """JWT Encryption Algorithm"""

    @property
    def slug(self) -> str:
        """Return a slugified name.

        Returns:
            `self.NAME`, all lowercase and hyphens instead of spaces.
        """
        return slugify(self.NAME)

    # def __post_init__(self) -> None:
    #     if isinstance(self.ALLOWED_CORS_ORIGINS, str):
    #         if not self.ALLOWED_CORS_ORIGINS.startswith("["):
    #             self.ALLOWED_CORS_ORIGINS = [
    #                 host.strip() for host in self.ALLOWED_CORS_ORIGINS.split(",")]
    #         elif self.ALLOWED_CORS_ORIGINS.startswith(
    #             "[",
    #         ) and self.ALLOWED_CORS_ORIGINS.endswith("]"):
    #             self.ALLOWED_CORS_ORIGINS = list(self.ALLOWED_CORS_ORIGINS)


@dataclass
class Settings:
    app: AppSettings = field(default_factory=AppSettings)
    server: ServerSettings = field(default_factory=ServerSettings)
    log: LogSettings = field(default_factory=LogSettings)
    redis: RedisSettings = field(default_factory=RedisSettings)
    saq: SaqSettings = field(default_factory=SaqSettings)

    @classmethod
    def from_env(cls, dotenv_filename: str = ".env") -> Settings:
        from litestar.cli._utils import console

        env_file = Path(f"{os.curdir}/{dotenv_filename}")
        if env_file.is_file():
            from dotenv import load_dotenv

            console.print(
                f"[yellow]Loading environment configuration from {dotenv_filename}[/]")

            load_dotenv(env_file)
        return Settings()


@lru_cache(maxsize=1, typed=True)
def get_settings() -> Settings:
    return Settings.from_env()
