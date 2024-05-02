import logging
from typing import cast

from litestar.config.compression import CompressionConfig
from litestar.logging.config import LoggingConfig, StructLoggingConfig
from litestar.middleware.logging import LoggingMiddlewareConfig
from litestar.plugins.structlog import StructlogConfig
from litestar_saq import SAQConfig
# from litestar_vite import ViteConfig

from .base import get_settings

settings = get_settings()

compression = CompressionConfig(backend="gzip")
saq = SAQConfig(
    redis=settings.redis.client,
    web_enabled=settings.saq.WEB_ENABLED,
    worker_processes=settings.saq.PROCESSES,
    use_server_lifespan=settings.saq.USE_SERVER_LIFESPAN,
    # queue_configs=[
    #     QueueConfig(
    #         name="system-tasks",
    #         tasks=["app.domain.system.tasks.system_task",
    #                "app.domain.system.tasks.system_upkeep"],
    #         scheduled_tasks=[
    #             CronJob(
    #                 function="app.domain.system.tasks.system_upkeep",
    #                 unique=True,
    #                 cron="0 * * * *",
    #                 timeout=500,
    #             ),
    #         ],
    #     ),
    #     QueueConfig(
    #         name="background-tasks",
    #         tasks=["app.domain.system.tasks.background_worker_task"],
    #         scheduled_tasks=[
    #             CronJob(
    #                 function="app.domain.system.tasks.background_worker_task",
    #                 unique=True,
    #                 cron="* * * * *",
    #                 timeout=300,
    #             ),
    #         ],
    #     ),
    # ],
)

log = StructlogConfig(
    structlog_logging_config=StructLoggingConfig(
        log_exceptions="always",
        traceback_line_limit=4,
        standard_lib_logging_config=LoggingConfig(
            root={"level": logging.getLevelName(settings.log.LEVEL), "handlers": [
                "queue_listener"]},
            loggers={
                "uvicorn.access": {
                    "propagate": False,
                    "level": settings.log.UVICORN_ACCESS_LEVEL,
                    "handlers": ["queue_listener"],
                },
                "uvicorn.error": {
                    "propagate": False,
                    "level": settings.log.UVICORN_ERROR_LEVEL,
                    "handlers": ["queue_listener"],
                },
                "granian.access": {
                    "propagate": False,
                    "level": settings.log.GRANIAN_ACCESS_LEVEL,
                    "handlers": ["queue_listener"],
                },
                "granian.error": {
                    "propagate": False,
                    "level": settings.log.GRANIAN_ERROR_LEVEL,
                    "handlers": ["queue_listener"],
                },
                "saq": {
                    "propagate": False,
                    "level": settings.log.SAQ_LEVEL,
                    "handlers": ["queue_listener"],
                },
            },
        ),
    ),
    middleware_logging_config=LoggingMiddlewareConfig(
        request_log_fields=["method", "path", "path_params", "query"],
        response_log_fields=["status_code"],
    ),
)
