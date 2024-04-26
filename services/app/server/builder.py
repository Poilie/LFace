# pylint: disable=[invalid-name,import-outside-toplevel]
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from litestar.config.response_cache import ResponseCacheConfig, default_cache_key_builder
from litestar.plugins import CLIPluginProtocol, InitPluginProtocol
from litestar.security.jwt import OAuth2Login
from litestar.stores.redis import RedisStore
from litestar.stores.registry import StoreRegistry

if TYPE_CHECKING:
    from click import Group
    from litestar import Request
    from litestar.config.app import AppConfig
    from redis.asyncio import Redis


T = TypeVar("T")


class ApplicationConfigurator(InitPluginProtocol, CLIPluginProtocol):
    """Application configuration plugin."""

    __slots__ = ("redis", "app_slug")
    redis: Redis
    app_slug: str

    def __init__(self) -> None:
        """Initialize ``ApplicationConfigurator``.

        Args:
            config: configure and start SAQ.
        """

    def on_cli_init(self, cli: Group) -> None:
        # from app.cli.commands import user_management_app
        from app.config import get_settings

        settings = get_settings()
        self.redis = settings.redis.get_client()
        self.app_slug = settings.app.slug
        # cli.add_command(user_management_app)

        return super().on_cli_init(cli)

    def on_app_init(self, app_config: AppConfig) -> AppConfig:
        """Configure application for use with SQLAlchemy.

        Args:
            app_config: The :class:`AppConfig <.config.app.AppConfig>` instance.
        """

        from advanced_alchemy.exceptions import RepositoryError
        from litestar.security.jwt import Token

        from app.config import constants, get_settings
        # from app.db.models import User as UserModel
        from app.lib.exceptions import ApplicationError, exception_to_http_response

        settings = get_settings()
        self.redis = settings.redis.get_client()
        self.app_slug = settings.app.slug
        app_config.response_cache_config = ResponseCacheConfig(
            default_expiration=constants.CACHE_EXPIRATION,
            key_builder=self._cache_key_builder,
        )
        app_config.stores = StoreRegistry(
            default_factory=self.redis_store_factory)
        # type: ignore[attr-defined]
        app_config.on_shutdown.append(self.redis.aclose)
        # app_config.signature_namespace.update(
        #     {
        #         "Token": Token,
        #         "OAuth2Login": OAuth2Login,
        #         "UserModel": UserModel,
        #     },
        # )
        app_config.exception_handlers = {
            ApplicationError: exception_to_http_response,
            RepositoryError: exception_to_http_response,
        }
        return app_config

    def redis_store_factory(self, name: str) -> RedisStore:
        return RedisStore(self.redis, namespace=f"{self.app_slug}:{name}")

    def _cache_key_builder(self, request: Request) -> str:
        """App name prefixed cache key builder.

        Args:
            request (Request): Current request instance.

        Returns:
            str: App slug prefixed cache key.
        """

        return f"{self.app_slug}:{default_cache_key_builder(request)}"
