from advanced_alchemy.extensions.litestar import SQLAlchemyPlugin
from litestar.plugins.structlog import StructlogPlugin
from litestar_granian import GranianPlugin
from litestar_saq import SAQPlugin
from litestar_vite import VitePlugin

from app.config import app as config
from app.server.builder import ApplicationConfigurator

structlog = StructlogPlugin(config=config.log)
saq = SAQPlugin(config=config.saq)
granian = GranianPlugin()
app_config = ApplicationConfigurator()
