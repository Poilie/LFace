from ovmsclient import make_grpc_client, make_http_client
import time
import cv2
import numpy as np
import asyncio
from typing import Dict

from litestar import Litestar, get

from litestar_saq import QueueConfig, SAQConfig, SAQPlugin

saq = SAQPlugin(config=SAQConfig(redis_url="redis://localhost:6397/0",
                queue_configs=[QueueConfig(name="samples")]))


client = make_grpc_client("localhost:9322")


async def predict_image() -> None:
    img = cv2.imread(
        "/home/misa/Workshop/LFace/test_folder/test_data/112_user/5.jpg")
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    # output = client.predict({"input.1": img}, "face-embedder")
    # await for client to predict
    client.predict({"input.1": img}, "face-embedder")
    print("Predicted image")


@get("/")
async def hello_world() -> Dict[str, str]:
    """Handler function that returns a greeting dictionary."""
    # process_image()
    await predict_image()
    # wait 1 second
    # await asyncio.sleep(1)
    return {"hello": "world"}


def create_app() -> Litestar:
    from app.config import app as app_config
    from app.server import openapi, plugins
    """Create the Litestar application."""
    return Litestar(
        route_handlers=[hello_world],
        openapi_config=openapi.config,
        plugins=[
            plugins.structlog,
            # saq,
            plugins.saq,
            plugins.granian,
            plugins.app_config,
        ]
    )


app = create_app()
