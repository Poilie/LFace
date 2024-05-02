
import os
import yaml
import logging

result_logger = logging.getLogger("storage.result")
# load cow to log to elk

log_config = os.getenv("LOG_CONFIG", "logging.yaml")
logging.config.dictConfig(yaml.safe_load(open(log_config, mode="r")))
