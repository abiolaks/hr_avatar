# logger.py
import os
import json
import logging
import time
from datetime import datetime, timezone
from functools import wraps
from config import LOGS_DIR, LOG_LEVEL

# Create logger
logger = logging.getLogger("hr_avatar")
logger.setLevel(getattr(logging, LOG_LEVEL))

# File handler
log_file = os.path.join(LOGS_DIR, "hr_avatar.log")
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

class EventLogger:
    """
    Writes one JSON record per conversation turn to logs/events.jsonl.
    Each record is a complete, self-contained snapshot of what happened:
    who asked, which tool fired, what came back, and how long it took.
    Parse with pandas or jq for aggregated reports.
    """

    def __init__(self, log_dir: str):
        self._path = os.path.join(log_dir, "events.jsonl")

    def log(self, event: dict) -> None:
        event["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._path, "a") as fh:
            fh.write(json.dumps(event, default=str) + "\n")


event_logger = EventLogger(LOGS_DIR)


def log_performance(func):
    """Decorator to log function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"PERF | {func.__name__} took {elapsed:.3f}s")
        return result
    return wrapper
