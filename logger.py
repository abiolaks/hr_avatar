# logger.py
import os
import logging
import time
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
