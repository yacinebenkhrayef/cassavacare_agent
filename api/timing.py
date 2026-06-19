import time
from contextlib import contextmanager
import logging

logger = logging.getLogger("cassavacare-rag.timing")

@contextmanager
def timed_stage(stage_name: str, accumulator: dict):
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    accumulator[stage_name] = round(elapsed_ms, 2)
    logger.info(f"[TIMING] {stage_name}: {elapsed_ms:.2f}ms")