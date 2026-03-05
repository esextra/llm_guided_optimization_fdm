# print_quality/utils/logging_utils.py
from __future__ import annotations

import logging
from typing import Optional

def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a console logger with a compact formatter.
    Prevents duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name if name else "print_quality")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            fmt="%(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger
