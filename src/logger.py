#custom logger function to handle all logs
import os
import logging
import sys
from typing import Optional

def get_logger(name:str, level:Optional[str]="INFO") -> logging.Logger:
    """
    Creates and returns a logger with the given name and level.
    If the logger already has handlers, it returns the existing instance.

    Args:
        name: Usually __name__ from the calling module.
        level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s ",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
    