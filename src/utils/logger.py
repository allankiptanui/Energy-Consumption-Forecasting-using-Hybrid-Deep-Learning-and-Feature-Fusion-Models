import logging
from pathlib import Path
from typing import Optional, Union


_DEFAULT_FORMAT = "%(asctime)s — %(levelname)s — %(name)s — %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str = __name__,
               level: Union[int, str] = logging.INFO,
               logfile: Optional[str] = None) -> logging.Logger:

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid double logging in some environments

    # If handlers already exist, return same logger
    if logger.handlers:
        return logger

    # Console / stream handler
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh_formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATEFMT)
    sh.setFormatter(sh_formatter)
    logger.addHandler(sh)

    # Optional file handler
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, mode="a")
        fh.setLevel(level)
        fh_formatter = logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATEFMT)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
