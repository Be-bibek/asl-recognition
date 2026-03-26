import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized: set[str] = set()


def get_logger(name: str, log_file: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger configured with a StreamHandler (stdout) and an
    optional FileHandler. Safe to call multiple times — handlers are only
    added once per logger name.
    """
    logger = logging.getLogger(name)

    if name in _initialized:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    _initialized.add(name)
    return logger
