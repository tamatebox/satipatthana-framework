import logging
import sys
from typing import Optional

# Flag to ensure logging is configured only once (singleton-like).
_IS_CONFIGURED = False


def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None):
    """Sets up the logging configuration for the entire Samadhi Framework.

    It is recommended to call this function once at the beginning of the main script.

    Args:
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs will be written.
    """
    global _IS_CONFIGURED
    if _IS_CONFIGURED:
        return

    # Defines the log format: [Time] Level [ModuleName] Message
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Console output handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File output handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _IS_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retrieves a logger instance for a given module.

    Usage:
        from samadhi.utils.logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: The name of the logger, typically `__name__` of the calling module.

    Returns:
        A `logging.Logger` instance.
    """
    # get_logger should not implicitly set up logging.
    # Users should call setup_logging() explicitly at the application's entry point.
    return logging.getLogger(name)
