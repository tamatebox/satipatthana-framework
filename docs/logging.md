# Logging Guidelines for Satipatthana Framework

The Satipatthana Framework uses a centralized logging utility (`satipatthana/utils/logger.py`) to provide consistent and configurable logging across the project. This document outlines how to set up and use logging effectively.

## 1. Setting Up Logging

To ensure consistent logging behavior, you *must* call `setup_logging()` once at the entry point of your application or script. This configures the root logger, including handlers for console output and optional file output.

**Example (e.g., in `main.py` or the start of a training script):**

```python
import logging
from satipatthana.utils.logger import setup_logging

def main():
    # Configure logging: INFO level to console, DEBUG level to a file
    setup_logging(level=logging.INFO, log_file="application.log")

    # Your application logic here
    # ...

if __name__ == "__main__":
    main()
```

### `setup_logging` Parameters

* `level`: The minimum logging level to capture (e.g., `logging.DEBUG`, `logging.INFO`, `logging.WARNING`, `logging.ERROR`, `logging.CRITICAL`). Defaults to `logging.INFO`.
* `log_file`: (Optional) A string path to a file where logs will also be written. If `None`, logs are only sent to the console.

## 2. Obtaining a Logger Instance

Within any module, retrieve a logger instance using `get_logger(__name__)`. This ensures that log messages are associated with the correct module name, making it easier to trace their origin.

**Example:**

```python
from satipatthana.utils.logger import get_logger

# Get a logger for the current module
logger = get_logger(__name__)

def some_function():
    logger.debug("This is a debug message.")
    logger.info("An informative message here.")
    logger.warning("Something potentially problematic happened.")
    logger.error("An error occurred!")
    logger.critical("Critical issue: application might crash.")

some_function()
```

## 3. Best Practices

* **Call `setup_logging` only once:** Do not call `setup_logging` multiple times. It is designed to be a singleton-like configuration, and subsequent calls will be ignored.
* **Use `__name__` for `get_logger`:** Always pass `__name__` to `get_logger()` to leverage Python's hierarchical logging system and easily filter logs by module.
* **Choose appropriate logging levels:**
  * `DEBUG`: Detailed information, typically only of interest when diagnosing problems.
  * `INFO`: Confirmation that things are working as expected.
  * `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., 'disk space low'). The software is still working as expected.
  * `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
  * `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.
* **Avoid excessive logging:** While logging is crucial for debugging and monitoring, avoid logging overly verbose or sensitive information, especially at lower levels in production environments.
* **No implicit setup:** `get_logger()` *will not* implicitly call `setup_logging()`. If you forget to call `setup_logging()` at your application's entry point, `get_logger()` will return a default logger that might not have the desired configuration.
