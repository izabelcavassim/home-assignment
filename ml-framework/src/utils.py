import logging
import sys
from pathlib import Path
from typing import Optional
import time
from functools import wraps


def setup_logging(
    name: str = "ml_framework",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Set up structured logging for the framework.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional file path to write logs to

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.

    Args:
        logger: Logger instance (uses default if None)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger("ml_framework")
            start_time = time.time()
            _logger.info(f"Starting {func.__name__}...")

            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                _logger.info(f"Completed {func.__name__} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                _logger.error(f"Failed {func.__name__} after {elapsed:.2f}s: {str(e)}")
                raise

        return wrapper

    return decorator


class PerformanceProfiler:
    """Simple performance profiler for tracking memory and time."""

    def __init__(self, name: str = "Profile"):
        self.name = name
        self.start_time = None
        self.start_memory = None

    def __enter__(self):
        import psutil
        import os

        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import psutil
        import os

        elapsed = time.time() - self.start_time
        process = psutil.Process(os.getpid())
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_delta = end_memory - self.start_memory

        logger = logging.getLogger("ml_framework")
        logger.info(
            f"{self.name}: {elapsed:.2f}s, Memory: {memory_delta:+.2f}MB "
            f"(Total: {end_memory:.2f}MB)"
        )


def print_section(title: str, data: dict, width: int = 60):
    """
    Print a formatted section with title and key-value pairs.

    Args:
        title: Section title
        data: Dictionary of key-value pairs to display
        width: Width of the section (default: 60)

    Example:
        >>> print_section("Results", {
        ...     "Accuracy": 0.95,
        ...     "Precision": 0.92,
        ...     "Recall": 0.88,
        ...     "F1-Score": 0.90
        ... })
    """
    print(f"\n{'='*width}")
    print(f"{title.center(width)}")
    print(f"{'='*width}")
    for key, value in data.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:>10.4f}")
        elif isinstance(value, int):
            print(f"{key:<30} {value:>10}")
        else:
            print(f"{key:<30} {str(value):>10}")
    print(f"{'='*width}\n")


def save_json(data, file_path, logger=None, description: str = None):
    """
    Save data to a JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        file_path: Path to the output file (str or Path)
        logger: Optional logger for logging the save operation
        description: Optional description for the log message
    """
    import json

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    
    if logger:
        desc = description or path.name
        logger.info(f"Saved {desc} to {path}")


def save_dataframe(df, file_path, logger=None, description: str = None, index: bool = False):
    """
    Save a pandas DataFrame to CSV.

    Args:
        df: pandas DataFrame to save
        file_path: Path to the output file (str or Path)
        logger: Optional logger for logging the save operation
        description: Optional description for the log message
        index: Whether to include the index in the CSV (default: False)
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    
    if logger:
        desc = description or path.name
        logger.info(f"Saved {desc} to {path}")
