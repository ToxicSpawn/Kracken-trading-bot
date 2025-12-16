"""Enhanced logging for the trading bot."""
import logging
import sys
from typing import Optional

# Configure root logger
logger = logging.getLogger("kracken")
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)

# Add handler if not already added
if not logger.handlers:
    logger.addHandler(console_handler)

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    if name:
        return logging.getLogger(f"kracken.{name}")
    return logger

