"""Structured logging with rotation."""
import logging
import logging.handlers
import sys
from pathlib import Path
from .config import settings

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

log = logging.getLogger("bot")
log.setLevel(getattr(logging, settings.log_level.upper()))
h = logging.handlers.TimedRotatingFileHandler(
    "logs/bot.log", when="midnight", backupCount=7
)
h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
log.addHandler(h)
log.addHandler(logging.StreamHandler(sys.stdout))

