from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def send_telegram_message(message: str, token: Optional[str] = None, chat_id: Optional[str] = None) -> bool:
    """Minimal Telegram notifier stub.

    Safe default for tests: no network calls. Logs and returns False.
    """
    token = token or os.getenv("TELEGRAM_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        logger.info("Telegram not configured; skipping message: %s", message)
        return False

    logger.info("Telegram configured; message suppressed in stub: %s", message)
    return False

