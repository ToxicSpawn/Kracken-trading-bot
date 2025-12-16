"""Kraken API wrapper with single retry logic."""
import requests
import hmac
import hashlib
import time
import logging
import base64
from .nonce import next_nonce
from .config import settings

log = logging.getLogger(__name__)
API = "https://api.kraken.com/0"

def _sign(path: str, data: dict) -> bytes:
    """Sign API request."""
    post = "&".join([f"{k}={v}" for k, v in sorted(data.items())])
    sha256_hash = hashlib.sha256((data["nonce"] + post).encode()).digest()
    return hmac.new(
        base64.b64decode(settings.kraken_private_key),
        path.encode() + sha256_hash,
        hashlib.sha512,
    ).digest()

def request(path: str, data: dict = None, retries: int = 3):
    """Make API request with retry logic."""
    data = data or {}
    data["nonce"] = str(next_nonce())
    headers = {
        "API-Key": settings.kraken_api_key,
        "API-Sign": base64.b64encode(_sign(path, data)).decode(),
    }
    
    for attempt in range(retries):
        try:
            r = requests.post(API + path, data=data, headers=headers, timeout=10)
            r.raise_for_status()
            js = r.json()
            if js.get("error"):
                raise ValueError(js["error"])
            return js["result"]
        except Exception as e:
            if attempt < retries - 1:
                log.warning("kraken error %s, retrying %s", e, attempt + 1)
                time.sleep(2 ** attempt)
            else:
                log.error("kraken retries exhausted")
                raise RuntimeError("kraken retries exhausted") from e

