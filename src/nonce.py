"""Monotonic, disk-backed nonce generator."""
import json
import time
import pathlib
import threading

_lock = threading.Lock()
_file = pathlib.Path("data/nonce.counter")

def next_nonce() -> int:
    """Generate next monotonic nonce."""
    with _lock:
        n = (_file.exists() and int(_file.read_text())) or 0
        candidate = max(n + 1, int(time.time() * 1000))
        _file.parent.mkdir(parents=True, exist_ok=True)
        _file.write_text(str(candidate))
        return candidate

