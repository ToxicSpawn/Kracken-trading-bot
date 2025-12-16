"""Security utilities for API key encryption and management."""

from __future__ import annotations

import os
import base64
from pathlib import Path
from typing import Optional

import logging

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

# Default key file location
DEFAULT_KEY_FILE = Path.home() / ".kraken_bot" / "encryption.key"


def _get_or_create_key(key_file: Path = DEFAULT_KEY_FILE) -> bytes:
    """
    Get existing encryption key or create a new one.

    Args:
        key_file: Path to the key file

    Returns:
        Encryption key bytes
    """
    key_file.parent.mkdir(parents=True, exist_ok=True)

    if key_file.exists():
        return key_file.read_bytes()

    # Generate new key
    key = Fernet.generate_key()
    key_file.write_bytes(key)
    key_file.chmod(0o600)  # Read/write for owner only
    logger.info(f"Generated new encryption key at {key_file}")
    return key


def _derive_key_from_password(password: str, salt: bytes) -> bytes:
    """
    Derive encryption key from password using PBKDF2.

    Args:
        password: User password
        salt: Salt bytes

    Returns:
        Derived key bytes
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    return key


def encrypt_api_key(api_key: str, key_file: Path = DEFAULT_KEY_FILE) -> bytes:
    """
    Encrypt an API key.

    Args:
        api_key: Plain text API key
        key_file: Path to encryption key file

    Returns:
        Encrypted API key bytes
    """
    key = _get_or_create_key(key_file)
    cipher_suite = Fernet(key)
    encrypted = cipher_suite.encrypt(api_key.encode())
    return encrypted


def decrypt_api_key(encrypted_key: bytes, key_file: Path = DEFAULT_KEY_FILE) -> str:
    """
    Decrypt an API key.

    Args:
        encrypted_key: Encrypted API key bytes
        key_file: Path to encryption key file

    Returns:
        Decrypted API key string
    """
    key = _get_or_create_key(key_file)
    cipher_suite = Fernet(key)
    decrypted = cipher_suite.decrypt(encrypted_key)
    return decrypted.decode()


def get_encrypted_env(env_var: str, key_file: Path = DEFAULT_KEY_FILE) -> Optional[str]:
    """
    Get and decrypt an environment variable that contains encrypted data.

    Args:
        env_var: Environment variable name
        key_file: Path to encryption key file

    Returns:
        Decrypted value or None if not found
    """
    encrypted_value = os.getenv(env_var)
    if not encrypted_value:
        return None

    try:
        # Try to decode as base64 first (if stored as base64 string)
        encrypted_bytes = base64.b64decode(encrypted_value)
        return decrypt_api_key(encrypted_bytes, key_file)
    except Exception as e:
        logger.warning(f"Failed to decrypt {env_var}: {e}. Trying as plain text.")
        # Fallback to plain text (for backward compatibility)
        return encrypted_value


def store_encrypted_env(env_var: str, value: str, key_file: Path = DEFAULT_KEY_FILE) -> str:
    """
    Encrypt and store a value as an environment variable string (base64 encoded).

    Args:
        env_var: Environment variable name (for logging)
        value: Plain text value to encrypt
        key_file: Path to encryption key file

    Returns:
        Base64-encoded encrypted string (ready for .env file)
    """
    encrypted = encrypt_api_key(value, key_file)
    return base64.b64encode(encrypted).decode()

