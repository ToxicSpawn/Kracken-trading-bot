"""Tests for security utilities."""

import pytest
import os
import tempfile
from pathlib import Path
from utils.security import (
    encrypt_api_key,
    decrypt_api_key,
    get_encrypted_env,
    store_encrypted_env,
)


def test_encrypt_decrypt():
    """Test encryption and decryption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "test.key"
        api_key = "test_api_key_12345"

        # Encrypt
        encrypted = encrypt_api_key(api_key, key_file)

        # Decrypt
        decrypted = decrypt_api_key(encrypted, key_file)

        assert decrypted == api_key
        assert encrypted != api_key.encode()  # Should be different


def test_store_encrypted_env():
    """Test storing encrypted value as base64 string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "test.key"
        value = "secret_value"

        # Store encrypted
        encrypted_str = store_encrypted_env("TEST_VAR", value, key_file)

        # Should be base64 encoded
        assert isinstance(encrypted_str, str)
        assert len(encrypted_str) > 0

        # Decrypt and verify
        import base64
        encrypted_bytes = base64.b64decode(encrypted_str)
        decrypted = decrypt_api_key(encrypted_bytes, key_file)
        assert decrypted == value


def test_get_encrypted_env_not_set():
    """Test getting non-existent env var."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "test.key"
        result = get_encrypted_env("NON_EXISTENT_VAR", key_file)
        assert result is None


def test_get_encrypted_env_plain_text_fallback():
    """Test fallback to plain text if decryption fails."""
    with tempfile.TemporaryDirectory() as tmpdir:
        key_file = Path(tmpdir) / "test.key"
        os.environ["TEST_PLAIN"] = "plain_text_value"

        result = get_encrypted_env("TEST_PLAIN", key_file)
        assert result == "plain_text_value"

        # Cleanup
        del os.environ["TEST_PLAIN"]

