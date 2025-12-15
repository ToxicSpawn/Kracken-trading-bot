"""
Security guardrails for input sanitization and content filtering.
Uses guardrails-ai and custom validation.
"""
import logging
import re
from typing import Optional, Dict, Any, List

try:
    from guardrails import Guard
    from guardrails.hub import ProfanityFilter, ToxicLanguage, DetectPII
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

logger = logging.getLogger(__name__)


class SecurityGuard:
    """
    Security guard for input sanitization and validation.
    """

    def __init__(
        self,
        enable_profanity_filter: bool = True,
        enable_toxic_language: bool = True,
        enable_pii_detection: bool = True,
        max_length: int = 10000,
        allowed_patterns: Optional[List[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize security guard.

        Args:
            enable_profanity_filter: Filter profanity
            enable_toxic_language: Detect toxic language
            enable_pii_detection: Detect personally identifiable information
            max_length: Maximum input length
            allowed_patterns: Allowed regex patterns
            blocked_patterns: Blocked regex patterns
        """
        self.max_length = max_length
        self.allowed_patterns = allowed_patterns or []
        self.blocked_patterns = blocked_patterns or [
            r"<\s*script[^>]*>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"on\w+\s*=",  # Event handlers
        ]

        # Initialize guardrails if available
        self.guard = None
        if GUARDRAILS_AVAILABLE:
            try:
                self.guard = Guard()
                if enable_profanity_filter:
                    self.guard.use(ProfanityFilter())
                if enable_toxic_language:
                    self.guard.use(ToxicLanguage())
                if enable_pii_detection:
                    self.guard.use(DetectPII())
                logger.info("Guardrails initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize guardrails: {e}")
                self.guard = None

    def validate(self, text: str) -> Dict[str, Any]:
        """
        Validate and sanitize input text.

        Returns:
            Dictionary with validation result and sanitized text
        """
        result = {
            "valid": True,
            "sanitized": text,
            "violations": [],
            "warning": False,
        }

        # Check length
        if len(text) > self.max_length:
            result["valid"] = False
            result["violations"].append(f"Input exceeds maximum length of {self.max_length}")
            return result

        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result["valid"] = False
                result["violations"].append(f"Blocked pattern detected: {pattern}")
                result["sanitized"] = re.sub(pattern, "", text, flags=re.IGNORECASE)

        # Check allowed patterns if specified
        if self.allowed_patterns:
            if not any(re.search(pattern, text, re.IGNORECASE) for pattern in self.allowed_patterns):
                result["warning"] = True
                result["violations"].append("Input does not match allowed patterns")

        # Use guardrails if available
        if self.guard and result["valid"]:
            try:
                guard_result = self.guard.validate(text)
                if not guard_result.validation_passed:
                    result["warning"] = True
                    result["violations"].extend(guard_result.validation_failures or [])
                    # Don't block, just warn
            except Exception as e:
                logger.warning(f"Guardrails validation error: {e}")

        # Basic sanitization
        result["sanitized"] = self._sanitize(result["sanitized"])

        return result

    def _sanitize(self, text: str) -> str:
        """Basic text sanitization."""
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove control characters except newlines and tabs
        text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
        # Limit consecutive whitespace
        text = re.sub(r"\s{3,}", " ", text)
        return text.strip()


class PromptInjectionGuard:
    """Guard against prompt injection attacks."""

    def __init__(self):
        self.injection_patterns = [
            r"(?i)ignore\s+(previous|above|all)\s+instructions",
            r"(?i)forget\s+(everything|all)",
            r"(?i)you\s+are\s+now",
            r"(?i)system\s*:\s*",
            r"(?i)assistant\s*:\s*",
            r"(?i)user\s*:\s*.*\n.*system",
            r"<\|im_start\|>",
            r"<\|im_end\|>",
        ]

    def detect(self, text: str) -> bool:
        """Detect potential prompt injection."""
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                return True
        return False

    def sanitize(self, text: str) -> str:
        """Remove potential injection patterns."""
        sanitized = text
        for pattern in self.injection_patterns:
            sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        return sanitized.strip()

