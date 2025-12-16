"""Retry utilities with exponential backoff."""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, TypeVar, ParamSpec

import tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on (default: all exceptions)

    Returns:
        Decorated function with retry logic
    """
    if retry_on is None:
        retry_on = (Exception,)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base,
            ),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
            before_sleep=_log_retry,
        )
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        return wrapper

    return decorator


def async_retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 60.0,
    exponential_base: float = 2.0,
    retry_on: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        initial_wait: Initial wait time in seconds
        max_wait: Maximum wait time in seconds
        exponential_base: Base for exponential backoff
        retry_on: Tuple of exception types to retry on (default: all exceptions)

    Returns:
        Decorated async function with retry logic
    """
    if retry_on is None:
        retry_on = (Exception,)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base,
            ),
            retry=retry_if_exception_type(retry_on),
            reraise=True,
            before_sleep=_log_retry,
        )
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def _log_retry(retry_state: RetryCallState) -> None:
    """Log retry attempts."""
    logger.warning(
        "Retrying %s after %s seconds (attempt %s/%s)",
        retry_state.fn.__name__,
        retry_state.next_action.sleep,
        retry_state.attempt_number,
        retry_state.outcome_timestamp,
    )

