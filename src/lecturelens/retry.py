"""Retry transient Gemini API failures with exponential backoff.

The API occasionally returns 503 ("high demand") or 429 ("rate limit") for
a few seconds. Retrying after a growing delay (2s, 4s, 8s...) turns these
into a short pause instead of an error the user sees.
"""

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai import errors

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 503, 504}


def _api_status(exc: BaseException | None) -> int | None:
    """Find the HTTP status of a Gemini API error, even when LangChain has wrapped it.

    LangChain re-raises API errors as its own type (`raise ... from e`), so the
    original error is on the `__cause__` chain.
    """
    while exc is not None:
        if isinstance(exc, errors.APIError):
            return exc.code
        exc = exc.__cause__
    return None


def is_retryable(exc: BaseException) -> bool:
    return _api_status(exc) in _RETRYABLE_STATUS


def with_retries(fn: Callable[[], T], attempts: int = 4, base_delay: float = 2.0) -> T:
    """Call `fn()`, retrying retryable API errors up to `attempts` times in total."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts or not is_retryable(exc):
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Gemini API error %s, retrying in %.0fs (%d/%d)", _api_status(exc), delay, attempt, attempts - 1)
            time.sleep(delay)
    raise AssertionError("unreachable")
