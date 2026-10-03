"""Retry transient Gemini API failures.

The API occasionally returns 503 ("high demand") or 429 ("rate limit").
When the server says how long to wait (a RetryInfo "retryDelay", e.g. on
free-tier quota limits), we wait exactly that long. Otherwise we back off
exponentially (2s, 4s, 8s...). Either way a short pause replaces an error.
"""

import logging
import re
import time
from collections.abc import Callable
from typing import TypeVar

from google.genai import errors

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RETRYABLE_STATUS = {429, 500, 503, 504}
_MAX_SERVER_DELAY = 90.0  # never wait longer than this, whatever the server asks


def _api_error(exc: BaseException | None) -> errors.APIError | None:
    """Find the Gemini API error, even when LangChain has wrapped it.

    LangChain re-raises API errors as its own type (`raise ... from e`), so the
    original error is on the `__cause__` chain.
    """
    while exc is not None:
        if isinstance(exc, errors.APIError):
            return exc
        exc = exc.__cause__
    return None


def _api_status(exc: BaseException) -> int | None:
    api_error = _api_error(exc)
    return api_error.code if api_error else None


def server_retry_delay(exc: BaseException) -> float | None:
    """Seconds the server asked us to wait (RetryInfo 'retryDelay': '52s'), if any."""
    api_error = _api_error(exc)
    details = (api_error.details or {}) if api_error else {}
    for detail in details.get("error", {}).get("details", []):
        match = re.fullmatch(r"([\d.]+)s", str(detail.get("retryDelay", "")))
        if match:
            return float(match.group(1))
    return None


def describe_api_error(exc: BaseException) -> str:
    """A short, user-facing explanation of a failed Gemini API call."""
    status = _api_status(exc)
    if status == 429:
        delay = server_retry_delay(exc)
        if delay is not None and delay > _MAX_SERVER_DELAY:
            hours = max(1, round(delay / 3600))
            return f"The free Gemini API quota for today is used up. It resets in about {hours} hour(s)."
        return "The Gemini API rate limit was reached. Please wait a minute and try again."
    if status in _RETRYABLE_STATUS:
        return "The Gemini API is temporarily unavailable. Please try again shortly."
    if status in (400, 401, 403):
        return "The Gemini API rejected the request. Check GOOGLE_API_KEY and the model names in .env."
    return f"Unexpected error ({type(exc).__name__}). Please try again."


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
            requested = server_retry_delay(exc)
            if requested is not None and requested > _MAX_SERVER_DELAY:
                raise  # e.g. a daily quota: waiting will not help within this request
            delay = requested + 1 if requested is not None else base_delay * 2 ** (attempt - 1)
            logger.warning("Gemini API error %s, retrying in %.0fs (%d/%d)", _api_status(exc), delay, attempt, attempts - 1)
            time.sleep(delay)
    raise AssertionError("unreachable")
