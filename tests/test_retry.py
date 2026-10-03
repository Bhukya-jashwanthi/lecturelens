import pytest
from google.genai import errors

from lecturelens.retry import is_retryable, with_retries


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "message": "test", "status": "TEST"}})


class Flaky:
    """Fails with the given errors first, then succeeds."""

    def __init__(self, *failures):
        self.failures = list(failures)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "ok"


def test_retries_transient_errors_then_succeeds():
    fn = Flaky(api_error(503), api_error(429))
    assert with_retries(fn, base_delay=0) == "ok"
    assert fn.calls == 3


def test_does_not_retry_permanent_errors():
    fn = Flaky(api_error(400))
    with pytest.raises(errors.APIError):
        with_retries(fn, base_delay=0)
    assert fn.calls == 1


def test_gives_up_after_max_attempts():
    fn = Flaky(*[api_error(503)] * 5)
    with pytest.raises(errors.APIError):
        with_retries(fn, attempts=3, base_delay=0)
    assert fn.calls == 3


def test_detects_errors_wrapped_by_langchain():
    try:
        try:
            raise api_error(503)
        except errors.APIError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        assert is_retryable(wrapped)
    assert not is_retryable(ValueError("unrelated"))
