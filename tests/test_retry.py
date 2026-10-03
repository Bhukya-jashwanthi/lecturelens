import pytest
from google.genai import errors

from lecturelens.retry import describe_api_error, is_retryable, server_retry_delay, with_retries


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


def quota_error(delay: str) -> errors.APIError:
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]
    return errors.APIError(429, {"error": {"code": 429, "message": "quota", "details": details}})


def test_reads_server_requested_delay():
    assert server_retry_delay(quota_error("52.8s")) == 52.8
    assert server_retry_delay(api_error(503)) is None
    assert server_retry_delay(ValueError("unrelated")) is None


def test_waits_for_server_requested_delay(monkeypatch):
    sleeps = []
    monkeypatch.setattr("lecturelens.retry.time.sleep", sleeps.append)
    assert with_retries(Flaky(quota_error("3s")), base_delay=0) == "ok"
    assert sleeps == [4.0]  # requested 3s + 1s margin


def test_gives_up_immediately_when_server_asks_for_a_very_long_wait(monkeypatch):
    monkeypatch.setattr("lecturelens.retry.time.sleep", lambda s: pytest.fail("should not sleep"))
    fn = Flaky(quota_error("3600s"))
    with pytest.raises(errors.APIError):
        with_retries(fn)
    assert fn.calls == 1


def test_describe_api_error_gives_user_friendly_messages():
    assert "quota for today" in describe_api_error(quota_error("21354s"))
    assert "about 6 hour" in describe_api_error(quota_error("21354s"))
    assert "wait a minute" in describe_api_error(quota_error("30s"))
    assert "temporarily unavailable" in describe_api_error(api_error(503))
    assert "GOOGLE_API_KEY" in describe_api_error(api_error(403))
    assert "Unexpected error (ValueError)" in describe_api_error(ValueError("x"))


def test_detects_errors_wrapped_by_langchain():
    try:
        try:
            raise api_error(503)
        except errors.APIError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as wrapped:
        assert is_retryable(wrapped)
    assert not is_retryable(ValueError("unrelated"))
