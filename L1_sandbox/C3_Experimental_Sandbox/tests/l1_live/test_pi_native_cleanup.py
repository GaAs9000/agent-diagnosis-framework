"""Bridge cleanup regression checks using only fake processes and in-memory pipes."""
import io

import pytest

from c3_sandbox.l1.runtime.pi_native_validation import NativePiInfrastructureError, NativePiValidator


class FakePipe(io.BytesIO):
    def __init__(self, *, write_error=None, flush_error=None, close_error=None):
        super().__init__()
        self.write_error = write_error
        self.flush_error = flush_error
        self.close_error = close_error
        self.close_calls = 0

    def write(self, payload):
        if self.write_error is not None:
            raise self.write_error
        return super().write(payload)

    def flush(self):
        if self.flush_error is not None:
            raise self.flush_error
        return super().flush()

    def close(self):
        self.close_calls += 1
        super().close()
        if self.close_error is not None:
            raise self.close_error


class FakeProcess:
    def __init__(self, stdin, stdout):
        self.stdin = stdin
        self.stdout = stdout
        self.returncode = None
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout):
        return self.returncode


def prepared_validator(stdin=None, stdout=None):
    validator = NativePiValidator()
    validator._process = FakeProcess(
        stdin if stdin is not None else FakePipe(),
        stdout if stdout is not None else FakePipe(),
    )
    validator._provenance = {"dependency_identity": "fixture"}
    return validator


@pytest.mark.parametrize("operation", ["write", "flush"])
def test_write_failure_survives_secondary_broken_pipe_on_close(operation):
    stdin = FakePipe(**{f"{operation}_error": BrokenPipeError("write failed")},
                     close_error=BrokenPipeError("close failed"))
    validator = prepared_validator(stdin)
    with pytest.raises(NativePiInfrastructureError, match="Native Pi bridge write failed"):
        validator.validate("read", {"path": "a"})
    assert stdin.closed and validator._process.stdout.closed
    assert validator._process.kill_calls == 1
    with pytest.raises(NativePiInfrastructureError, match="validator is closed"):
        validator.validate("read", {"path": "a"})
    validator.close()


def test_child_eof_survives_secondary_broken_pipe_on_close():
    validator = prepared_validator(FakePipe(close_error=BrokenPipeError("close failed")))
    validator._read_responses()
    with pytest.raises(NativePiInfrastructureError, match="bridge closed or exceeded response limit"):
        validator.validate("read", {"path": "a"})
    assert validator._process.stdin.closed and validator._process.stdout.closed
    validator.close()


def test_already_closed_pipe_is_a_typed_write_failure():
    stdin = FakePipe(close_error=ValueError("already closed"))
    with pytest.raises(ValueError):
        stdin.close()
    validator = prepared_validator(stdin)
    with pytest.raises(NativePiInfrastructureError, match="Native Pi bridge write failed"):
        validator.validate("read", {"path": "a"})
    assert validator._process.stdout.closed
    validator.close()


@pytest.mark.parametrize("response,message", [
    ({"ok": True, "provenance": {"dependency_identity": "changed"}}, "dependency identity changed"),
    ({"provenance": {"dependency_identity": "fixture"}}, "omitted validation outcome"),
])
def test_cleanup_preserves_identity_and_outcome_failures(response, message):
    validator = prepared_validator(FakePipe(close_error=OSError("close failed")))
    validator._responses.put(response)
    with pytest.raises(NativePiInfrastructureError, match=message):
        validator.validate("read", {"path": "a"})
    assert validator._process.stdout.closed


def test_normal_validation_and_close_remain_idempotent():
    validator = prepared_validator()
    response = {"ok": False, "provenance": validator._provenance,
                "native_result": {"isError": True, "content": [{"type": "text", "text": "native rejection"}]}}
    validator._responses.put(response)
    assert validator.validate("read", {}) == response
    validator.close()
    validator.close()
    assert validator._process.stdin.closed and validator._process.stdout.closed
    assert validator._process.kill_calls == 1
    with pytest.raises(NativePiInfrastructureError, match="validator is closed"):
        validator.validate("read", {})
