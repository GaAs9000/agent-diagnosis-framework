from pathlib import Path

import pytest

from c3_sandbox.cli import serial_execution


def test_second_execution_is_rejected_and_lock_is_released(tmp_path: Path):
    lock = tmp_path / "execution.lock"
    with serial_execution(lock):
        with pytest.raises(RuntimeError, match="concurrency is fixed at 1"):
            with serial_execution(lock):
                pytest.fail("a second execution acquired the serial lock")
    with serial_execution(lock):
        pass


def test_failed_execution_releases_lock(tmp_path: Path):
    lock = tmp_path / "execution.lock"
    with pytest.raises(ValueError, match="fixture failure"):
        with serial_execution(lock):
            raise ValueError("fixture failure")
    with serial_execution(lock):
        pass

