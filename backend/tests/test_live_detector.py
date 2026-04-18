from __future__ import annotations

import io

import numpy as np
import pytest

from live_detector import DetectorConfig, LiveDetector, gcc_phat_tdoa, tau_to_direction


class _FakeModel:
    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        return np.tile(np.array([[0.9, 0.05, 0.05]], dtype=np.float32), (batch.shape[0], 1))


def _make_detector() -> LiveDetector:
    detector = LiveDetector.__new__(LiveDetector)
    detector.cfg = DetectorConfig()
    detector.model = _FakeModel()
    detector._running = False
    return detector


def test_tau_to_direction_respects_deadband() -> None:
    cfg = DetectorConfig(mic_distance_m=0.1, speed_of_sound=343.0, direction_deadband_deg=10.0)

    assert tau_to_direction(0.0, cfg) == 0


def test_tau_to_direction_clamps_to_left_and_right() -> None:
    cfg = DetectorConfig(mic_distance_m=0.1, speed_of_sound=343.0, direction_deadband_deg=10.0)
    tau = 0.1 / 343.0

    assert tau_to_direction(-tau, cfg) == -1
    assert tau_to_direction(tau, cfg) == 1


def test_gcc_phat_tdoa_detects_relative_delay() -> None:
    fs = 16000
    base = np.zeros(512, dtype=np.float32)
    base[40] = 1.0

    delayed = np.roll(base, 4)
    tau = gcc_phat_tdoa(base, delayed, fs)

    assert tau == pytest.approx(-4 / fs, abs=1e-6)


def test_read_exact_reads_until_requested_size() -> None:
    detector = _make_detector()
    detector._running = True

    data = io.BytesIO(b"abcdef")

    assert detector._read_exact(data, 4) == b"abcd"


def test_read_exact_returns_empty_bytes_when_stream_ends_early() -> None:
    detector = _make_detector()
    detector._running = True

    data = io.BytesIO(b"ab")

    assert detector._read_exact(data, 4) == b""
