from __future__ import annotations

import io
import subprocess
import threading
from typing import cast

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
    detector._lock = threading.Lock()
    detector._running = False
    detector._thread = None
    detector._cap_thread = None
    detector._ema_probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    detector._latest = {"sound": "n", "direction": 0}
    detector._failure = None
    detector._audio_lock = threading.Lock()
    detector._audio_buf = np.zeros((0, detector.cfg.channels), dtype=np.float32)
    detector._capture_proc = None
    detector._capture_proc_lock = threading.Lock()
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


class _FakeStdout:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeProc:
    def __init__(self) -> None:
        self.stdout = _FakeStdout()
        self.terminated = False
        self.killed = False
        self.wait_called = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> None:
        del timeout
        self.wait_called = True

    def kill(self) -> None:
        self.killed = True


def test_stop_capture_process_terminates_arecord_and_clears_reference() -> None:
    detector = _make_detector()
    proc = _FakeProc()
    detector._capture_proc = cast(subprocess.Popen[bytes], proc)

    detector._stop_capture_process()

    assert proc.stdout.closed is True
    assert proc.terminated is True
    assert proc.wait_called is True
    assert proc.killed is False
    assert detector._capture_proc is None


def test_set_failure_stops_capture_process() -> None:
    detector = _make_detector()
    proc = _FakeProc()
    detector._capture_proc = cast(subprocess.Popen[bytes], proc)
    detector._running = True

    detector._set_failure(RuntimeError("boom"))

    assert detector._running is False
    assert detector._failure is not None
    assert proc.terminated is True
    assert detector._capture_proc is None


def test_stop_capture_process_kills_process_when_terminate_fails() -> None:
    class _UncooperativeProc(_FakeProc):
        def terminate(self) -> None:
            self.terminated = True
            raise RuntimeError("stuck process")

    detector = _make_detector()
    proc = _UncooperativeProc()
    detector._capture_proc = cast(subprocess.Popen[bytes], proc)

    detector._stop_capture_process()

    assert proc.terminated is True
    assert proc.killed is True


def test_run_worker_marks_failure_when_target_exits_while_running() -> None:
    detector = _make_detector()
    detector._running = True

    def target() -> None:
        return None

    detector._run_worker("infer", target)

    assert detector._failure is not None
    assert "exited unexpectedly" in str(detector._failure)
    assert detector._running is False


def test_run_worker_wraps_exceptions_with_traceback() -> None:
    detector = _make_detector()
    detector._running = True

    def target() -> None:
        raise ValueError("bad frame")

    detector._run_worker("capture", target)

    assert detector._failure is not None
    message = str(detector._failure)
    assert "live detector capture worker crashed: bad frame" in message
    assert "Traceback" in message


def test_get_status_returns_copy_of_latest_status() -> None:
    detector = _make_detector()
    detector._latest = {"sound": "h", "direction": -1}

    status = detector.get_status()
    status["sound"] = "n"

    assert detector._latest == {"sound": "h", "direction": -1}


def test_raise_if_unhealthy_re_raises_failure() -> None:
    detector = _make_detector()
    detector._failure = RuntimeError("device gone")

    with pytest.raises(RuntimeError, match="device gone"):
        detector.raise_if_unhealthy()


def test_infer_loop_updates_status_for_detected_siren(monkeypatch) -> None:
    detector = _make_detector()
    detector.cfg = DetectorConfig(sample_rate=4, channels=2, block_seconds=1.0, hop_seconds=0.01)
    detector.cfg.smooth_alpha = 0.0
    detector._audio_buf = np.array(
        [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3], [0.4, 0.4]],
        dtype=np.float32,
    )
    detector._running = True

    class _SirenModel:
        def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
            del batch, verbose
            return np.array([[0.95, 0.03, 0.02], [0.9, 0.05, 0.05]], dtype=np.float32)

    detector.model = _SirenModel()
    monkeypatch.setattr(
        "live_detector.waveform_to_logspec",
        lambda waveform, frame_length, frame_step, fft_length: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("live_detector.gcc_phat_tdoa", lambda left, right, fs: 0.05)

    def stop_after_iteration(_: float) -> None:
        detector._running = False

    monkeypatch.setattr("time.sleep", stop_after_iteration)

    detector._infer_loop()

    assert detector.get_status() == {"sound": "s", "direction": 1}


def test_infer_loop_forces_noise_direction_to_center(monkeypatch) -> None:
    detector = _make_detector()
    detector.cfg = DetectorConfig(sample_rate=4, channels=2, block_seconds=1.0, hop_seconds=0.01)
    detector.cfg.smooth_alpha = 0.0
    detector._audio_buf = np.full((4, 2), 0.1, dtype=np.float32)
    detector._running = True

    class _NoiseModel:
        def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
            del batch, verbose
            return np.array([[0.01, 0.01, 0.98], [0.02, 0.03, 0.95]], dtype=np.float32)

    detector.model = _NoiseModel()
    monkeypatch.setattr(
        "live_detector.waveform_to_logspec",
        lambda waveform, frame_length, frame_step, fft_length: np.ones((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr("live_detector.gcc_phat_tdoa", lambda left, right, fs: -0.05)

    def stop_after_iteration(_: float) -> None:
        detector._running = False

    monkeypatch.setattr("time.sleep", stop_after_iteration)

    detector._infer_loop()

    assert detector.get_status() == {"sound": "n", "direction": 0}


def test_infer_loop_skips_clipped_audio_and_preserves_previous_status(monkeypatch) -> None:
    detector = _make_detector()
    detector.cfg = DetectorConfig(
        sample_rate=4,
        channels=2,
        block_seconds=1.0,
        hop_seconds=0.01,
        peak_limit=0.5,
    )
    detector._audio_buf = np.full((4, 2), 0.75, dtype=np.float32)
    detector._latest = {"sound": "h", "direction": -1}
    detector._running = True

    def stop_after_iteration(_: float) -> None:
        detector._running = False

    monkeypatch.setattr("time.sleep", stop_after_iteration)

    detector._infer_loop()

    assert detector.get_status() == {"sound": "h", "direction": -1}
