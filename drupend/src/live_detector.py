import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import tensorflow as tf
from siren_detector.ai.middleman import waveform_to_logspec
import subprocess

LABELS = ["siren", "honk", "noise"]
LABEL_TO_CHAR = {"siren": "s", "honk": "h", "noise": "n"}

@dataclass
class DetectorConfig:
    model_path: str = "siren_detector/ai/trained_car_alert_model.h5"
    sample_rate: int = 16000
    channels: int = 2
    block_seconds: float = 1.0
    hop_seconds: float = 0.25
    peak_limit: float = 0.5

    frame_length: int = 512
    frame_step: int = 128
    fft_length: int = 512

    mic_distance_m: float = 0.35 # TODO: measure and edit
    speed_of_sound: float = 343.0
    direction_deadband_deg: float = 10.0

    smooth_alpha: float = 0.6

    device: Optional[int] = None
    
    arecord_device: str = "plughw:2,0"
    arecord_format: str = "S32_LE"

def gcc_phat_tdoa(x: np.ndarray, y: np.ndarray, fs: int) -> float:
    """
    Use GCC_PHAT to estimate time delays
    """
    n = 1
    L = len(x) + len(y)
    while n < L:
        n *= 2

    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)

    denom = np.abs(R)
    denom[denom < 1e-12] = 1e-12
    R /= denom

    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-(n//2):], cc[:(n//2)]))

    max_shift = int(n // 2)
    shift = np.argmax(cc) - max_shift
    tau = shift / float(fs)
    return tau


def tau_to_direction(tau: float, cfg: DetectorConfig) -> int:
    """
    Convert time delay to left/center/right.
    Approximate angle using sin(theta)=tau*c/d. Clamp to [-1,1].
    Convention:
      - If tau > 0 => left channel leads right => sound from LEFT => direction = -1
      - If tau < 0 => right leads left => sound from RIGHT => direction = +1
    """
    s = (tau * cfg.speed_of_sound) / max(cfg.mic_distance_m, 1e-6)
    s = float(np.clip(s, -1.0, 1.0))
    theta = np.degrees(np.arcsin(s))

    if abs(theta) <= cfg.direction_deadband_deg:
        return 0
    return -1 if theta > 0 else 1


class LiveDetector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = tf.keras.models.load_model(cfg.model_path, compile=False)

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._ema_probs = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        self._latest: Dict[str, object] = {"sound": "n", "direction": 0}

        self._audio_lock = threading.Lock()
        self._audio_buf = np.zeros((0, cfg.channels), dtype=np.float32)
        self._cap_thread: Optional[threading.Thread] = None

    def _read_exact(self, pipe, nbytes: int) -> bytes:
        out = bytearray()
        while len(out) < nbytes and self._running:
            chunk = pipe.read(nbytes - len(out))
            if not chunk:
                return b""
            out.extend(chunk)
        return bytes(out)

    def start(self):
        print("DETECTOR: start() called")
        if self._running:
            return
        self._running = True

        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._cap_thread.start()

        self._thread = threading.Thread(target=self._infer_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap_thread:
            self._cap_thread.join(timeout=2)

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._latest)

    def _capture_loop(self):
        cfg = self.cfg
        block_len = int(cfg.sample_rate * cfg.block_seconds)
        hop_len = int(cfg.sample_rate * cfg.hop_seconds)

        bytes_per_sample = 4  # S32_LE
        frame_bytes = hop_len * cfg.channels * bytes_per_sample

        cmd = [
            "arecord",
            "-D", cfg.arecord_device,
            "-f", cfg.arecord_format,
            "-r", str(cfg.sample_rate),
            "-c", str(cfg.channels),
            "-t", "raw",
            "--buffer-size=262144",
            "--period-size=32768",
            "-q",
        ]

        print("DETECTOR: capture loop starting arecord...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

        try:
            while self._running:
                if proc.poll() is not None:
                    print("DETECTOR: arecord died, restarting...")
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

                assert proc.stdout is not None
                raw = self._read_exact(proc.stdout, frame_bytes)
                if not raw or len(raw) != frame_bytes:
                    continue

                audio_i32 = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                audio_i32 /= 2147483648.0  # -> [-1,1]
                chunk = audio_i32.reshape(hop_len, cfg.channels)

                with self._audio_lock:
                    self._audio_buf = np.concatenate([self._audio_buf, chunk], axis=0)
                    if self._audio_buf.shape[0] > block_len:
                        self._audio_buf = self._audio_buf[-block_len:, :]
        finally:
            try:
                proc.kill()
            except Exception:
                pass

    def _infer_loop(self):
        print("DETECTOR: infer loop running")
        cfg = self.cfg
        block_len = int(cfg.sample_rate * cfg.block_seconds)

        while self._running:
            time.sleep(cfg.hop_seconds)

            with self._audio_lock:
                if self._audio_buf.shape[0] < block_len:
                    continue
                window = self._audio_buf.copy()

            left = window[:, 0].astype(np.float32)
            right = window[:, 1].astype(np.float32)

            peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
            if peak >= cfg.peak_limit:
                continue

            def standardize(spec: np.ndarray) -> np.ndarray:
                return (spec - spec.mean()) / (spec.std() + 1e-6)

            spec_l = waveform_to_logspec(left, cfg.frame_length, cfg.frame_step, cfg.fft_length)
            spec_r = waveform_to_logspec(right, cfg.frame_length, cfg.frame_step, cfg.fft_length)

            spec_l = standardize(spec_l)
            spec_r = standardize(spec_r)

            X = np.stack([spec_l[..., np.newaxis], spec_r[..., np.newaxis]], axis=0).astype(np.float32)

            probs = self.model.predict(X, verbose=0)
            probs_mean = probs.mean(axis=0).astype(np.float32)

            self._ema_probs = cfg.smooth_alpha * self._ema_probs + (1.0 - cfg.smooth_alpha) * probs_mean

            idx = int(np.argmax(self._ema_probs))
            label = LABELS[idx]

            tau = gcc_phat_tdoa(left, right, cfg.sample_rate)
            direction = tau_to_direction(tau, cfg) if label != "noise" else 0

            status = {"sound": LABEL_TO_CHAR[label], "direction": int(direction)}
            with self._lock:
                self._latest = status
