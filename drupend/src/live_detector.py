import os
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
import tensorflow as tf
import sounddevice as sd
from siren_detector.ai.middleman import waveform_to_logspec

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

    def start(self):
        print("DETECTOR: start() called")
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return dict(self._latest)

    def _loop(self):
        print("DETECTOR: loop running")
        cfg = self.cfg
        block_len = int(cfg.sample_rate * cfg.block_seconds)
        hop_len = int(cfg.sample_rate * cfg.hop_seconds)

        buffer = np.zeros((0, cfg.channels), dtype=np.float32)

        with sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="float32",
            blocksize=hop_len,
            device=cfg.device
        ) as stream:

            while self._running:
                chunk, _ = stream.read(hop_len)
                if chunk is None or len(chunk) == 0:
                    continue

                buffer = np.concatenate([buffer, chunk], axis=0)

                if buffer.shape[0] > block_len:
                    buffer = buffer[-block_len:, :]

                if buffer.shape[0] < block_len:
                    continue 

                left = buffer[:, 0].astype(np.float32)
                right = buffer[:, 1].astype(np.float32)

                peak = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))))
                if peak >= cfg.peak_limit:
                    continue

                spec_l = waveform_to_logspec(left, cfg.frame_length, cfg.frame_step, cfg.fft_length)
                spec_r = waveform_to_logspec(right, cfg.frame_length, cfg.frame_step, cfg.fft_length)

                x_l = spec_l[..., np.newaxis]
                x_r = spec_r[..., np.newaxis]

                X = np.stack([x_l, x_r], axis=0).astype(np.float32)

                probs = self.model.predict(X, verbose=0)
                probs_mean = probs.mean(axis=0).astype(np.float32)

                self._ema_probs = (
                    cfg.smooth_alpha * self._ema_probs + (1.0 - cfg.smooth_alpha) * probs_mean
                )

                idx = int(np.argmax(self._ema_probs))
                label = LABELS[idx]

                tau = gcc_phat_tdoa(left, right, cfg.sample_rate)
                direction = tau_to_direction(tau, cfg) if label != "noise" else 0

                status = {
                    "sound": LABEL_TO_CHAR[label],
                    "direction": int(direction),
                }

                with self._lock:
                    self._latest = status
