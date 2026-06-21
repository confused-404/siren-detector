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
    audio_sample_rate: int = 48000
    model_sample_rate: int = 16000
    channels: int = 2
    block_seconds: float = 1.0
    hop_seconds: float = 0.25
    peak_limit: float = 0.5

    frame_length: int = 512
    frame_step: int = 128
    fft_length: int = 512

    mic_distance_m: float = 0.23 # TODO: measure and edit
    speed_of_sound: float = 343.24
    direction_deadband_deg: float = 15.0
    direction_center_deg: float = 10.0
    direction_side_deg: float = 20.0
    direction_conf_min: float = 1.55
    direction_history: int = 7
    gcc_phat_beta: float = 0.5
    gcc_bandpass_low_hz: float = 100.0
    gcc_bandpass_high_hz: float = 700.0
    alert_hold_seconds: float = 1.5

    smooth_alpha: float = 0.6

    device: Optional[int] = None
    
    arecord_device: str = "plughw:2,0"
    arecord_format: str = "S32_LE"

def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)

    source_len = audio.shape[0]
    if source_rate > target_rate and source_rate % target_rate == 0:
        ratio = source_rate // target_rate
        usable_len = source_len - (source_len % ratio)
        if usable_len > 0:
            return audio[:usable_len].reshape(-1, ratio).mean(axis=1).astype(np.float32)

    target_len = int(round(source_len * target_rate / source_rate))
    if source_len == 0 or target_len <= 0:
        return np.zeros((0,), dtype=np.float32)

    source_x = np.linspace(0.0, 1.0, source_len, endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)

def bandpass_fft(audio: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    low = max(0.0, low_hz)
    high = min(float(fs) / 2.0, high_hz)
    if low >= high:
        return audio.astype(np.float32, copy=False)

    windowed = audio.astype(np.float32, copy=False) * np.hanning(audio.shape[0]).astype(np.float32)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / fs)
    mask = (freqs >= low) & (freqs <= high)
    spectrum *= mask
    return np.fft.irfft(spectrum, n=audio.shape[0]).astype(np.float32)

def gcc_phat_tdoa(
    x: np.ndarray,
    y: np.ndarray,
    fs: int,
    max_tau: Optional[float] = None,
    phat_beta: float = 1.0,
):
    """
    Estimate TDOA using GCC-PHAT, constrained to a physically valid delay range.
    Returns (tau, peak_confidence).
    """
    n = 1
    L = len(x) + len(y)
    while n < L:
        n *= 2

    X = np.fft.rfft(x, n=n)
    Y = np.fft.rfft(y, n=n)
    R = X * np.conj(Y)

    denom = np.abs(R) ** phat_beta
    denom[denom < 1e-12] = 1e-12
    R /= denom

    cc = np.fft.irfft(R, n=n)
    cc = np.concatenate((cc[-(n // 2):], cc[:(n // 2)]))

    center = n // 2

    if max_tau is None:
        max_shift = center
    else:
        max_shift = min(int(fs * max_tau), center)

    cc_window = cc[center - max_shift : center + max_shift + 1]
    peak_idx = int(np.argmax(cc_window))
    interpolated_idx = float(peak_idx)
    if 0 < peak_idx < len(cc_window) - 1:
        y0, y1, y2 = cc_window[peak_idx - 1], cc_window[peak_idx], cc_window[peak_idx + 1]
        interp_denom = y0 - 2.0 * y1 + y2
        if abs(interp_denom) > 1e-12:
            interpolated_idx += 0.5 * (y0 - y2) / interp_denom

    rel_shift = interpolated_idx - max_shift
    tau = rel_shift / float(fs)

    # crude confidence: peak relative to mean abs correlation in valid window
    peak = float(np.max(np.abs(cc_window)))
    mean = float(np.mean(np.abs(cc_window)) + 1e-12)
    confidence = peak / mean

    return tau, confidence

def tau_to_theta(tau: float, cfg: DetectorConfig) -> float:
    s = (tau * cfg.speed_of_sound) / max(cfg.mic_distance_m, 1e-6)
    s = float(np.clip(s, -1.0, 1.0))
    return float(np.degrees(np.arcsin(s)))

def tau_to_direction(tau: float, cfg: DetectorConfig, last_direction: int = 0) -> int:
    """
    Convert time delay to left/center/right.
    Approximate angle using sin(theta)=tau*c/d. Clamp to [-1,1].
    """
    theta = tau_to_theta(tau, cfg)

    if theta <= -cfg.direction_side_deg:
        return 1
    if theta >= cfg.direction_side_deg:
        return -1
    if abs(theta) <= cfg.direction_center_deg:
        return 0
    return last_direction


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

        self._tau_hist = []
        self._last_direction = 0
        self._display_sound = "n"
        self._display_direction = 0
        self._last_alert_time = 0.0

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
        block_len = int(cfg.audio_sample_rate * cfg.block_seconds)
        hop_len = int(cfg.audio_sample_rate * cfg.hop_seconds)

        bytes_per_sample = 4  # S32_LE
        frame_bytes = hop_len * cfg.channels * bytes_per_sample

        cmd = [
            "arecord",
            "-D", cfg.arecord_device,
            "-f", cfg.arecord_format,
            "-r", str(cfg.audio_sample_rate),
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
        block_len = int(cfg.audio_sample_rate * cfg.block_seconds)

        print(
            f"CONFIG audio_fs={cfg.audio_sample_rate} "
            f"model_fs={cfg.model_sample_rate} "
            f"d={cfg.mic_distance_m} "
            f"max_tau={cfg.mic_distance_m / cfg.speed_of_sound * 1e6:.0f}us "
            f"max_shift={int(cfg.audio_sample_rate * cfg.mic_distance_m / cfg.speed_of_sound)}"
        )

        while self._running:
            time.sleep(cfg.hop_seconds)

            with self._audio_lock:
                if self._audio_buf.shape[0] < block_len:
                    continue
                window = self._audio_buf.copy()

            left_dir = window[:, 0].astype(np.float32)
            right_dir = window[:, 1].astype(np.float32)

            peak = max(float(np.max(np.abs(left_dir))), float(np.max(np.abs(right_dir))))
            if peak >= cfg.peak_limit:
                continue

            def standardize(spec: np.ndarray) -> np.ndarray:
                return (spec - spec.mean()) / (spec.std() + 1e-6)

            left_model = resample_linear(left_dir, cfg.audio_sample_rate, cfg.model_sample_rate)
            right_model = resample_linear(right_dir, cfg.audio_sample_rate, cfg.model_sample_rate)

            spec_l = waveform_to_logspec(left_model, cfg.frame_length, cfg.frame_step, cfg.fft_length)
            spec_r = waveform_to_logspec(right_model, cfg.frame_length, cfg.frame_step, cfg.fft_length)

            spec_l = standardize(spec_l)
            spec_r = standardize(spec_r)

            X = np.stack([spec_l[..., np.newaxis], spec_r[..., np.newaxis]], axis=0).astype(np.float32)

            probs = self.model.predict(X, verbose=0)
            probs_mean = probs.mean(axis=0).astype(np.float32)

            self._ema_probs = cfg.smooth_alpha * self._ema_probs + (1.0 - cfg.smooth_alpha) * probs_mean

            idx = int(np.argmax(self._ema_probs))
            label = LABELS[idx]

            max_tau = cfg.mic_distance_m / cfg.speed_of_sound
            left_gcc = bandpass_fft(
                left_dir,
                cfg.audio_sample_rate,
                cfg.gcc_bandpass_low_hz,
                cfg.gcc_bandpass_high_hz,
            )
            right_gcc = bandpass_fft(
                right_dir,
                cfg.audio_sample_rate,
                cfg.gcc_bandpass_low_hz,
                cfg.gcc_bandpass_high_hz,
            )
            tau, tau_conf = gcc_phat_tdoa(
                left_gcc,
                right_gcc,
                cfg.audio_sample_rate,
                max_tau=max_tau,
                phat_beta=cfg.gcc_phat_beta,
            )

            theta = tau_to_theta(tau, cfg)

            print(
                f"tau={tau*1e6:7.0f}us "
                f"theta={theta:6.1f} "
                f"conf={tau_conf:5.2f} "
                f"label={label}"
            )

            if tau_conf >= cfg.direction_conf_min:
                self._tau_hist.append(tau)
                self._tau_hist = self._tau_hist[-cfg.direction_history:]

            if len(self._tau_hist) >= 3:
                tau_for_dir = float(np.median(self._tau_hist))
                direction = tau_to_direction(tau_for_dir, cfg, self._last_direction)
                self._last_direction = direction
            else:
                direction = self._last_direction

            now = time.monotonic()
            sound = LABEL_TO_CHAR[label]
            if sound != "n":
                self._display_sound = sound
                self._display_direction = int(direction)
                self._last_alert_time = now
            elif now - self._last_alert_time > cfg.alert_hold_seconds:
                self._display_sound = "n"
                self._display_direction = int(direction)

            status = {"sound": self._display_sound, "direction": self._display_direction}
            with self._lock:
                self._latest = status
