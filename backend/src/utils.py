import numpy as np


def _frame_waveform(
    waveform: np.ndarray,
    *,
    frame_length: int,
    frame_step: int,
) -> np.ndarray:
    if frame_length <= 0:
        raise ValueError("frame_length must be positive")
    if frame_step <= 0:
        raise ValueError("frame_step must be positive")

    if waveform.size == 0:
        return np.zeros((1, frame_length), dtype=np.float32)

    frame_starts = np.arange(0, waveform.shape[0], frame_step, dtype=np.int32)
    total_length = int(frame_starts[-1] + frame_length)
    padded = np.pad(waveform, (0, max(0, total_length - waveform.shape[0])), mode="constant")
    indices = frame_starts[:, np.newaxis] + np.arange(frame_length, dtype=np.int32)
    return padded[indices]


def waveform_to_logspec(
    waveform_1d: np.ndarray,
    frame_length: int = 512,
    frame_step: int = 128,
    fft_length: int = 512,
) -> np.ndarray:
    """
    waveform_1d: shape (16000,)
    returns: log spectrogram, shape (time_frames, freq_bins)
    """
    waveform = np.asarray(waveform_1d, dtype=np.float32).reshape(-1)
    frames = _frame_waveform(waveform, frame_length=frame_length, frame_step=frame_step)
    window = np.hanning(frame_length).astype(np.float32)
    windowed = frames * window
    stft = np.fft.rfft(windowed, n=fft_length, axis=1)
    magnitude = np.abs(stft).astype(np.float32)
    return np.log(magnitude + 1e-6).astype(np.float32)
