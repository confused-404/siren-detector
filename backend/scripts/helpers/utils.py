import numpy as np
import tensorflow as tf

def waveform_to_logspec(waveform_1d: np.ndarray,
                        frame_length: int = 512,
                        frame_step: int = 128,
                        fft_length: int = 512) -> np.ndarray:
    """
    waveform_1d: shape (16000,)
    returns: log spectrogram, shape (time_frames, freq_bins)
    """
    w = tf.convert_to_tensor(waveform_1d, dtype=tf.float32)
    stft = tf.signal.stft(
        w,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=True
    )  # (frames, fft_bins)
    mag = tf.abs(stft)
    logmag = tf.math.log(mag + 1e-6)
    return logmag.numpy().astype(np.float32)

