from __future__ import annotations

import numpy as np

from utils import waveform_to_logspec

def test_waveform_to_logspec_returns_float32_2d_array() -> None:
    waveform = np.sin(np.linspace(0, 8 * np.pi, 16000, dtype=np.float32))

    spec = waveform_to_logspec(
        waveform,
        frame_length=256,
        frame_step=128,
        fft_length=256,
    )

    assert spec.ndim == 2
    assert spec.dtype == np.float32
    assert spec.shape[1] == 129
    assert np.isfinite(spec).all()

def test_waveform_to_logspec_handles_silence_without_infinities() -> None:
    waveform = np.zeros(16000, dtype=np.float32)

    spec = waveform_to_logspec(waveform)

    assert np.isfinite(spec).all()
    assert np.all(spec <= 0.0)
