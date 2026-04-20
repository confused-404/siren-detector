from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("tensorflow")
pytest.importorskip("pandas")

from helpers.middleman import (
    _fix_length,
    _one_hot,
    _passes_peak_filter,
    _split_stereo_to_examples,
    load_manifest_dataset_channels_as_examples,
)


def test_fix_length_pads_and_truncates_to_target_size() -> None:
    padded = _fix_length(np.array([1.0, 2.0], dtype=np.float32), target_len=4)
    truncated = _fix_length(np.arange(6, dtype=np.float32), target_len=4)

    assert padded.tolist() == [1.0, 2.0, 0.0, 0.0]
    assert truncated.tolist() == [0.0, 1.0, 2.0, 3.0]


def test_one_hot_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Unknown label"):
        _one_hot("ambulance")


def test_passes_peak_filter_uses_strict_upper_bound() -> None:
    assert bool(_passes_peak_filter(np.array([0.1, -0.49], dtype=np.float32), peak_limit=0.5))
    assert not bool(_passes_peak_filter(np.array([0.1, -0.5], dtype=np.float32), peak_limit=0.5))


def test_split_stereo_to_examples_supports_layouts_and_rejects_ambiguous_shape() -> None:
    samples_by_channels = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    channels_by_samples = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)

    sample_split = _split_stereo_to_examples(samples_by_channels, target_len=3)
    channel_split = _split_stereo_to_examples(channels_by_samples, target_len=3)

    assert [example.tolist() for example in sample_split] == [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    assert [example.tolist() for example in channel_split] == [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]

    with pytest.raises(ValueError, match="Ambiguous stereo shape"):
        _split_stereo_to_examples(np.zeros((2, 2), dtype=np.float32), target_len=2)


def _write_manifest(dataset_dir: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(dataset_dir / "manifest.csv", index=False)


def test_load_manifest_dataset_channels_as_examples_builds_grouped_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir()

    siren_clip = clips_dir / "siren.npy"
    honk_clip = clips_dir / "honk.npy"
    noise_clip = clips_dir / "noise.npy"

    np.save(siren_clip, np.array([[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]], dtype=np.float32))
    np.save(honk_clip, np.array([[0.2, 0.1], [0.4, 0.3], [0.1, 0.2]], dtype=np.float32))
    np.save(noise_clip, np.array([[0.05, 0.04], [0.03, 0.02], [0.01, 0.02]], dtype=np.float32))

    _write_manifest(
        dataset_dir,
        [
            {"file": "clips/siren.npy", "event": "siren"},
            {"file": "clips/honk.npy", "event": "honk"},
            {"file": "clips/noise.npy", "event": "noise"},
        ],
    )

    monkeypatch.setattr(
        "helpers.middleman.waveform_to_logspec",
        lambda ex: np.full((2, 3), np.max(ex, initial=0.0), dtype=np.float32),
    )

    x_train, y_train, groups = load_manifest_dataset_channels_as_examples(
        dataset_dir=str(dataset_dir),
        target_len=3,
        shuffle=False,
        normalize=False,
        peak_limit=0.5,
    )

    assert x_train.shape == (6, 2, 3, 1)
    assert y_train.shape == (6, 3)
    assert groups.shape == (6,)
    assert set(groups.tolist()) == {str(siren_clip), str(honk_clip), str(noise_clip)}
    assert [int(x) for x in y_train.sum(axis=0)] == [2, 2, 2]


def test_load_manifest_dataset_channels_as_examples_raises_when_manifest_columns_missing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    _write_manifest(dataset_dir, [{"file": "clips/example.npy", "direction": "left"}])

    with pytest.raises(ValueError, match="Manifest must have columns"):
        load_manifest_dataset_channels_as_examples(dataset_dir=str(dataset_dir), shuffle=False)


def test_load_manifest_dataset_channels_as_examples_raises_when_all_examples_filtered_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    clips_dir = dataset_dir / "clips"
    clips_dir.mkdir()

    for label in ("siren", "honk", "noise"):
        np.save(clips_dir / f"{label}.npy", np.full((3, 2), 0.9, dtype=np.float32))

    _write_manifest(
        dataset_dir,
        [
            {"file": "clips/siren.npy", "event": "siren"},
            {"file": "clips/honk.npy", "event": "honk"},
            {"file": "clips/noise.npy", "event": "noise"},
        ],
    )

    monkeypatch.setattr(
        "helpers.middleman.waveform_to_logspec",
        lambda ex: np.ones((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="No training examples remained"):
        load_manifest_dataset_channels_as_examples(
            dataset_dir=str(dataset_dir),
            target_len=3,
            shuffle=False,
            peak_limit=0.5,
        )
