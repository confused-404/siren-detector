# middleman.py
import os
import numpy as np
import pandas as pd

LABELS = ["siren", "honk", "noise"]
LABEL_TO_INDEX = {name: i for i, name in enumerate(LABELS)}

def _fix_length(audio_1d: np.ndarray, target_len: int = 16000) -> np.ndarray:
    a = np.asarray(audio_1d).astype(np.float32).reshape(-1)
    if a.shape[0] == target_len:
        return a
    if a.shape[0] > target_len:
        return a[:target_len]
    return np.pad(a, (0, target_len - a.shape[0]), mode="constant")

def _passes_peak_filter(audio: np.ndarray, peak_limit: float = 0.5) -> bool:
    """
    Returns True if the clip peak is within the allowed range.
    """
    return np.max(np.abs(audio)) < peak_limit

def _one_hot(label: str) -> np.ndarray:
    label = label.strip().lower()
    if label not in LABEL_TO_INDEX:
        raise ValueError(f"Unknown label '{label}'. Expected one of {LABELS}.")
    y = np.zeros((len(LABELS),), dtype=np.float32)
    y[LABEL_TO_INDEX[label]] = 1.0
    return y

def _split_stereo_to_examples(audio: np.ndarray, target_len: int = 16000) -> list[np.ndarray]:
    """
    Takes stereo audio and returns [left_1d, right_1d] as separate examples.
    Supported shapes:
      - (16000, 2)  -> samples x channels
      - (2, 16000)  -> channels x samples
      - (16000,)    -> treated as mono -> [mono]
    """
    a = np.asarray(audio)

    if a.ndim == 1:
        return [_fix_length(a, target_len)]

    if a.ndim != 2:
        raise ValueError(f"Unsupported audio shape {a.shape}; expected (N,2) or (2,N) or (N,)")

    # (samples, channels)
    if a.shape[1] == 2 and a.shape[0] != 2:
        left = _fix_length(a[:, 0], target_len)
        right = _fix_length(a[:, 1], target_len)
        return [left, right]

    # (channels, samples)
    if a.shape[0] == 2 and a.shape[1] != 2:
        left = _fix_length(a[0, :], target_len)
        right = _fix_length(a[1, :], target_len)
        return [left, right]

    # ambiguous
    raise ValueError(f"Ambiguous stereo shape {a.shape}; can't infer channel axis.")

def load_manifest_dataset_channels_as_examples(
    dataset_dir: str = "3_2_test_dataset",
    manifest_name: str = "manifest.csv",
    target_len: int = 16000,
    shuffle: bool = True,
    seed: int = 1337,
    normalize: bool = False,
    peak_limit: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads manifest and loads .npy stereo clips.
    For each clip, produces TWO training examples: left and right channel, same label.

    Returns:
      x_train: (N*2, target_len)
      y_train: (N*2, 3)
    """
    manifest_path = os.path.join(dataset_dir, manifest_name)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "file" not in df.columns or "event" not in df.columns:
        raise ValueError(f"Manifest must have columns ['file','event']. Found: {list(df.columns)}")

    print("Manifest rows:", len(df))
    print(df["event"].value_counts())

    print("Original distribution:")
    print(df["event"].value_counts())

    noise_df = df[df["event"] == "noise"]
    honk_df = df[df["event"] == "honk"]
    siren_df = df[df["event"] == "siren"]

    target = min(len(honk_df), len(siren_df))

    noise_df = noise_df.sample(n=target, random_state=seed)

    df = pd.concat([noise_df, honk_df, siren_df])

    print("After undersampling:")
    print(df["event"].value_counts())

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    x_list = []
    y_list = []

    kept = 0
    dropped = 0

    for i, row in df.iterrows():
        rel_path = str(row["file"])
        label = str(row["event"])

        if os.path.exists(rel_path):
            npy_path = rel_path
        else:
            npy_path = os.path.join(dataset_dir, rel_path)

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Clip not found (row {i}): {npy_path}")

        audio = np.load(npy_path)
        examples = _split_stereo_to_examples(audio, target_len=target_len)

        y = _one_hot(label)

        for ex in examples:
            if not _passes_peak_filter(ex, peak_limit):
                dropped += 1
                continue

            kept += 1

            if normalize:
                peak = np.max(np.abs(ex)) + 1e-9
                ex = ex / peak

            x_list.append(ex.astype(np.float32))
            y_list.append(y)

    print(f"Kept {kept} channel examples")
    print(f"Dropped {dropped} channel examples above {peak_limit}")

    x_train = np.stack(x_list, axis=0).astype(np.float32)
    y_train = np.stack(y_list, axis=0).astype(np.float32)
    return x_train, y_train

def training_data_from_manifest(**kwargs):
    return load_manifest_dataset_channels_as_examples(**kwargs)
