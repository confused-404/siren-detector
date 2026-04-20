from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("tensorflow")
pytest.importorskip("sklearn")

from trainer import (
    _group_split_indices,
    _min_groups_per_class,
    stratified_group_train_val_test_split,
)


def test_group_split_keeps_each_clip_in_single_partition() -> None:
    x_train = np.zeros((30, 4, 4, 1), dtype=np.float32)
    y_train = np.zeros((30, 3), dtype=np.float32)
    groups = []

    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    for clip_idx, label in enumerate(labels):
        y_train[clip_idx * 2 : clip_idx * 2 + 2, label] = 1.0
        groups.extend([f"clip-{clip_idx}", f"clip-{clip_idx}"])

    groups_array = np.asarray(groups, dtype=str)
    labels_array = y_train.argmax(axis=1)

    train_full_idx, test_idx = _group_split_indices(
        labels_array,
        groups_array,
        n_splits=5,
        random_state=42,
    )

    x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test = (
        stratified_group_train_val_test_split(x_train, y_train, groups_array)
    )

    del x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test

    full_clip_ids = set(groups_array[train_full_idx])
    test_clip_ids = set(groups_array[test_idx])

    assert full_clip_ids.isdisjoint(test_clip_ids)
    assert len(full_clip_ids) + len(test_clip_ids) == 15


def test_min_groups_per_class_counts_unique_group_ids() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    groups = np.array(["a", "a", "b", "c", "d", "d"], dtype=str)

    assert _min_groups_per_class(labels, groups) == 1


def test_train_val_test_split_supports_three_groups_per_class() -> None:
    x_train = np.zeros((18, 4, 4, 1), dtype=np.float32)
    y_train = np.zeros((18, 3), dtype=np.float32)
    groups = []

    labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    for clip_idx, label in enumerate(labels):
        y_train[clip_idx * 2 : clip_idx * 2 + 2, label] = 1.0
        groups.extend([f"clip-{clip_idx}", f"clip-{clip_idx}"])

    (
        x_train_split,
        x_val,
        y_train_split,
        y_val,
        x_train_full,
        y_train_full,
        x_test,
        y_test,
    ) = stratified_group_train_val_test_split(x_train, y_train, np.asarray(groups, dtype=str))

    assert x_train_split.size > 0
    assert x_val.size > 0
    assert x_test.size > 0
    assert y_train_split.shape[1] == 3
    assert y_val.shape[1] == 3
    assert y_test.shape[1] == 3
    assert x_train_full.shape[0] == y_train_full.shape[0]


def test_train_val_test_split_rejects_too_few_groups_per_class() -> None:
    x_train = np.zeros((12, 4, 4, 1), dtype=np.float32)
    y_train = np.zeros((12, 3), dtype=np.float32)
    groups = []

    labels = [0, 0, 1, 1, 2, 2]
    for clip_idx, label in enumerate(labels):
        y_train[clip_idx * 2 : clip_idx * 2 + 2, label] = 1.0
        groups.extend([f"clip-{clip_idx}", f"clip-{clip_idx}"])

    with pytest.raises(ValueError, match="requires at least 3 grouped clips per class"):
        stratified_group_train_val_test_split(x_train, y_train, np.asarray(groups, dtype=str))
