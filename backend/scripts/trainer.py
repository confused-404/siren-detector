from __future__ import annotations

import argparse

import numpy as np
import tensorflow as tf
from helpers.create_model import (
    NUM_CLASSES,
    SpecCnnDropoutConfig,
    create_spec_cnn_with_custom_dropouts,
)
from helpers.middleman import training_data_from_manifest
from helpers.training import find_epochs, train_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


def _min_groups_per_class(labels: np.ndarray, groups: np.ndarray) -> int:
    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        raise ValueError("Training split requires at least one labeled example.")

    return min(
        len(np.unique(groups[labels == label]))
        for label in unique_labels
    )


def _choose_split_count(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    stage_name: str,
    minimum_required_groups: int,
    max_splits: int = 5,
) -> int:
    min_groups = _min_groups_per_class(labels, groups)
    if min_groups < minimum_required_groups:
        raise ValueError(
            f"{stage_name} split requires at least {minimum_required_groups} grouped clips per "
            f"class, but the dataset only has {min_groups}."
        )
    return min(max_splits, min_groups)


def _group_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    features = np.zeros(labels.shape[0], dtype=np.uint8)
    train_idx, test_idx = next(splitter.split(features, labels, groups))
    return train_idx, test_idx


def stratified_group_train_val_test_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    labels = y_train.argmax(axis=1)
    outer_splits = _choose_split_count(
        labels,
        groups,
        stage_name="Train/validation/test",
        minimum_required_groups=3,
    )

    train_full_idx, test_idx = _group_split_indices(
        labels,
        groups,
        n_splits=outer_splits,
        random_state=random_state,
    )

    x_train_full = x_train[train_full_idx]
    y_train_full = y_train[train_full_idx]
    train_full_groups = groups[train_full_idx]
    train_full_labels = labels[train_full_idx]
    inner_splits = _choose_split_count(
        train_full_labels,
        train_full_groups,
        stage_name="Train/validation",
        minimum_required_groups=2,
    )

    train_idx, val_idx = _group_split_indices(
        train_full_labels,
        train_full_groups,
        n_splits=inner_splits,
        random_state=random_state + 1,
    )

    x_train_split = x_train_full[train_idx]
    y_train_split = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    x_test = x_train[test_idx]
    y_test = y_train[test_idx]

    return x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the siren detector CNN.")
    parser.add_argument("--dataset-dir", default="3_2_test_dataset")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--peak-limit", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-version", type=int, default=0)
    parser.add_argument("--output-model", default="trained_car_alert_model.h5")
    parser.add_argument("--output-tflite", default="trained_car_alert_model.tflite")
    parser.add_argument("--dropout-conv1", type=float, default=0.2)
    parser.add_argument("--dropout-conv2", type=float, default=0.25)
    parser.add_argument("--dropout-conv3", type=float, default=0.3)
    parser.add_argument("--dropout-dense", type=float, default=0.35)
    return parser.parse_args()


def build_dropout_config(args: argparse.Namespace) -> SpecCnnDropoutConfig:
    return SpecCnnDropoutConfig(
        conv_block_1=args.dropout_conv1,
        conv_block_2=args.dropout_conv2,
        conv_block_3=args.dropout_conv3,
        dense_layer=args.dropout_dense,
    )


def set_training_seed(seed: int) -> None:
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)

    x_train, y_train, groups = training_data_from_manifest(
        dataset_dir=args.dataset_dir,
        shuffle=True,
        normalize=args.normalize,
        peak_limit=args.peak_limit,
    )
    print("x_train:", x_train.shape, x_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("First label:", y_train[0])

    x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test = (
        stratified_group_train_val_test_split(x_train, y_train, groups, random_state=args.seed)
    )

    print("train split:", x_train_split.shape, y_train_split.shape)
    print("validation split:", x_val.shape, y_val.shape)
    print("test split:", x_test.shape, y_test.shape)

    dropout_config = build_dropout_config(args)
    model = create_spec_cnn_with_custom_dropouts(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
        dropout_config=dropout_config,
    )

    best_epoch = find_epochs(
        model,
        (x_train_split, y_train_split),
        (x_val, y_val),
        version=args.history_version,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    print("Best epoch:", best_epoch)

    final_model = create_spec_cnn_with_custom_dropouts(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
        dropout_config=dropout_config,
    )
    train_model(final_model, (x_train_full, y_train_full), best_epoch)

    final_model.save(args.output_model)
    print(f"Saved {args.output_model}")

    converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
    tflite_model = converter.convert()
    with open(args.output_tflite, "wb") as tflite_file:
        tflite_file.write(tflite_model)
    print(f"Saved {args.output_tflite}")

    y_true = y_test.argmax(axis=1)
    y_pred = final_model.predict(x_test, verbose=0).argmax(axis=1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["siren", "honk", "noise"]))


if __name__ == "__main__":
    main()
