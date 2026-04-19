from __future__ import annotations

import numpy as np
from helpers.create_model import NUM_CLASSES, create_spec_cnn_with_custom_dropouts
from helpers.middleman import training_data_from_manifest
from helpers.training import find_epochs, train_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


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

    train_full_idx, test_idx = _group_split_indices(
        labels,
        groups,
        n_splits=5,
        random_state=random_state,
    )

    x_train_full = x_train[train_full_idx]
    y_train_full = y_train[train_full_idx]
    train_full_groups = groups[train_full_idx]
    train_full_labels = labels[train_full_idx]

    train_idx, val_idx = _group_split_indices(
        train_full_labels,
        train_full_groups,
        n_splits=5,
        random_state=random_state + 1,
    )

    x_train_split = x_train_full[train_idx]
    y_train_split = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]
    x_test = x_train[test_idx]
    y_test = y_train[test_idx]

    return x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test


def main() -> None:
    x_train, y_train, groups = training_data_from_manifest(
        dataset_dir="3_2_test_dataset",
        shuffle=True,
        normalize=False,
        peak_limit=0.5,
    )
    print("x_train:", x_train.shape, x_train.dtype)
    print("y_train:", y_train.shape, y_train.dtype)
    print("First label:", y_train[0])

    x_train_split, x_val, y_train_split, y_val, x_train_full, y_train_full, x_test, y_test = (
        stratified_group_train_val_test_split(x_train, y_train, groups)
    )

    print("train split:", x_train_split.shape, y_train_split.shape)
    print("validation split:", x_val.shape, y_val.shape)
    print("test split:", x_test.shape, y_test.shape)

    model = create_spec_cnn_with_custom_dropouts(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
    )

    best_epoch = find_epochs(
        model,
        (x_train_split, y_train_split),
        (x_val, y_val),
        version=0,
        max_epochs=100,
        patience=5,
    )
    print("Best epoch:", best_epoch)

    final_model = create_spec_cnn_with_custom_dropouts(
        input_shape=x_train.shape[1:],
        num_classes=NUM_CLASSES,
    )
    train_model(final_model, (x_train_full, y_train_full), best_epoch)

    final_model.save("trained_car_alert_model.h5")
    print("Saved trained_car_alert_model.h5")

    y_true = y_test.argmax(axis=1)
    y_pred = final_model.predict(x_test, verbose=0).argmax(axis=1)

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["siren", "honk", "noise"]))


if __name__ == "__main__":
    main()
