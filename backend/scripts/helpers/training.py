import numpy as np
import pandas as pd
from tensorflow import keras

# TODO: shuffle training data before input


def find_epochs(
    model: keras.Model,
    training_data: tuple[np.ndarray, np.ndarray],
    validation_data: tuple[np.ndarray, np.ndarray],
    version: int = -1,
    max_epochs: int = 100,
    patience: int = 3,
) -> int:
    """
    model: tensorflow H5 model object
    training_data: tuple of (input, output) from func format_training_data
    validation_data: held-out validation split used only for epoch selection
    version: just for csv naming and visualization (if you want to save, put at > -1)
    """
    x_train, y_train = training_data
    x_val, y_val = validation_data

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,  # upper limit
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
    )

    if version > -1:
        history_df = pd.DataFrame(history.history)
        history_df["epoch"] = range(1, len(history_df) + 1)

        if "accuracy" not in history_df.columns:
            history_df["accuracy"] = float("nan")
        if "val_accuracy" not in history_df.columns:
            history_df["val_accuracy"] = float("nan")

        columns_to_save = ["epoch", "loss", "val_loss", "accuracy", "val_accuracy"]
        columns_to_save = [col for col in columns_to_save if col in history_df.columns]

        csv_filename = f"accuracy_history_v{version}"
        history_df[columns_to_save].to_csv(csv_filename, index=False)
        print(f"Training history saved to {csv_filename}")

    best_epoch = np.argmin(history.history["val_loss"]) + 1
    return best_epoch


def train_model(
    model: keras.Model,
    training_data: tuple[np.ndarray, np.ndarray],
    optimal_epochs: int,
) -> None:
    x_train, y_train = training_data
    model.fit(
        x_train,
        y_train,
        epochs=optimal_epochs,
    )


def format_training_data(inputs, outputs):
    """
    Each input array corresponds to one output array

    Input format: 1 second of samples from 1 mic
    Output format: [% confidence of siren, % confidence of honk, % confidence of noise]

    For sending to frontend: pick one with highest confidence
    """
    x_train = np.array(inputs).reshape(len(inputs), 16000).astype("float32")
    y_train = np.array(outputs).reshape(len(outputs), 3).astype("float32")
    return x_train, y_train
