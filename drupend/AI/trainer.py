from middleman import training_data_from_manifest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from training import find_epochs, train_model
from create_model import create_mlp_model, INPUT_DIMENSION, NUM_CLASSES, DROPOUT
from sklearn.metrics import confusion_matrix, classification_report

training_data = training_data_from_manifest(
    dataset_dir="3_2_test_dataset",
    shuffle=True,
    normalize=False,
    peak_limit=0.5,
)

x_train, y_train = training_data
print("x_train:", x_train.shape, x_train.dtype)
print("y_train:", y_train.shape, y_train.dtype)
print("First label:", y_train[0])

model = create_mlp_model(INPUT_DIMENSION, NUM_CLASSES, DROPOUT)

best_epoch = find_epochs(model, training_data, version=0, max_epochs=100, patience=3)
print("Best epoch:", best_epoch)

train_model(model, training_data, best_epoch)

model.save("trained_car_alert_model.h5")
print("Saved trained_car_alert_model.h5")

y_true = y_train.argmax(axis=1)

y_pred = model.predict(x_train, verbose=0).argmax(axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["siren", "honk", "noise"]))
