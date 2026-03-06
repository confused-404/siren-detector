import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#note:6000 samples first

INPUT_DIMENSION = 16000
NUM_CLASSES = 3
DROPOUT = 0.45 #0.3 OG

def create_mlp_model(input_dimension, num_classes, dropout):

    model = keras.Sequential([
        keras.Input(shape=(input_dimension,)),

        layers.Dense(2000, activation='relu', name='hidden_layer_1'),
        layers.Dropout(dropout),

        layers.Dense(500, activation='relu', name='hidden_layer_2'),
        layers.Dropout(dropout-0.1),

        layers.Dense(125, activation='relu', name='hidden_layer_3'),
        layers.Dropout(dropout-0.2),

        layers.Dense(25, activation='relu', name='hidden_layer_4'),
        layers.Dropout(dropout-0.3),

        layers.Dense(num_classes, activation='softmax', name='output_layer')
    ])


    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_spec_cnn(input_shape=(126, 257, 1), num_classes=3):
    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def create_spec_cnn_with_custom_dropouts(input_shape=(126, 257, 1), num_classes=3):
    def get_layer_dropout(layer: str) -> float:
        return float(input(f"Dropout for layer [{layer}]: "))

    model = keras.Sequential([
        keras.Input(shape=input_shape),

        layers.Conv2D(16, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(get_layer_dropout("conv_block_1")),

        layers.Conv2D(32, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(get_layer_dropout("conv_block_2")),

        layers.Conv2D(64, (3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(get_layer_dropout("conv_block_3")),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(get_layer_dropout("dense_layer")),

        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
