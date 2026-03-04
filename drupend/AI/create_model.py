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


mlp_model = create_mlp_model(INPUT_DIMENSION, NUM_CLASSES, DROPOUT)

mlp_model.save("car_alert_model.h5")
# Display the model summary
mlp_model.summary()
print("Made and Saved")
