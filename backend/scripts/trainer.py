from helpers.create_model import NUM_CLASSES, create_spec_cnn_with_custom_dropouts
from helpers.middleman import training_data_from_manifest
from helpers.training import find_epochs, train_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

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

labels = y_train.argmax(axis=1)

x_train_full, x_test, y_train_full, y_test, labels_train_full, _ = train_test_split(
    x_train,
    y_train,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels,
)

x_train_split, x_val, y_train_split, y_val = train_test_split(
    x_train_full,
    y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=labels_train_full,
)

print("train split:", x_train_split.shape, y_train_split.shape)
print("validation split:", x_val.shape, y_val.shape)
print("test split:", x_test.shape, y_test.shape)

model = create_spec_cnn_with_custom_dropouts(input_shape=x_train.shape[1:], num_classes=NUM_CLASSES)

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
    input_shape=x_train.shape[1:], num_classes=NUM_CLASSES
)
train_model(final_model, (x_train_full, y_train_full), best_epoch)

final_model.save("trained_car_alert_model.h5")
print("Saved trained_car_alert_model.h5")

y_true = y_test.argmax(axis=1)

y_pred = final_model.predict(x_test, verbose=0).argmax(axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=["siren", "honk", "noise"]))
