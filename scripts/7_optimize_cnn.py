"""
STEP 7: OPTIMIZED CNN
========================
Same 2-conv-block architecture as Step 3, with three additions
aimed at increasing accuracy:

1. CLEANED LABELS: same 4,710 flagged images removed as in Step 6.
2. HORIZONTAL FLIP AUGMENTATION: valid for clothing images (unlike
   digits), and confirmed to add ~0 compute cost in testing.
3. LEARNING RATE SCHEDULING: ReduceLROnPlateau automatically lowers the
   learning rate when validation accuracy stalls, instead of training at
   a fixed rate the whole time -- aimed at the plateau-then-bounce
   pattern seen in Step 3's later epochs.

Uses new file names throughout (cnn_v2_*) so the original v1 CNN and
this optimized version can be compared side by side rather than one
overwriting the other.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.random.set_seed(42)
np.random.seed(42)

CHECKPOINT_PATH = "outputs/cnn_v2_checkpoint.keras"
BEST_CHECKPOINT_PATH = "outputs/cnn_v2_best.keras"
BEST_META_PATH = "outputs/cnn_v2_best_meta.json"
HISTORY_PATH = "outputs/cnn_v2_history.json"
EPOCHS_PER_RUN = 5


class SaveEveryEpoch(keras.callbacks.Callback):
    """Save each epoch and track the best checkpoint in dedicated metadata."""

    def __init__(self, history_log, best_so_far):
        super().__init__()
        self.history_log = history_log
        self.best_so_far = best_so_far

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in ["accuracy", "val_accuracy", "loss", "val_loss", "lr"]:
            value = logs.get(key)
            if value is None and key == "lr":
                value = float(self.model.optimizer.learning_rate.numpy())
            self.history_log.setdefault(key, []).append(float(value) if value is not None else float("nan"))

        self.model.save(CHECKPOINT_PATH)

        val_acc = logs.get("val_accuracy", 0)
        if val_acc > self.best_so_far["val_accuracy"]:
            self.best_so_far["val_accuracy"] = val_acc
            self.model.save(BEST_CHECKPOINT_PATH)
            with open(BEST_META_PATH, "w") as f:
                json.dump({"val_accuracy": val_acc}, f)
            print(f"  [new best val_accuracy: {val_acc*100:.2f}% -- saved]")

        with open(HISTORY_PATH, "w") as f:
            json.dump(self.history_log, f)
        print(f"  [checkpoint saved after epoch {len(self.history_log['accuracy'])}]")


# -----------------------------------------------------------------------
# 1. LOAD DATA AND REMOVE FLAGGED LABEL ISSUES
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_train_full_all = data["X_train"].reshape(-1, 28, 28, 1)
y_train_full_all = data["y_train"]
X_test = data["X_test"].reshape(-1, 28, 28, 1)
y_test = data["y_test"]

issue_indices = np.load("outputs/label_issue_indices.npy")
keep_mask = np.ones(len(y_train_full_all), dtype=bool)
keep_mask[issue_indices] = False
X_train_full = X_train_full_all[keep_mask]
y_train_full = y_train_full_all[keep_mask]
print(f"Training on {len(y_train_full)} cleaned images "
      f"(removed {len(issue_indices)} flagged label issues out of {len(y_train_full_all)})")

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full
)

# -----------------------------------------------------------------------
# 2. LOAD CHECKPOINT OR BUILD A NEW MODEL
# -----------------------------------------------------------------------
if os.path.exists(CHECKPOINT_PATH):
    print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
    model = keras.models.load_model(CHECKPOINT_PATH)
    with open(HISTORY_PATH) as f:
        history_log = json.load(f)
    start_epoch = len(history_log["accuracy"])
    if os.path.exists(BEST_META_PATH):
        with open(BEST_META_PATH) as f:
            best_so_far = json.load(f)
    else:
        model.save(BEST_CHECKPOINT_PATH)
        seed_val = history_log["val_accuracy"][-1] if history_log["val_accuracy"] else 0.0
        best_so_far = {"val_accuracy": seed_val}
        with open(BEST_META_PATH, "w") as f:
            json.dump(best_so_far, f)
else:
    print("No checkpoint found -- building a new model")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        # Data augmentation (training-time only)
        keras.layers.RandomFlip("horizontal"),       # NEW: valid for clothing, ~free
        keras.layers.RandomRotation(0.05),
        keras.layers.RandomTranslation(0.1, 0.1),
        keras.layers.RandomZoom(0.1),

        # Convolutional feature extraction (same as v1 -- see Entry 8 for
        # why a deeper network was tested and rejected on compute grounds)
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        # Classification head
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_log = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": [], "lr": []}
    start_epoch = 0
    best_so_far = {"val_accuracy": 0.0}

model.summary()

# -----------------------------------------------------------------------
# 3. TRAIN, WITH LEARNING RATE SCHEDULING
# -----------------------------------------------------------------------
# NEW: automatically halves the learning rate if validation accuracy
# hasn't improved for 3 epochs, instead of training at a fixed rate the
# whole time. Helps settle into a plateau more precisely rather than
# bouncing around it (see Entry 8).
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=3, min_lr=1e-5, verbose=1
)

print(f"\nTraining epochs {start_epoch + 1} to {start_epoch + EPOCHS_PER_RUN}...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PER_RUN,
    batch_size=128,
    verbose=2,
    callbacks=[SaveEveryEpoch(history_log, best_so_far), reduce_lr],
)

# -----------------------------------------------------------------------
# 4. EVALUATE THE BEST CHECKPOINT ON THE HELD-OUT TEST SET
# -----------------------------------------------------------------------
total_epochs = len(history_log["accuracy"])
best_val_acc = max(history_log["val_accuracy"])
print(f"\nTotal epochs trained so far: {total_epochs}")
print(f"Best validation accuracy so far: {best_val_acc * 100:.2f}%")

best_model = keras.models.load_model(BEST_CHECKPOINT_PATH)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test set accuracy (BEST checkpoint): {test_accuracy * 100:.2f}%")
print("(Step 3's original CNN scored 89.05% on this same test set)")

with open("outputs/cnn_v2_results.txt", "w") as f:
    f.write(f"Total epochs trained: {total_epochs}\n")
    f.write(f"Best validation accuracy: {best_val_acc * 100:.2f}%\n")
    f.write(f"Test set accuracy (best checkpoint): {test_accuracy * 100:.2f}%\n")
    f.write(f"Previous version (v1, uncleaned labels, no flip/LR schedule): 89.05%\n")
