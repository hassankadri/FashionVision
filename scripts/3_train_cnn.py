"""
STEP 3: CONVOLUTIONAL NEURAL NETWORK + DATA AUGMENTATION
=============================================================
Replaces v2's plain MLP (dense layers only) with a CNN, which is a much
better fit for image data: convolutional layers slide small filters
across the image and learn to detect local patterns (edges, curves,
textures) regardless of where they appear in the image. An MLP, in
contrast, treats every pixel as an independent, position-specific input
and has to re-learn the same pattern separately for every location.

DATA AUGMENTATION: small random rotations, shifts, and zooms are applied
to each training image on the fly, every epoch. The model never sees the
exact same image twice, which acts like having a bigger, more varied
training set and helps it generalize instead of memorizing.

RESUMABILITY: this sandbox has 1 CPU core and each epoch takes ~20-22s,
so a full training run (needing more epochs than fit in a single command)
is split across multiple invocations of this script. Each run:
  1. loads the checkpoint from the previous run, if one exists
  2. trains for a fixed number of additional epochs
  3. saves an updated checkpoint and appends to the training history log
Run this script repeatedly (e.g. via a small shell loop) until validation
accuracy stops improving.
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

CHECKPOINT_PATH = "outputs/cnn_checkpoint.keras"       # latest, for resuming training
BEST_CHECKPOINT_PATH = "outputs/cnn_best.keras"        # best validation accuracy seen so far
BEST_META_PATH = "outputs/cnn_best_meta.json"          # records the ACTUAL val_accuracy of cnn_best.keras
HISTORY_PATH = "outputs/cnn_history.json"
EPOCHS_PER_RUN = 5  # kept small: each epoch takes ~40s on this sandbox's single CPU core


class SaveEveryEpoch(keras.callbacks.Callback):
    """Saves the model and appends to the history log after EVERY epoch,
    not just at the end of the whole fit() call. This matters here because
    the training environment can be interrupted mid-run (e.g. a timeout) --
    without this, an interruption during epoch 8 of 10 would lose all 7
    completed epochs of progress instead of just the incomplete one.

    Also separately tracks the BEST validation accuracy checkpoint, with
    its true value stored in its own metadata file (BEST_META_PATH) --
    deliberately NOT inferred as max(history_log["val_accuracy"]). Only
    the latest epoch's weights are ever saved to CHECKPOINT_PATH, so if a
    better epoch happened earlier and wasn't the one saved as "best" at
    the time, its weights are gone even though its number lives on in the
    history log. Tracking the best value from a metadata file tied
    directly to the weights we actually still have avoids exactly the
    bug this project hit: seeding a "best so far" number from history
    that no longer corresponds to any saved weights, which then silently
    blocked every real future improvement from being recognized."""

    def __init__(self, history_log, best_so_far):
        super().__init__()
        self.history_log = history_log
        self.best_so_far = best_so_far

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key in ["accuracy", "val_accuracy", "loss", "val_loss"]:
            self.history_log[key].append(float(logs.get(key, float("nan"))))

        self.model.save(CHECKPOINT_PATH)

        val_acc = logs.get("val_accuracy", 0)
        if val_acc > self.best_so_far["val_accuracy"]:
            self.best_so_far["val_accuracy"] = val_acc
            self.model.save(BEST_CHECKPOINT_PATH)
            with open(BEST_META_PATH, "w") as f:
                json.dump({"val_accuracy": val_acc}, f)
            print(f"  [new best val_accuracy: {val_acc*100:.2f}% -- saved to {BEST_CHECKPOINT_PATH}]")

        with open(HISTORY_PATH, "w") as f:
            json.dump(self.history_log, f)
        print(f"  [checkpoint saved after epoch {len(self.history_log['accuracy'])}]")

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_train_full = data["X_train"].reshape(-1, 28, 28, 1)
y_train_full = data["y_train"]
X_test = data["X_test"].reshape(-1, 28, 28, 1)
y_test = data["y_test"]

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
        # No best-checkpoint metadata yet. Seed it with the CURRENT model's
        # own last-recorded val_accuracy (the weights we actually have),
        # not any historical max we can no longer reproduce.
        model.save(BEST_CHECKPOINT_PATH)
        seed_val = history_log["val_accuracy"][-1] if history_log["val_accuracy"] else 0.0
        best_so_far = {"val_accuracy": seed_val}
        with open(BEST_META_PATH, "w") as f:
            json.dump(best_so_far, f)
else:
    print("No checkpoint found -- building a new model")
    model = keras.Sequential([
        keras.layers.Input(shape=(28, 28, 1)),

        # --- Data augmentation (training-time only; skipped automatically
        #     during evaluation/prediction) ---
        keras.layers.RandomRotation(0.05),          # +/- ~18 degrees
        keras.layers.RandomTranslation(0.1, 0.1),   # shift up to 10% of image size
        keras.layers.RandomZoom(0.1),               # zoom in/out up to 10%

        # --- Convolutional feature extraction ---
        keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),

        # --- Classification head ---
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
    history_log = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    start_epoch = 0
    best_so_far = {"val_accuracy": 0.0}

model.summary()

# -----------------------------------------------------------------------
# 3. TRAIN FOR THIS RUN'S EPOCHS
# -----------------------------------------------------------------------
print(f"\nTraining epochs {start_epoch + 1} to {start_epoch + EPOCHS_PER_RUN}...")
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS_PER_RUN,
    batch_size=128,
    verbose=2,
    callbacks=[SaveEveryEpoch(history_log, best_so_far)],
)

# -----------------------------------------------------------------------
# 4. FINAL STATUS (checkpoint + history were already saved after each epoch)
# -----------------------------------------------------------------------
total_epochs = len(history_log["accuracy"])
best_val_acc = max(history_log["val_accuracy"])
print(f"\nTotal epochs trained so far: {total_epochs}")
print(f"Best validation accuracy so far: {best_val_acc * 100:.2f}%")
print(f"Latest validation accuracy: {history_log['val_accuracy'][-1] * 100:.2f}%")

# -----------------------------------------------------------------------
# 5. EVALUATE THE BEST CHECKPOINT ON THE HELD-OUT TEST SET
# -----------------------------------------------------------------------
best_model = keras.models.load_model(BEST_CHECKPOINT_PATH)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Test set accuracy (BEST checkpoint, val_accuracy={best_so_far['val_accuracy']*100:.2f}%): "
      f"{test_accuracy * 100:.2f}%")

with open("outputs/cnn_results.txt", "w") as f:
    f.write(f"Total epochs trained: {total_epochs}\n")
    f.write(f"Best validation accuracy: {best_val_acc * 100:.2f}%\n")
    f.write(f"Test set accuracy (best checkpoint): {test_accuracy * 100:.2f}%\n")
