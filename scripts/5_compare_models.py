"""
STEP 5: FINAL COMPARISON
============================
Loads the classical baseline and the best CNN checkpoint, evaluates both
on the exact same held-out test set, and reports per-class performance
for each.
"""

import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# -----------------------------------------------------------------------
# 1. LOAD TEST DATA
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_test_flat = data["X_test"].reshape(data["X_test"].shape[0], -1)
X_test_images = data["X_test"].reshape(-1, 28, 28, 1)
y_test = data["y_test"]

# -----------------------------------------------------------------------
# 2. LOAD BOTH MODELS
# -----------------------------------------------------------------------
pca = joblib.load("outputs/pca_transformer.joblib")
classical_model = joblib.load("outputs/best_classical_model.joblib")
cnn_model = tf.keras.models.load_model("outputs/cnn_best.keras")

# -----------------------------------------------------------------------
# 3. PREDICT WITH BOTH
# -----------------------------------------------------------------------
X_test_pca = pca.transform(X_test_flat)
classical_preds = classical_model.predict(X_test_pca)

cnn_probs = cnn_model.predict(X_test_images, verbose=0)
cnn_preds = np.argmax(cnn_probs, axis=1)

# -----------------------------------------------------------------------
# 4. COMPARE
# -----------------------------------------------------------------------
classical_acc = accuracy_score(y_test, classical_preds)
cnn_acc = accuracy_score(y_test, cnn_preds)

print("=" * 60)
print("FINAL COMPARISON: Classical ML vs CNN")
print("=" * 60)
print(f"Classical model ({type(classical_model).__name__} + PCA): {classical_acc*100:.2f}%")
print(f"CNN (with data augmentation):                    {cnn_acc*100:.2f}%")
print("=" * 60)

winner = "CNN" if cnn_acc > classical_acc else (
    "Classical model" if classical_acc > cnn_acc else "Tie"
)
print(f"\nWinner: {winner} (by {abs(cnn_acc - classical_acc)*100:.2f} percentage points)\n")

# -----------------------------------------------------------------------
# 5. PER-CLASS BREAKDOWN FOR THE WINNER
# -----------------------------------------------------------------------
winner_preds = cnn_preds if winner == "CNN" else classical_preds
print(f"---- Per-class report for {winner} ----")
print(classification_report(y_test, winner_preds, target_names=CLASS_NAMES, digits=3))

print("---- Confusion matrix (which classes get mixed up) ----")
cm = confusion_matrix(y_test, winner_preds)
print("Rows = actual class, Columns = predicted class")
print("     " + "".join(f"{n[:4]:>6}" for n in CLASS_NAMES))
for i, row in enumerate(cm):
    print(f"{CLASS_NAMES[i]:<12}" + "".join(f"{v:>6}" for v in row))

# -----------------------------------------------------------------------
# 6. SAVE FULL RESULTS
# -----------------------------------------------------------------------
with open("outputs/final_comparison.txt", "w") as f:
    f.write("FINAL COMPARISON: Classical ML vs CNN\n")
    f.write("=" * 60 + "\n")
    f.write(f"Classical model ({type(classical_model).__name__} + PCA): {classical_acc*100:.2f}%\n")
    f.write(f"CNN (with data augmentation): {cnn_acc*100:.2f}%\n")
    f.write(f"Winner: {winner}\n\n")
    f.write(f"Per-class report for {winner}:\n")
    f.write(classification_report(y_test, winner_preds, target_names=CLASS_NAMES, digits=3))
    f.write("\nConfusion matrix (rows=actual, columns=predicted):\n")
    f.write("     " + "".join(f"{n[:4]:>6}" for n in CLASS_NAMES) + "\n")
    for i, row in enumerate(cm):
        f.write(f"{CLASS_NAMES[i]:<12}" + "".join(f"{v:>6}" for v in row) + "\n")

print("\nSaved full comparison to: outputs/final_comparison.txt")
