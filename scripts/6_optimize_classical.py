"""
STEP 6: OPTIMIZED CLASSICAL MODEL
====================================
Two changes from Step 2, both aimed at increasing accuracy:

1. CLEANED LABELS: removes the 4,710 training images cleanlab flagged as
   likely mislabeled (Step 4). Training on wrong labels teaches the model
   incorrect decision boundaries; removing them should help most exactly
   on the classes that were most confused (Shirt/Pullover/Coat/T-shirt).

2. FULL TRAINING SET: Step 2 tuned AND trained on only a 10,000-row
   subsample, because searching hyperparameters across 3 models was too
   slow otherwise on this sandbox's single CPU core. This script reuses
   the best hyperparameters already found (no repeated search needed) and
   fits once on the full, cleaned training set -- more data, real labels.
"""

import time
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------------
# 1. LOAD DATA AND REMOVE FLAGGED LABEL ISSUES
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_train_full = data["X_train"].reshape(data["X_train"].shape[0], -1)
y_train_full = data["y_train"]
X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
y_test = data["y_test"]

issue_indices = np.load("outputs/label_issue_indices.npy")
print(f"Removing {len(issue_indices)} flagged label issues out of {len(y_train_full)} training images")

keep_mask = np.ones(len(y_train_full), dtype=bool)
keep_mask[issue_indices] = False
X_train = X_train_full[keep_mask]
y_train = y_train_full[keep_mask]
print(f"Training on {len(y_train)} cleaned images (was {len(y_train_full)})")

# -----------------------------------------------------------------------
# 2. PCA (refit on the full cleaned set, not the old 10k subsample)
# -----------------------------------------------------------------------
start = time.time()
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} dims, in {time.time()-start:.1f}s")

# -----------------------------------------------------------------------
# 3. FIT WITH PREVIOUSLY-FOUND BEST HYPERPARAMETERS (no repeated search)
# -----------------------------------------------------------------------
print("\nFitting HistGradientBoostingClassifier on full cleaned data...")
start = time.time()
model = HistGradientBoostingClassifier(
    max_iter=100, learning_rate=0.1, random_state=42
)
model.fit(X_train_pca, y_train)
print(f"Fit took {time.time()-start:.1f}s")

# -----------------------------------------------------------------------
# 4. EVALUATE ON THE SAME HELD-OUT TEST SET AS BEFORE
# -----------------------------------------------------------------------
test_accuracy = accuracy_score(y_test, model.predict(X_test_pca))
print(f"\nTest set accuracy: {test_accuracy * 100:.2f}%")
print("(Step 2's version, trained on a 10,000-row subsample, scored 84.87%)")

# -----------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------
joblib.dump(model, "outputs/best_classical_model_v2.joblib")
joblib.dump(pca, "outputs/pca_transformer_v2.joblib")

with open("outputs/classical_results_v2.txt", "w") as f:
    f.write("Optimized classical model: HistGradientBoosting, full cleaned training set\n")
    f.write(f"Training images used: {len(y_train)} (removed {len(issue_indices)} flagged label issues)\n")
    f.write(f"Test set accuracy: {test_accuracy * 100:.2f}%\n")
    f.write("Previous version (10k subsample, uncleaned labels): 84.87%\n")

print("Saved to: outputs/best_classical_model_v2.joblib")
