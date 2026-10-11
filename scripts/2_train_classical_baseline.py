"""
STEP 2: CLASSICAL ML BASELINE
================================
This dataset (60,000 x 784 features) is much bigger than v2's dataset
(1,800 x 64 features). A full GridSearchCV across many models and many
hyperparameter combinations, like we did in v2, would take a very long
time here -- SVMs especially scale poorly with sample count.

Adaptations for this dataset size:
1. PCA (dimensionality reduction) before classical models -- keeps 95%
   of the meaningful variance in far fewer numbers, which speeds up
   every model that follows.
2. RandomizedSearchCV instead of GridSearchCV -- tries a fixed number of
   random combinations instead of every single one, giving most of the
   benefit for a fraction of the cost.
3. HistGradientBoostingClassifier instead of GradientBoostingClassifier
   -- scikit-learn's histogram-based implementation, built for exactly
   this kind of larger dataset.
4. Tuning and training run on a STRATIFIED 10,000-ROW SUBSAMPLE of the
   60,000 training images, not the full set -- this sandbox has a single
   CPU core, and a single Random Forest fit on the full data took over
   90 seconds; multiplied across cross-validation folds and search
   candidates for 3 models, the full-data version would take much longer
   than is practical here. The model is still EVALUATED on the complete,
   untouched 10,000-image test set, so the reported accuracy is honest.
   The training subsample bounds the cost of cross-validation.
"""

import time
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# -----------------------------------------------------------------------
# 1. LOAD CLEANED DATA
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_train_full = data["X_train"].reshape(data["X_train"].shape[0], -1)  # flatten 28x28 -> 784
y_train_full = data["y_train"]
X_test = data["X_test"].reshape(data["X_test"].shape[0], -1)
y_test = data["y_test"]

# Stratified subsample for tuning/training (see module docstring point 4)
X_train, _, y_train, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=10000, random_state=42, stratify=y_train_full,
)

print(f"Full train available: {X_train_full.shape} | Using subsample: {X_train.shape}")
print(f"Test (untouched, full size): {X_test.shape}")

# -----------------------------------------------------------------------
# 2. DIMENSIONALITY REDUCTION WITH PCA
# -----------------------------------------------------------------------
# Fit PCA on training data only (never on test data -- that would leak
# information about the test set into training).
print("\nFitting PCA...")
start = time.time()
pca = PCA(n_components=0.95, random_state=42)  # keep 95% of variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA reduced {X_train.shape[1]} pixels down to {X_train_pca.shape[1]} "
      f"components (95% variance retained), in {time.time() - start:.1f}s")

joblib.dump(pca, "outputs/pca_transformer.joblib")

# -----------------------------------------------------------------------
# 3. RANDOMIZED SEARCH ACROSS MODELS
# -----------------------------------------------------------------------
search_configs = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {"C": [0.01, 0.1, 1, 10]},
        "n_iter": 3,
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 150],
            "max_depth": [15, 25],
        },
        "n_iter": 3,
    },
    "Hist Gradient Boosting": {
        "model": HistGradientBoostingClassifier(random_state=42),
        "params": {
            "max_iter": [75, 100],
            "learning_rate": [0.1, 0.2],
        },
        "n_iter": 3,
    },
}

best_models = {}
cv_results = {}

for name, config in search_configs.items():
    print(f"\nTuning {name}...")
    start = time.time()

    search = RandomizedSearchCV(
        config["model"],
        config["params"],
        n_iter=config["n_iter"],
        cv=2,               # 2-fold, given this sandbox has a single CPU core
        scoring="accuracy",
        random_state=42,
    )
    search.fit(X_train_pca, y_train)

    elapsed = time.time() - start
    best_models[name] = search.best_estimator_
    cv_results[name] = search.best_score_

    print(f"  Best params: {search.best_params_}")
    print(f"  Best cross-validated accuracy: {search.best_score_ * 100:.2f}%")
    print(f"  Took {elapsed:.1f}s")

# -----------------------------------------------------------------------
# 4. PICK BEST MODEL AND EVALUATE ON HELD-OUT TEST SET
# -----------------------------------------------------------------------
best_name = max(cv_results, key=cv_results.get)
best_model = best_models[best_name]
test_accuracy = best_model.score(X_test_pca, y_test)

print(f"\nBest model: {best_name} (CV accuracy: {cv_results[best_name] * 100:.2f}%)")
print(f"Test set accuracy: {test_accuracy * 100:.2f}%")

# -----------------------------------------------------------------------
# 5. SAVE
# -----------------------------------------------------------------------
joblib.dump(best_model, "outputs/best_classical_model.joblib")

with open("outputs/classical_results.txt", "w") as f:
    f.write("Classical ML baseline (PCA + RandomizedSearchCV, 3-fold CV)\n")
    for name, acc in sorted(cv_results.items(), key=lambda x: -x[1]):
        f.write(f"  {name}: {acc * 100:.2f}%\n")
    f.write(f"\nBest model: {best_name}\n")
    f.write(f"Test set accuracy: {test_accuracy * 100:.2f}%\n")

print("\nSaved best model to: outputs/best_classical_model.joblib")
