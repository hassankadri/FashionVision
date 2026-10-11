"""
STEP 1: EXPLORE AND CLEAN
============================
Even a well-known research dataset can have real quality issues:
duplicate images, corrupted (blank/constant) images, or missing files.
This script actually checks for these -- rather than assuming a "famous"
dataset is automatically clean -- and only "cleans" what it genuinely
finds.
"""

import numpy as np

data = np.load("data/processed/fashion_mnist.npz")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

# -----------------------------------------------------------------------
# 1. CHECK FOR MISSING / NaN VALUES
# -----------------------------------------------------------------------
print(f"\nNaN values in train set: {np.isnan(X_train.astype(float)).sum()}")

# -----------------------------------------------------------------------
# 2. CHECK FOR CORRUPTED IMAGES (all-blank or all-constant pixel values --
#    a real symptom of a bad scan or a sensor failure)
# -----------------------------------------------------------------------
flat_train = X_train.reshape(X_train.shape[0], -1)
constant_images = np.all(flat_train == flat_train[:, [0]], axis=1)
print(f"Fully blank/constant images found: {constant_images.sum()}")

# -----------------------------------------------------------------------
# 3. CHECK FOR EXACT DUPLICATE IMAGES
# -----------------------------------------------------------------------
# Hash each image so we can find exact duplicates efficiently, even
# across 60,000 images.
image_hashes = [hash(img.tobytes()) for img in X_train]
unique_hashes, counts = np.unique(image_hashes, return_counts=True)
n_duplicates = (counts > 1).sum()
print(f"Groups of exact duplicate images found: {n_duplicates}")

# -----------------------------------------------------------------------
# 4. CHECK CLASS BALANCE (already looked good in step 0, confirming here)
# -----------------------------------------------------------------------
class_counts = np.bincount(y_train)
print(f"\nClass balance (train): min={class_counts.min()}, max={class_counts.max()}")
print("-> Balanced. No class rebalancing needed.")

# -----------------------------------------------------------------------
# 5. NORMALIZE PIXEL VALUES
# -----------------------------------------------------------------------
# Neural networks train much better when inputs are small numbers
# (0-1) rather than 0-255. This is the one transformation this dataset
# genuinely needs.
X_train_norm = X_train.astype("float32") / 255.0
X_test_norm = X_test.astype("float32") / 255.0

# -----------------------------------------------------------------------
# 6. SAVE CLEANED / NORMALIZED DATA
# -----------------------------------------------------------------------
np.savez_compressed(
    "data/processed/fashion_mnist_clean.npz",
    X_train=X_train_norm, y_train=y_train,
    X_test=X_test_norm, y_test=y_test,
)
print("\nSaved normalized dataset to: data/processed/fashion_mnist_clean.npz")
