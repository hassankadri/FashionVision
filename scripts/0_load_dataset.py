"""
STEP 0: LOAD THE REAL DATASET
================================
Fashion-MNIST ships in the same binary "idx-ubyte" format as the original
MNIST dataset -- a compact format storing raw pixel bytes, not CSV. This
script decodes those files into numpy arrays we can work with directly.

No synthetic messiness is added here. Whatever imperfections exist in
this data (label ambiguity, visually confusing classes) are the dataset's
own real-world characteristics.
"""

import gzip
import struct
import numpy as np
import os

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)


def load_idx_images(path):
    """Decode an idx3-ubyte.gz file into a numpy array of shape (N, 28, 28)."""
    with gzip.open(path, "rb") as f:
        magic, n_images, rows, cols = struct.unpack(">IIII", f.read(16))
        buffer = f.read(rows * cols * n_images)
        images = np.frombuffer(buffer, dtype=np.uint8).reshape(n_images, rows, cols)
    return images


def load_idx_labels(path):
    """Decode an idx1-ubyte.gz file into a numpy array of shape (N,)."""
    with gzip.open(path, "rb") as f:
        magic, n_labels = struct.unpack(">II", f.read(8))
        buffer = f.read(n_labels)
        labels = np.frombuffer(buffer, dtype=np.uint8)
    return labels


CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# -----------------------------------------------------------------------
# LOAD TRAIN AND TEST SETS
# -----------------------------------------------------------------------
X_train = load_idx_images(f"{RAW_DIR}/train-images-idx3-ubyte.gz")
y_train = load_idx_labels(f"{RAW_DIR}/train-labels-idx1-ubyte.gz")
X_test = load_idx_images(f"{RAW_DIR}/t10k-images-idx3-ubyte.gz")
y_test = load_idx_labels(f"{RAW_DIR}/t10k-labels-idx1-ubyte.gz")

print(f"Train: {X_train.shape[0]} images, shape {X_train.shape[1:]} each")
print(f"Test:  {X_test.shape[0]} images, shape {X_test.shape[1:]} each")
print(f"Pixel value range: {X_train.min()} to {X_train.max()}")

print("\nClass distribution (train):")
for i, name in enumerate(CLASS_NAMES):
    count = (y_train == i).sum()
    print(f"  {i} - {name:<15}: {count}")

# -----------------------------------------------------------------------
# SAVE AS .npz (efficient for large image arrays -- CSV would be huge
# and slow for 70,000 x 784 pixel values)
# -----------------------------------------------------------------------
np.savez_compressed(
    f"{OUT_DIR}/fashion_mnist.npz",
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test,
)
print(f"\nSaved decoded dataset to: {OUT_DIR}/fashion_mnist.npz")
