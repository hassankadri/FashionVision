"""
STEP 4: DETECT REAL LABEL ERRORS WITH CLEANLAB
==================================================
In v2, we manually injected fake label noise and could never perfectly
find it again (that was the point -- realistic label noise isn't neatly
flaggable). Here, the dataset is real, and we use cleanlab's "confident
learning" approach to find images whose given label likely doesn't match
what a well-trained model believes the image actually shows.

How this works, conceptually:
1. Get each training image's predicted class probabilities from a model
   that never saw that specific image during its own training (via
   cross-validation) -- this is essential, since a model evaluated on
   data it was trained on will just agree with the labels it memorized.
2. Cleanlab compares each image's GIVEN label against what the model's
   out-of-sample probabilities suggest, and flags images where the given
   label looks confidently wrong.
3. This doesn't "fix" the dataset -- it surfaces candidates for a human
   to review. Some flagged images may be genuinely mislabeled; others may
   just be unusually ambiguous, hard examples (e.g. a "Shirt" that
   legitimately looks a lot like a "Pullover").
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# -----------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------
data = np.load("data/processed/fashion_mnist_clean.npz")
X_train_images = data["X_train"]                       # keep the 28x28 images for visualization later
X_train = X_train_images.reshape(X_train_images.shape[0], -1)
y_train = data["y_train"]

print(f"Training set: {X_train.shape[0]} images")

# -----------------------------------------------------------------------
# 2. REDUCE DIMENSIONS (same reasoning as Step 2 -- speeds up the model
#    we'll use to generate predictions, without losing much information)
# -----------------------------------------------------------------------
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train)
print(f"PCA: {X_train.shape[1]} -> {X_train_pca.shape[1]} dimensions")

# -----------------------------------------------------------------------
# 3. GET OUT-OF-SAMPLE PREDICTED PROBABILITIES VIA CROSS-VALIDATION
# -----------------------------------------------------------------------
# cross_val_predict splits the data into folds and, for each fold, predicts
# using a model trained ONLY on the other folds -- so every prediction is
# genuinely out-of-sample, never "cheating" by having seen that image's
# label during training.
print("\nGetting cross-validated predictions (this takes a minute)...")
start = time.time()
model = LogisticRegression(max_iter=1000, random_state=42)
pred_probs = cross_val_predict(
    model, X_train_pca, y_train, cv=3, method="predict_proba"
)
print(f"Done in {time.time() - start:.1f}s")

# -----------------------------------------------------------------------
# 4. FIND LABEL ISSUES
# -----------------------------------------------------------------------
issue_indices = find_label_issues(
    labels=y_train,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",  # most confidently wrong first
)

print(f"\nCleanlab flagged {len(issue_indices)} out of {len(y_train)} training "
      f"images ({len(issue_indices) / len(y_train) * 100:.2f}%) as likely "
      f"label issues.")

# -----------------------------------------------------------------------
# 5. LOOK AT THE TOP FLAGGED EXAMPLES
# -----------------------------------------------------------------------
top_n = 15
top_issues = issue_indices[:top_n]

print(f"\nTop {top_n} most likely label errors:")
print(f"{'Index':<8}{'Given label':<15}{'Model believes':<15}{'Confidence in given label'}")
for idx in top_issues:
    given = CLASS_NAMES[y_train[idx]]
    model_believes = CLASS_NAMES[np.argmax(pred_probs[idx])]
    confidence = pred_probs[idx][y_train[idx]]
    print(f"{idx:<8}{given:<15}{model_believes:<15}{confidence:.3f}")

# -----------------------------------------------------------------------
# 6. VISUALIZE THE TOP FLAGGED IMAGES
# -----------------------------------------------------------------------
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
for ax, idx in zip(axes.flat, top_issues):
    ax.imshow(X_train_images[idx], cmap="gray")
    given = CLASS_NAMES[y_train[idx]]
    believes = CLASS_NAMES[np.argmax(pred_probs[idx])]
    ax.set_title(f"Given: {given}\nModel: {believes}", fontsize=9)
    ax.axis("off")
plt.suptitle("Top 15 suspected label issues (cleanlab)", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/suspected_label_issues.png", dpi=100)
print("\nSaved visualization to: outputs/suspected_label_issues.png")

# -----------------------------------------------------------------------
# 7. SAVE FULL RESULTS
# -----------------------------------------------------------------------
with open("outputs/label_issues_report.txt", "w") as f:
    f.write(f"Cleanlab label issue detection\n")
    f.write(f"Total training images: {len(y_train)}\n")
    f.write(f"Flagged as likely label issues: {len(issue_indices)} "
            f"({len(issue_indices) / len(y_train) * 100:.2f}%)\n\n")
    f.write(f"Top {top_n} most likely label errors:\n")
    f.write(f"{'Index':<8}{'Given label':<15}{'Model believes':<15}{'Confidence in given label'}\n")
    for idx in top_issues:
        given = CLASS_NAMES[y_train[idx]]
        model_believes = CLASS_NAMES[np.argmax(pred_probs[idx])]
        confidence = pred_probs[idx][y_train[idx]]
        f.write(f"{idx:<8}{given:<15}{model_believes:<15}{confidence:.3f}\n")

np.save("outputs/label_issue_indices.npy", issue_indices)
print("Saved full report to: outputs/label_issues_report.txt")
