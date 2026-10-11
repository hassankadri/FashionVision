# Separate RGB fashion-image dataset pipeline

This package prepares existing RGB fashion photos for future work. It uses the
repository's existing Pillow and NumPy dependencies. It imports no model, API,
Fashion-MNIST, Cleanlab, or classical_v3 code. No dataset is bundled/downloaded,
no model is trained, and original files are never rewritten, moved or deleted.

## Expected layout

```text
dataset/
  train/<class-folder>/<image-file>
  validation/<class-folder>/<image-file>
  test/<class-folder>/<image-file>
```

All three split directories are required and must contain valid images. Nested
directories below a class folder are supported. Existing split membership is
preserved; this package never randomly resplits the supplied dataset. Recognized
extensions are JPG/JPEG, PNG, BMP, WEBP, TIF/TIFF and GIF (case-insensitive).
Multi-frame/animated images are rejected. Non-image extensions are listed as
ignored sidecars; they are not automatically inspected as images. Files directly
inside a split, unmapped classes, redirected/symlinked files or directories,
missing splits, empty splits, and missing training classes fail the audit.

## Class mapping

By default, class IDs come from sorted **training folder names only**. Validation
and test cannot introduce classes. Supply an explicit mapping to control ID
order, display names or several folder aliases for one class:

```json
{
  "class_names": ["Footwear", "Clothing"],
  "folders": {"shoes": 0, "shirts": 1, "dresses": 1}
}
```

IDs are contiguous integers starting at zero. Every declared class needs a
folder mapping. This mapping is independent of Fashion-MNIST and CLIP classes.

## Read-only audit and sample loading

```text
python -B -m fashion_rgb_data <existing-dataset-directory>
python -B -m fashion_rgb_data <existing-dataset-directory> --mapping <existing-mapping.json>
```

The CLI prints a JSON audit to stdout; it creates no report file. It exits 1 if
the audit finds blocking quality problems, and 0 otherwise. Mapping/layout
configuration errors also exit nonzero. No automatic repairs occur.

```python
from fashion_rgb_data import ClassMapping, RGBDataset, Transform, scan_dataset

mapping = ClassMapping(("Footwear", "Clothing"), {"shoes": 0, "shirts": 1, "dresses": 1})
audit = scan_dataset(dataset_path, mapping)
transform = Transform(size=(224, 224))  # height, width; no model-specific assumptions
train = RGBDataset(audit, "train", transform=transform, augment=True, seed=42)
validation = RGBDataset(audit, "validation", transform=transform)
test = RGBDataset(audit, "test", transform=transform)
sample = train.get(0, epoch=0)
# sample: image (HWC float32), class_id, class_name, original identity metadata
```

Loading refuses failed audits. Decode and EXIF orientation happen in memory;
grayscale/palette/CMYK inputs become RGB and transparency is composited on white.
Images are directly resized with bilinear resampling (aspect ratio may change).
Pixels become float32 via `/255`, then `(pixels - mean) / std` using configurable
three-channel constants. Defaults are mean=(0,0,0), std=(1,1,1), giving [0,1]. No
statistics are fitted on any split. Choose constants appropriate to a future
model explicitly; none are inferred from Fashion-MNIST or CLIP.

Training augmentation is opt-in: horizontal flip (default probability 0.5) and
brightness multiplier (default range 0.9–1.1). Both are configurable and operate
in memory. RNG derives from seed, epoch and original relative path, so repeated
access is reproducible independent of access order. Use `get(i, epoch=N)` to
vary augmentation by epoch. `dataset[i]` uses epoch zero. Validation/test reject
augmentation and return identical transforms regardless of epoch. Domain-specific
decisions such as whether mirroring text/logos is appropriate remain with callers.

## Quality checks and original identities

Each image record keeps original relative path, source split/class folder,
mapped label, byte count, original mode/dimensions/format, oriented RGB size,
original-byte SHA-256 and full-resolution decoded RGB SHA-256. Root location and
class mapping accompany the records. Corrupt images remain listed as INVALID;
they are never silently removed. Loading checks the original byte hash again
and refuses changed or redirected files. Rescan after any intentional dataset
change; already-created dataset objects do not discover newly added files.

Duplicate groups identify exact matching bytes and identical displayed RGB
pixels even when encoding/metadata differs. Cross-split duplicate groups are
leakage errors; conflicting mapped labels for identical images also fail.
Same-class duplicates within one split are reported without automatic removal.
One duplicate group may appear under both hash types: do not sum group counts
as a unique image count. This does not detect approximate matches, crops, lossy
re-encodings, or subject/session overlap; those require a separate policy and
source metadata. Passing the audit is not proof against every kind of leakage.

Statistics contain per-split valid/invalid image totals, every mapped class's
count and proportion, missing classes, and max/min class-count ratio. Ratio is
null if any class has zero valid images. Missing validation/test classes are
visible in statistics but do not independently block loading. No sampling,
rebalancing, deduplication or relabeling is performed.

## Tests

```text
python -B -m unittest discover -s tests -p test_fashion_rgb_data.py -v
```

Tests generate small temporary synthetic image fixtures only. They do not
constitute or persist a real-world dataset. No application/ML pipeline is changed.
