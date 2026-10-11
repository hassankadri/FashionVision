# Real-photo model

`model_quantized.onnx` is the quantized combined CLIP ViT-B/32 model from
[`Xenova/clip-vit-base-patch32`](https://huggingface.co/Xenova/clip-vit-base-patch32),
an ONNX conversion of OpenAI's CLIP model. It is used only for real-photo
uploads; the original Fashion-MNIST CNN remains responsible for the sample
gallery.

- Model file SHA-256: `0898a3facfdb27f0a041e57649b4989cfd094e4a0040d6ae75ed69917dfc7328`
- License: MIT (see the upstream model card)
- Reinstall/verify: `python scripts/download_photo_model.py`

The prompt-to-class mapping is versioned in `ml/photo_classifier.py`, including
the important distinction between a pullover/quarter-zip and a button-front
shirt.
