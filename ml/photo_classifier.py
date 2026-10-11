"""Real-photo clothing classification with a quantized CLIP ONNX model.

The original project CNN is intentionally kept for Fashion-MNIST samples.
It is not used here: its 28x28 grayscale training distribution does not match
real product photography.  CLIP provides a zero-shot, colour-photo path for
the upload UI while preserving the project's ten public class names.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "clip-vit-base-patch32"
MODEL_PATH = MODEL_DIR / "model_quantized.onnx"
VOCAB_PATH = MODEL_DIR / "vocab.json"
MERGES_PATH = MODEL_DIR / "merges.txt"

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# Multiple concrete descriptions make the mapping from broad natural-language
# concepts to Fashion-MNIST's terse taxonomy explicit.  In particular, a
# quarter-zip fleece is a pullover; a shirt prompt is limited to button-front
# garments so that the two classes do not collapse into "long-sleeve top".
CLASS_PROMPTS: dict[str, tuple[str, ...]] = {
    "T-shirt/top": (
        "a product photo of a short-sleeve t-shirt",
        "a product photo of a sleeveless top",
    ),
    "Trouser": (
        "a product photo of trousers",
        "a product photo of pants",
    ),
    "Pullover": (
        "a product photo of a pullover sweater",
        "a product photo of a fleece pullover",
        "a product photo of a quarter-zip pullover",
        "a product photo of a sweatshirt",
    ),
    "Dress": (
        "a product photo of a dress",
        "a product photo of a one-piece dress",
    ),
    "Coat": (
        "a product photo of a coat",
        "a product photo of a full-zip jacket",
    ),
    "Sandal": (
        "a product photo of a sandal",
        "a product photo of an open-toe shoe",
    ),
    "Shirt": (
        "a product photo of a button-front shirt",
        "a product photo of a collared dress shirt",
        "a product photo of a blouse",
    ),
    "Sneaker": (
        "a product photo of a sneaker",
        "a product photo of an athletic shoe",
    ),
    "Bag": (
        "a product photo of a handbag",
        "a product photo of a bag",
    ),
    "Ankle boot": (
        "a product photo of an ankle boot",
        "a product photo of a short boot",
    ),
}


def _bytes_to_unicode() -> dict[int, str]:
    """Byte-to-unicode table used by OpenAI's CLIP/GPT-2 BPE tokenizer."""
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1))
    visible += list(range(ord("®"), ord("ÿ") + 1))
    chars = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(visible, map(chr, chars)))


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(zip(word, word[1:]))


class ClipTokenizer:
    """Small dependency-free implementation of the tokenizer CLIP expects."""

    def __init__(self, vocab_path: Path = VOCAB_PATH, merges_path: Path = MERGES_PATH):
        with vocab_path.open(encoding="utf-8") as handle:
            self.encoder: dict[str, int] = json.load(handle)
        with merges_path.open(encoding="utf-8") as handle:
            merge_lines = handle.read().splitlines()[1:]
        merges = [tuple(line.split()) for line in merge_lines if line.strip()]
        self.ranks = {pair: rank for rank, pair in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self.start_id = self.encoder["<|startoftext|>"]
        self.end_id = self.encoder["<|endoftext|>"]
        self.cache: dict[str, str] = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = re.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|"
            r"[a-z]+|[0-9]|[^\s\w]+",
            re.IGNORECASE,
        )

    def _bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.ranks.get(pair, math.inf))
            if bigram not in self.ranks:
                break
            first, second = bigram
            merged: list[str] = []
            index = 0
            while index < len(word):
                try:
                    next_match = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_match])
                index = next_match
                if index < len(word) - 1 and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = _pairs(word)

        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode_batch(self, texts: Iterable[str], length: int = 77) -> tuple[np.ndarray, np.ndarray]:
        rows: list[list[int]] = []
        for text in texts:
            clean = html.unescape(html.unescape(text)).strip().lower()
            clean = " ".join(clean.split())
            ids: list[int] = []
            for token in self.pattern.findall(clean):
                encoded = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
                ids.extend(self.encoder[piece] for piece in self._bpe(encoded).split(" "))
            rows.append(([self.start_id] + ids[: length - 2] + [self.end_id]))

        input_ids = np.zeros((len(rows), length), dtype=np.int64)
        attention_mask = np.zeros_like(input_ids)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = row
            attention_mask[row_index, : len(row)] = 1
        return input_ids, attention_mask


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Match CLIP's resize, centre-crop, channel order, and normalization."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    scale = 224 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))
    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    pixels = (pixels - mean) / std
    return np.transpose(pixels, (2, 0, 1))[None, ...].astype(np.float32)


@lru_cache(maxsize=1)
def _session() -> ort.InferenceSession:
    missing = [path for path in (MODEL_PATH, VOCAB_PATH, MERGES_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Real-photo model is not installed. Run: python scripts/download_photo_model.py"
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"])


@lru_cache(maxsize=1)
def _prompt_inputs() -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    labels: list[str] = []
    prompts: list[str] = []
    for label in CLASS_NAMES:
        for prompt in CLASS_PROMPTS[label]:
            labels.append(label)
            prompts.append(prompt)
    input_ids, attention_mask = ClipTokenizer().encode_batch(prompts)
    return labels, prompts, input_ids, attention_mask


def classify_image(image: Image.Image) -> dict[str, object]:
    labels, _prompts, input_ids, attention_mask = _prompt_inputs()
    outputs = _session().run(
        ["logits_per_image"],
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": preprocess_image(image),
        },
    )[0][0]

    # Each prompt is an alternate (disjunctive) description of a class, so use
    # its strongest matching description. Averaging would penalize classes
    # that deliberately have more varied forms (for example, pullover includes
    # sweater, sweatshirt, fleece, and quarter-zip variants).
    class_logits = np.array(
        [np.max([outputs[i] for i, candidate in enumerate(labels) if candidate == label]) for label in CLASS_NAMES],
        dtype=np.float64,
    )
    class_logits -= class_logits.max()
    probabilities = np.exp(class_logits)
    probabilities /= probabilities.sum()
    ranked = sorted(
        (
            {"label": label, "probability": float(probabilities[index])}
            for index, label in enumerate(CLASS_NAMES)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    top_margin = ranked[0]["probability"] - ranked[1]["probability"]
    return {
        "prediction": ranked[0]["label"],
        "confidence": ranked[0]["probability"],
        "uncertain": bool(ranked[0]["probability"] < 0.60 or top_margin < 0.20),
        "predictions": ranked,
        "model": "CLIP ViT-B/32 (quantized ONNX, zero-shot)",
    }


def classify_path(path: os.PathLike[str] | str) -> dict[str, object]:
    with Image.open(path) as image:
        return classify_image(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify a real clothing photo.")
    parser.add_argument("image")
    args = parser.parse_args()
    print(json.dumps(classify_path(args.image), indent=2))
