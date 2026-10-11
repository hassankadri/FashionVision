"""Class-specific Grad-CAM, using the existing graph and weights without mutation."""
import base64
import io

import numpy as np
from PIL import Image

LAYER_NAME = "conv_1"


class GradCAM:
    def __init__(self, model):
        import keras
        backbone = model.get_layer("mobilenetv3small_backbone")
        layer = backbone.get_layer(LAYER_NAME)
        if not isinstance(layer, keras.layers.Conv2D) or len(layer.output.shape) != 4:
            raise ValueError("Unsupported Grad-CAM target layer")
        self.extractor = keras.Model(backbone.input, [layer.output, backbone.output])
        self.augmentation = model.get_layer("training_only_augmentation")
        self.dropout = model.get_layer("head_dropout")
        self.head = model.get_layer("class_probabilities")
        if self.head.activation.__name__ != "softmax":
            raise ValueError("Expected the verified softmax classification head")

    def compute(self, pixels, class_id=None):
        import tensorflow as tf
        inputs = tf.convert_to_tensor(pixels, dtype=tf.float32)
        if tuple(inputs.shape) != (1, 160, 160, 3):
            raise ValueError("Grad-CAM requires one preprocessed RGB image")
        with tf.GradientTape() as tape:
            # Watch input so frozen upstream layers also remain differentiable.
            tape.watch(inputs)
            prepared = self.augmentation(inputs, training=False)
            activations, features = self.extractor(prepared, training=False)
            features = self.dropout(features, training=False)
            # Read the existing Dense parameters. Do not replace its activation.
            logits = tf.matmul(features, self.head.kernel) + self.head.bias
            probabilities = tf.nn.softmax(logits, axis=-1)
            target = int(tf.argmax(probabilities[0])) if class_id is None else class_id
            if type(target) is not int or not 0 <= target < 10:
                raise ValueError("Invalid explanation class")
            score = logits[0, target]
        gradients = tape.gradient(score, activations)
        if gradients is None:
            raise ValueError("Disconnected Grad-CAM gradient graph")
        if not bool(tf.reduce_all(tf.math.is_finite(gradients))):
            raise ValueError("Nonfinite Grad-CAM gradients")
        weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
        raw = tf.nn.relu(tf.reduce_sum(weights * activations, axis=-1))[0]
        if not bool(tf.reduce_all(tf.math.is_finite(raw))):
            raise ValueError("Nonfinite Grad-CAM attribution")
        maximum = float(tf.reduce_max(raw))
        heatmap = raw / maximum if maximum > 0 else tf.zeros_like(raw)
        return {"class_id": target, "probabilities": probabilities.numpy()[0],
                "heatmap": heatmap.numpy(), "gradients": gradients.numpy(),
                "status": "OK" if maximum > 0 else "NO_POSITIVE_ATTRIBUTION"}


def overlay_png(pixels, heatmap):
    """Red intensity overlay on the model-input view; zero attribution is uncolored."""
    resized = np.asarray(Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize(
        (160, 160), Image.Resampling.BILINEAR))
    alpha = np.clip(resized, 0, 1)[..., None] * .5
    rgb = np.clip(np.asarray(pixels[0], dtype=np.float32), 0, 255)
    overlay = np.rint(rgb * (1 - alpha) + np.array([255, 0, 0]) * alpha).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(overlay).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")
