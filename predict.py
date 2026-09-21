from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "image_detector.keras"
IMG_SIZE = 128

_model = None


def load_model():
    """Load the trained model once and reuse it for later predictions."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Train the model first or add the supplied model file."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def preprocess_image(image_path):
    """Convert an input image to the CNN's expected tensor format."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))

    pixels = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(pixels, axis=0)


def predict_image(image_path):
    """Return the predicted class and confidence percentage."""
    model = load_model()
    image_tensor = preprocess_image(image_path)

    prediction = float(model.predict(image_tensor, verbose=0)[0][0])

    # The training directory uses FAKE=0 and REAL=1.
    if prediction >= 0.5:
        result = "REAL IMAGE"
        confidence = prediction * 100
    else:
        result = "AI-GENERATED IMAGE"
        confidence = (1 - prediction) * 100

    return result, round(confidence, 2)
