import numpy as np
from PIL import Image
from io import BytesIO


def load_image_from_bytes(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return image
    except Exception as e:
        print("Failed to open image:", str(e))
        return None


def preprocess_image(pil_image, target_size):
    resized = pil_image.resize(target_size)
    array = np.array(resized, dtype=np.float32)
    array = array / 255.0

    array = np.expand_dims(array, axis=0)
    return array
