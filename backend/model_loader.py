import os
import tensorflow as tf

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "plant_disease_model.keras")


CLASS_LABELS = [
    "Pepper - Bacterial Spot",
    "Pepper - Healthy",
    "Potato - Early Blight",
    "Potato - Late Blight",
    "Potato - Healthy",
    "Tomato - Bacterial Spot"
]


IMAGE_SIZE = (128, 128) 

trained_model = None


def load_model():
    global trained_model

    if not os.path.exists(MODEL_PATH):
        print("WARNING: Model file not found at:", MODEL_PATH)
        print("Server will run, but /predict will return an error until you train a model.")
        trained_model = None
        return

    try:
        trained_model = tf.keras.models.load_model(MODEL_PATH)
        print("Plant disease model loaded successfully.")
    except Exception as e:
        print("ERROR while loading model:", str(e))
        trained_model = None


def get_model():
    return trained_model


def get_class_labels():
    return CLASS_LABELS


def get_image_size():
    return IMAGE_SIZE
