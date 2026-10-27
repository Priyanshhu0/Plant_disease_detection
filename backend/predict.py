import numpy as np

from utils import load_image_from_bytes, preprocess_image
from model_loader import get_model, get_class_labels, get_image_size


def predict_disease(image_bytes):
    result = {
        "prediction": None,
        "confidence": 0.0,
    }

    # Step 1: decode the image
    pil_image = load_image_from_bytes(image_bytes)
    if pil_image is None:
        result["error"] = "Invalid image file"
        return result

    # Step 2: check the model is actually loaded
    model = get_model()
    if model is None:
        result["error"] = "Model missing. Please train and place plant_disease_model.keras in backend/model/"
        return result

    # Step 3: preprocess and predict
    try:
        image_array = preprocess_image(pil_image, get_image_size())
        predictions = model.predict(image_array, verbose=0)
    except Exception as e:
        result["error"] = "Prediction failed: " + str(e)
        return result

    predicted_index = int(np.argmax(predictions[0]))
    confidence_score = float(predictions[0][predicted_index]) * 100


    
    class_labels = get_class_labels()

    if predicted_index >= len(class_labels):
        result["error"] = (
            f"Model predicted class index {predicted_index}, "
            f"but only {len(class_labels)} labels are configured."
        )
        return result

    result["prediction"] = class_labels[predicted_index]
    result["confidence"] = round(confidence_score, 2)

    return result
