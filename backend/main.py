from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import model_loader
from predict import predict_disease

app = FastAPI(title="Plant Disease Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    print("Starting Plant Disease Detector backend...")
    model_loader.load_model()


@app.get("/")
def read_root():
    return {"message": "Server Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    result = predict_disease(image_bytes)

    response = {
        "prediction": result.get("prediction"),
        "confidence": result.get("confidence", 0.0),
    }

    if "error" in result:
        response["error"] = result["error"]

    return response
