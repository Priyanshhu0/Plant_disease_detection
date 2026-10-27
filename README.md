# 🌿 Plant Disease Detection System Using Deep Learning

An AI-powered web application that identifies plant leaf diseases from images using **Convolutional Neural Networks (CNNs)** and deep learning. 

## 📌 Overview

The **Plant Disease Detection System** uses deep learning to analyze images of plant leaves and predict the corresponding disease category learned during model training.

## 🧠 How It Works

1. **Image Upload:** The user selects an image of a plant leaf.
2. **Image Preprocessing:** The image is resized to the dimensions expected by the trained model and converted into the required numerical format.
3. **Feature Extraction:** CNN layers learn visual patterns such as edges, textures, spots, and discoloration.
4. **Classification:** The trained model predicts a class from its configured output categories.
5. **Result Display:** The application presents the predicted class to the user.

## 📂 Project Structure

```text
Plant_disease_detection/
│
├── backend/
│   └── Backend API and prediction logic
├── frontend/
│   └── HTML, CSS and JavaScript interface
├── ml/
│   ├── train_model.py
│   ├── confusion_matrix.png
│   └── loss_graph.png
└─── dataset/
    └── Plant disease image dataset
```

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Priyanshhu0/Plant_disease_detection.git
cd Plant_disease_detection
```

### 2. Install Dependencies

```bash
pip install tensorflow fastapi uvicorn python-multipart pillow numpy matplotlib scikit-learn
```


### 3. Prepare the Dataset

Download the [PlantVillage dataset from Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease) if you need to train the model from scratch.

Organize the extracted images according to the directory structure expected by `ml/train_model.py`.

### 4. Train the Model

Run the training script from the repository root:

```bash
python ml/train_model.py
```

### 5. Start the Backend

```bash
uvicorn backend.main:app --reload
```
---

**Made with Python, TensorFlow, and a passion for applying AI to agriculture.** 🌿
