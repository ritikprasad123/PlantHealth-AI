# 🌱 PlantHealth AI

AI-powered plant disease screening web application built with deep learning.

## 📌 Project Overview

PlantHealth AI analyzes a plant leaf image and predicts one of 38 supported plant disease/health classes.

**Workflow:**

`Leaf Image → Image Check → MobileNetV2 → Prediction → Confidence → Top 3 Candidates → Suggested Action`

> **Note:** This is an educational and preliminary screening system. AI predictions should not replace professional agricultural diagnosis.

## 🎯 Objective

The project aims to provide a simple image-based plant health screening workflow for identifying common plant diseases from leaf images.

## 🌿 Supported Plants

The current model supports 14 plant types:

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

**Total classes: 38**

## 📊 Dataset

The project uses the **PlantVillage dataset**.

- Total images: **54,305**
- Classes: **38**
- Model input size: **224 × 224**
- Training images: **37,997**
- Validation images: **8,129**
- Test images: **8,179**

The train, validation and test sets are separated by class, with all 38 classes represented in each split.

## 🧠 Final Model

**Fine-tuned MobileNetV2**

- Architecture: MobileNetV2
- Transfer learning with ImageNet pretrained weights
- Input: `224 × 224 × 3`
- Output: 38 classes
- Optimizer: Adam
- Fine-tuning with a low learning rate

### Final Evaluation

- **Test Accuracy: 87.43%**
- **Test Loss: 0.3938**
- **Test Images: 8,179**

The 87.43% result is the reproducible benchmark obtained on the fixed 8,179-image test split.

## 🔬 Evaluation

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Some visually similar disease classes remain challenging, including Apple Scab and Corn Gray Leaf Spot.

## 💻 Streamlit App

The application provides:

- Leaf image upload
- Image preview
- Resolution and format information
- Basic image quality checking
- AI prediction
- Prediction score
- Detected plant
- Top 3 predictions
- Suggested action
- Low-confidence handling
- Model information
- Supported plant information

## 📁 Project Structure

```text
PlantHealth-AI/
│
├── app.py
├── requirements.txt
│
├── models/
│   └── final_plant_disease_model.keras
│
├── dataset/
│   ├── color/
│   └── split/
│       ├── train/
│       ├── val/
│       └── test/
│
└── notebooks/
    ├── 01DataPreparation.ipynb
    ├── 02DataSplit.ipynb
    ├── 03CustomCNN.ipynb
    ├── 04TransferLearning.ipynb
    └── test.ipynb
```

## ⚙️ Installation

Open PowerShell in the project folder and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scriptsctivate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## ▶️ Run the Application

From the project root:

```powershell
streamlit run app.py
```

## 🧪 Testing

The application has been tested with:

- Healthy leaf images
- Diseased leaf images
- PlantVillage test images
- Real-world/random leaf images
- Low-confidence predictions

A low prediction score does not automatically mean low image quality. Real-world photos can differ from the PlantVillage training images.

## ⚠️ Limitations

Performance may change for images with different:

- Backgrounds
- Lighting
- Camera quality
- Leaf angles
- Disease visibility

The model only supports the 38 classes used during training. Images outside those classes may produce uncertain predictions.

## 🚀 Future Improvements

- Add more diverse real-world leaf images
- Add more plant/disease classes
- Improve visually similar disease recognition
- Calibrate prediction scores
- Convert the model to TensorFlow Lite
- Deploy as a mobile application

## 🛠️ Technology Stack

- Python
- TensorFlow
- Keras
- MobileNetV2
- Streamlit
- OpenCV
- Pillow
- Scikit-learn
- Jupyter Notebook

## 👨‍💻 Project

**PlantHealth AI — AI-Based Plant Disease Detection and Health Analysis**
