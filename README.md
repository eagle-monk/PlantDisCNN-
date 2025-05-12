🌿 Plant Disease Classifier using CNN (PlantVillage Dataset)
🔍 Project Overview
This project is a deep learning-based image classifier that identifies plant diseases from leaf images using a Convolutional Neural Network (CNN). It uses the PlantVillage dataset containing over 50,000 labeled images of healthy and diseased plant leaves.
# 🌿 Plant Disease Classifier using CNN

This project uses a Convolutional Neural Network (CNN) to classify plant leaf diseases using the **PlantVillage Dataset**. It aims to assist farmers and agronomists in early disease detection to increase crop yield and sustainability.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Future Improvements](#future-improvements)
- [Links](#links)

---

## 📖 Overview

- **Goal**: Classify healthy and diseased plant leaves from images.
- **Method**: Use CNN to learn visual patterns of disease symptoms.
- **Model Accuracy**: ~98% (training), ~86% (validation)

---

## 📂 Dataset

- **Name**: [PlantVillage Dataset](https://www.kaggle.com/datasets/spMohanty/plantvillage-dataset)
- **Source**: Kaggle
- **Size**: ~1.6 GB
- **Classes**: 38 (including various diseases and healthy leaves)
- **Images**: Over 54,000 labeled images

---

## 🛠️ Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib / Seaborn
- OpenCV
- Google Colab
- Kaggle API

---

## 🚀 Project Workflow

### 1. 📥 Data Collection

Download and unzip the dataset using the Kaggle API:

```python
from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download link(https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)


2. 🧹 Data Preprocessing
Resize images to 128x128

Normalize pixel values

Use ImageDataGenerator for augmentation

Split into training/validation sets (e.g., 80/20)

3. 🧠 Model Building (CNN)
Basic CNN architecture with:

3 Conv2D + MaxPooling layers
Dropout to avoid overfitting
Dense layers for classification (Softmax)

4. 📈 Training
python
Copy
Edit
model.fit(train_generator, validation_data=val_generator, epochs=10)

5. ✅ Evaluation
Accuracy and Loss plots
Confusion matrix
Classification report

6. 💾 Save Model
python
Copy
Edit
model.save('/content/drive/MyDrive/plant_disease_model.h5')
📊 Results
Train Accuracy: ~95%

Validation Accuracy: ~90%

Good generalization with augmentation and regularization

Capable of recognizing diseases across 15 plant species

📁 Folder Structure
bash
Copy
Edit
PlantDiseaseClassifier/
│
├── plant_disease_classifier.ipynb   # Colab Notebook
├── requirements.txt                 # Package list
├── plantvillage/                    # Dataset folder
├── model/                           # Saved models
├── results/                         # Evaluation plots & outputs
└── README.md                        # Project documentation
⚙️ Installation
Clone Repository
bash
Copy
Edit
git clone https://github.com/yourusername/PlantDiseaseClassifier.git
cd PlantDiseaseClassifier
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
🚧 Future Improvements
Use transfer learning (MobileNet, ResNet, EfficientNet)

Deploy as a web app (Streamlit / Flask)

Real-time detection from webcam

Export model to TensorFlow Lite for mobile use

🔗 Links
📦 Dataset on Kaggle

📄 Colab Notebook

📁 GitHub Repository

🧑‍💻 Author
OMM Narayan
📧 otikolia1@gmail.com
🌐 LinkedIn (https://www.linkedin.com/in/om-narayan-tikolia-962061bb/)


