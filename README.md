# 🧠 Cognitive-Retail-Technologies

## 📄 Document Classification Model

This repository contains a deep learning model built with **PyTorch** that classifies document images into specific categories.  
It uses a pretrained **ResNet-18** architecture and a simple prediction script (`predict.py`).

---

## 🚀 Setup Instructions

### 1. Clone the Repository

git clone https://github.com/Cognitive-retail-technologies/Congnitive-repo.git
cd Congnitive-repo


## 2. Create a virtual Environment
python -m venv venv
source venv/bin/activate     # On macOS/Linux
venv\Scripts\activate        # On Windows

## 3. Install Required Packages
pip install -r requirements.txt

## 4.Train you model Setup by running
python train.py

## Run Predictions File 
python predict.py --image path/to/your/image.jpg



