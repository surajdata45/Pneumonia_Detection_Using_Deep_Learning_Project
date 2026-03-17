# 🫁 Pneumonia Detection using Deep Learning (VGG19)

## 📌 Overview
This project is a deep learning-based web application that detects **Pneumonia from Chest X-ray images** using a 
Convolutional Neural Network (CNN) with transfer learning.

---

## 🎯 Problem Statement
Manual diagnosis of pneumonia from X-rays is time-consuming and requires expert radiologists.  
This project aims to assist in **fast and accurate preliminary detection** using AI.

---

## 🧠 Solution Approach
- Used **Transfer Learning with VGG19**
- Preprocessed chest X-ray dataset
- Applied **data augmentation**
- Fine-tuned model layers for better accuracy
- Built a **Flask web app** for real-time prediction

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy / Pandas  
- Flask  
- HTML/CSS  

---

## 📊 Model Details
- Architecture: VGG19 (Pretrained)
- Input Size: 128x128
- Output: Binary Classification (Pneumonia / Normal)

---

## 🚀 Features
- Upload chest X-ray image  
- Real-time prediction  
- Simple web interface  
- Fast and user-friendly  

---

## ⚙️ How to Run Locally

```bash
# Clone repo
git clone https://github.com/your-username/pneumonia-detection.git

# Go to folder
cd pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Run app
python app.py
