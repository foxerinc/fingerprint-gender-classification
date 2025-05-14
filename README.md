# Fingerprint Gender Classification using LBP and Ridge Density Features with SVM

This project is part of my undergraduate thesis titled:
**"Penggabungan Fitur Local Binary Pattern (LBP) dan Ridge Density dalam Klasifikasi Gender Berdasarkan Sidik Jari Menggunakan Support Vector Machine"**. 

The dataset used in this study is sourced from:  
👉 [SOCOFing - Fingerprint Gender, Hand, and Finger Type Dataset (Kaggle)](https://www.kaggle.com/datasets/ruizgara/socofing)



## Features
- Preprocessing: For LBP feature extraction: Grayscale conversion, resizing, noise reduction, and contrast adjustment using histogram equalization. For Ridge Density feature extraction: Grayscale conversion, resizing, binarization, thinning, and skeletonization to simplify ridge structures.
- Feature Extraction: LBP, Ridge Density, Hand Location
- Model Used: Support Vector Machine (SVM) with 10-fold cross-validation.
- Evaluation: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Features Extracted
- LBP histograms from segmented fingerprint blocks
- Ridge density estimation in defined fingerprint regions

## 🛠 Tools & Libraries
- Python 3.9
- OpenCV
- NumPy
- scikit-learn
- matplotlib

## 📁 Folder Structure
- `data/`: Fingerprint images and annotations
- `src/`: Feature extraction and model training code
- `report/`: PDF thesis report and presentation slides

- ## 💡 Results
Achieved **accuracy of 86.50%** on fingerprint dataset using combined LBP + Ridge Density + Hand Location with SVM.

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/fingerprint-gender-classification.git
2. Install dependencies: pip install -r requirements.txt
3. use the scripts in src/ to train/test the model.
