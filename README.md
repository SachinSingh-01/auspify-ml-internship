# Netflix Content Type Prediction (Task 2)

## Overview
A machine learning classification project built to predict whether a Netflix catalog title is a **Movie** or a **TV Show** based on its production country, maturity rating, and release year.

## Dataset & Preprocessing
- **Dataset Size:** 8,790 titles (69.69% Movies, 30.31% TV Shows)
- **Train/Test Split:** 80% Training (7,032 rows) / 20% Testing (1,758 rows) using stratified sampling
- **Features Used:** One-hot encoded `country` and `rating` combined with numerical `release_year` (94 total features)

## Model Comparison Results
| Model | Test Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 78.04% | 0.70 | 0.76 |
| Decision Tree | 77.76% | 0.72 | 0.77 |
| Random Forest (Default) | 78.33% | 0.73 | 0.78 |
| **Random Forest (Custom Class Weights)** | **77.36%** | **0.74** | **0.78** |

**Final Model Selection:** A Random Forest Classifier with custom class weights (`{"Movie": 1, "TV Show": 1.5}`) was selected because it achieved the highest 5-fold cross-validated Macro F1 score (**0.7321**) and improved minority-class (`TV Show`) recall from 57% to 65%.

## Project Files
- `task2_content_type_prediction.ipynb` — Exploratory data analysis, preprocessing, model evaluation, and artifact serialization
- `netflix_inference_artifacts.pkl` — Bundled trained Random Forest model, fitted `OneHotEncoder`, and 94-column feature schema
- `predict.py` — Standalone script that runs batch inference on `validation_samples.csv`
- `app.py` — Interactive Streamlit web application for real-time predictions

## How to Run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt