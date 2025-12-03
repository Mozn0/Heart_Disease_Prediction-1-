# Heart_Disease_Prediction-1-
Heart Disease Prediction : This repository contains all files, models, data, and code used to build a Machine Learning system that predicts heart disease. The project follows a clean and professional structure commonly used in Data Science and Machine Learning.

Heart Disease Prediction Using Machine Learning :

_ Executive Summary

This project aims to build a predictive machine learning model that can determine whether a patient is likely to have heart disease based on clinical and demographic features.
Several models were trained and evaluated, and the final chosen model was deployed in a Streamlit web application for easy user interaction.

_ Problem Statement

Heart disease remains one of the leading causes of mortality worldwide. Early prediction can save lives by enabling preventive care before critical health events occur.

The objective of this project is to:

Build a reliable prediction model using patient data

Evaluate multiple machine learning algorithms

Deploy the best-performing model in an interactive web app

_ Dataset Source

The dataset used in this project is the Heart Disease UCI dataset, publicly available on Kaggle/UCI Repository.
It contains key patient features such as:

Age

Gender

Cholesterol levels

Resting blood pressure

Chest pain type

Maximum heart rate

Blood sugar

ECG details

Exercise-induced angina

Several other clinical indicators

_ Methodology

The project follows the standard machine learning workflow:

1. Data Collection & Import

Dataset imported from CSV into pandas.

2. Data Cleaning & Preprocessing

Handling missing values

Encoding categorical variables

Feature scaling

Train-test split (80/20)

3. Exploratory Data Analysis (EDA)

Understanding patterns, distributions, correlations, and class imbalances.

4. Model Training

The following models were tested:

Logistic Regression

Random Forest

Decision Tree

K-Nearest Neighbors

Support Vector Machine

5. Model Evaluation

Evaluation metrics:

Accuracy

Precision

Recall

F1-score

Classification report

6. Model Saving

The best model was saved using:

joblib.dump(model, "heart_disease_model.pkl")

7. Deployment

A Streamlit web application was created to allow users to input medical data and receive a heart disease prediction instantly.

_ Results
Random Forest Classifier (Final Selected Model)

Accuracy: 0.90
Classification Report:

Precision

Class 0: 0.94

Class 1: 0.87

Recall

Class 0: 0.88

Class 1: 0.93

F1-Score

Class 0: 0.91

Class 1: 0.90

- Selected as the final model and saved as heart_disease_model.pkl.

Logistic Regression (Previous Best Model)

Accuracy: 0.61
- Not selected due to low accuracy/performance.

- Demonstration of the Application

The Streamlit app provides:

1-User-friendly form inputs for patient data

2- Real-time prediction using the trained model

3- Probability output for better interpretation

The app runs using:

streamlit run app.py
