import streamlit as st
import numpy as np
import joblib
import os

# Load trained model (robust path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../Model/heart_disease_model.pkl')
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Make sure it is in the 'Model' folder.")
model = joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below to predict the risk of heart disease.")

# --- User inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=55)
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholesterol (chol)", value=200)
thalach = st.number_input("Max Heart Rate Achieved (thalach)", value=150)
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
ca = st.number_input("Number of major vessels colored by fluoroscopy (0-3)", min_value=0, max_value=3, value=0)

sex = st.selectbox("Sex (0 = female, 1 = male)", [0,1])
cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
thal = st.selectbox("Thalassemia (1-3)", [1,2,3])
slope = st.selectbox("Slope of ST segment (0-2)", [0,1,2])
restecg = st.selectbox("Resting ECG (0-2)", [0,1,2])
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0,1])
exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0,1])

# --- Prepare input ---
custom_input = np.zeros(model.n_features_in_)
custom_input[0] = age
custom_input[1] = trestbps
custom_input[2] = chol
custom_input[3] = thalach
custom_input[4] = oldpeak
custom_input[5] = ca

# Adjust one-hot indices (check your model's X columns!)
if cp > 0: custom_input[6 + (cp-1)] = 1
if thal > 1: custom_input[9 + (thal-2)] = 1
if slope > 0: custom_input[11 + (slope-1)] = 1
if restecg > 0: custom_input[13 + (restecg-1)] = 1
if sex == 1: custom_input[15] = 1
if fbs == 1: custom_input[16] = 1
if exang == 1: custom_input[17] = 1

sample = custom_input.reshape(1, -1)

# --- Prediction ---
if st.button("Predict"):
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][prediction]
    st.write("**Prediction:**", "Heart Disease" if prediction==1 else "No Heart Disease")
    st.write("**Probability:**", round(probability*100,2), "%")
