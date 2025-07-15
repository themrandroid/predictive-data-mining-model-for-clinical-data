import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('diabetes_model.pkl')

# App title
st.title("🩺 Diabetes Prediction App")
st.write("Enter your health data below to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure (Diastolic)", min_value=0, max_value=150, value=70)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5, format="%.3f")
age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    prediction = model.predict(input_data)     

    result = "🚨 Diabetic" if prediction[0] == 1 else "✅ Not Diabetic"
    st.subheader("Prediction Result:")
    st.success(result)