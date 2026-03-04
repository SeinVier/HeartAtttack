import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load artifacts
scaler = joblib.load("preprocessor.pkl")
model = joblib.load("model.pkl")

def make_prediction(features):
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame([features], columns=columns)
    
    # Scale and predict
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)
    return prediction[0]

def main():
    st.title('Heart Attack Risk Prediction Model')
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input('Age', 1, 100, 50)
        sex = st.selectbox('Sex (1=M, 0=F)', [1, 0])
        cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)
        chol = st.number_input('Cholesterol', 100, 600, 200)
        fbs = st.selectbox('Fasting Blood Sugar > 120 (1=T, 0=F)', [0, 1])

    with col2:
        restecg = st.selectbox('Resting ECG (0-2)', [0, 1, 2])
        thalach = st.number_input('Max Heart Rate', 60, 220, 150)
        exang = st.selectbox('Exercise Angina (1=Y, 0=N)', [0, 1])
        oldpeak = st.number_input('ST Depression', 0.0, 10.0, 0.0)
        slope = st.selectbox('Slope (0-2)', [0, 1, 2])
        ca = st.selectbox('Major Vessels (0-4)', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thal (0-3)', [0, 1, 2, 3])
    
    if st.button('Make Prediction'):
        features = [age, sex, cp, trestbps, chol, fbs, restecg, 
                    thalach, exang, oldpeak, slope, ca, thal]
        result = make_prediction(features)
        
        if result == 1:
            st.error('Result: High Risk of Heart Attack')
        else:
            st.success('Result: Low Risk of Heart Attack')

if __name__ == '__main__':

    main()

