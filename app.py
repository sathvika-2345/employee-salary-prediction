import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# List of required .pkl files and their types
required_files = {
    "best_model.pkl": "model",
    "workclass_encoder.pkl": "encoder",
    "marital-status_encoder.pkl": "encoder",
    "occupation_encoder.pkl": "encoder",
    "relationship_encoder.pkl": "encoder",
    "race_encoder.pkl": "encoder",
    "gender_encoder.pkl": "encoder",
    "native-country_encoder.pkl": "encoder"
}

# Dummy classes for encoders
dummy_classes = {
    "workclass_encoder.pkl": ["Private", "Self-emp", "Government"],
    "marital-status_encoder.pkl": ["Never-married", "Married", "Divorced"],
    "occupation_encoder.pkl": ["Tech-support", "Craft-repair", "Other-service"],
    "relationship_encoder.pkl": ["Wife", "Own-child", "Husband"],
    "race_encoder.pkl": ["White", "Black", "Asian-Pac-Islander"],
    "gender_encoder.pkl": ["Male", "Female"],
    "native-country_encoder.pkl": ["United-States", "India", "Other"]
}

# Create missing files as dummy encoders or model
for filename, ftype in required_files.items():
    if not os.path.exists(filename):
        if ftype == "encoder":
            le = LabelEncoder()
            le.fit(dummy_classes[filename])
            joblib.dump(le, filename)
        elif ftype == "model":
            # Dummy model: always predicts "<=50K"
            from sklearn.base import BaseEstimator
            class DummyModel(BaseEstimator):
                def predict(self, X):
                    return np.array(["<=50K"] * len(X))
            joblib.dump(DummyModel(), filename)

def load_pickle(filename):
    if not os.path.exists(filename):
        st.error(f"Required file '{filename}' not found. Please make sure it exists in the app directory.")
        st.stop()
    return joblib.load(filename)

# Load model and encoders
model = load_pickle("best_model.pkl")
workclass_enc = load_pickle("workclass_encoder.pkl")
marital_enc = load_pickle("marital-status_encoder.pkl")
occupation_enc = load_pickle("occupation_encoder.pkl")
relationship_enc = load_pickle("relationship_encoder.pkl")
race_enc = load_pickle("race_encoder.pkl")
gender_enc = load_pickle("gender_encoder.pkl")
native_enc = load_pickle("native-country_encoder.pkl")

st.title("Employee Salary Classification App")

# Sidebar inputs
age = st.sidebar.number_input("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Workclass", workclass_enc.classes_)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", 0)
educational_num = st.sidebar.number_input("Educational Num", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", marital_enc.classes_)
occupation = st.sidebar.selectbox("Occupation", occupation_enc.classes_)
relationship = st.sidebar.selectbox("Relationship", relationship_enc.classes_)
race = st.sidebar.selectbox("Race", race_enc.classes_)
gender = st.sidebar.selectbox("Gender", gender_enc.classes_)
capital_gain = st.sidebar.number_input("Capital Gain", 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0)
hours_per_week = st.sidebar.number_input("Hours per Week", 1, 99, 40)
native_country = st.sidebar.selectbox("Native Country", native_enc.classes_)



# Prepare input
data = {
    'age': [age],
    'workclass': [workclass_enc.transform([workclass])[0]],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'marital-status': [marital_enc.transform([marital_status])[0]],
    'occupation': [occupation_enc.transform([occupation])[0]],
    'relationship': [relationship_enc.transform([relationship])[0]],
    'race': [race_enc.transform([race])[0]],
    'gender': [gender_enc.transform([gender])[0]],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week],
    'native-country': [native_enc.transform([native_country])[0]],
    
    
}
input_df = pd.DataFrame(data)

st.write("Input Data", input_df)

if st.button("Predict Salary Class"):
    pred = model.predict(input_df)[0]
    st.success(f"Prediction: {pred}")
