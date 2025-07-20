import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder 

st.set_page_config(page_title="Disease Readmission Prediction 🚑")

st.title("Disease Readmission Prediction 🚑")
st.markdown("Predict whether a patient will be readmitted within 30 days.")

# Load model
model = joblib.load("model/readmission_model.pkl")

# Input features
def user_input():
    race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Other", "Asian", "Hispanic"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 14, 3)
    num_lab_procedures = st.slider("Lab Procedures", 0, 132, 40)
    num_procedures = st.slider("Num Procedures", 0, 6, 1)
    num_medications = st.slider("Num Medications", 1, 81, 10)
    number_outpatient = st.slider("Outpatient visits", 0, 42, 0)
    number_emergency = st.slider("Emergency visits", 0, 76, 0)
    number_inpatient = st.slider("Inpatient visits", 0, 21, 0)
    
    data = {
        'race': race,
        'gender': gender,
        'age': age,
        'time_in_hospital': time_in_hospital,
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_outpatient': number_outpatient,
        'number_emergency': number_emergency,
        'number_inpatient': number_inpatient
    }
    return pd.DataFrame(data, index=[0])

df = user_input()

# Dummy encoding (must match training process!)
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# Predict
if st.button("Predict"):
    result = model.predict(df)
    if result[0] == 1:
        st.error("⚠️ High Risk: Likely to be readmitted within 30 days.")
    else:
        st.success("✅ Low Risk: Not likely to be readmitted within 30 days.")
