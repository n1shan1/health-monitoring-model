from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Mental Health Risk Prediction API")

# Load persisted artifacts
model = joblib.load('models/model.joblib')
scaler = joblib.load('models/scaler.joblib')
ordinal_encoder = joblib.load('models/ordinal_encoder.joblib')
feature_cols = joblib.load('models/feature_cols.joblib')

class PatientData(BaseModel):
    age: float
    income: float
    education_level: str
    sleep_pattern: str
    physical_activity: str
    marital_status: str
    smoking_status: str
    number_of_children: int

@app.post("/predict")
def predict_risk(patient: PatientData):
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Age': patient.age,
            'Income': patient.income,
            'Education Level': patient.education_level,
            'Sleep Patterns': patient.sleep_pattern,
            'Physical Activity Level': patient.physical_activity,
            'Marital Status': patient.marital_status,
            'Smoking Status': patient.smoking_status,
            'Number of Children': patient.number_of_children
        }])

        # Apply preprocessing
        ordinal_features = ['Education Level', 'Sleep Patterns', 'Physical Activity Level']
        nominal_features = ['Marital Status', 'Smoking Status']
        numerical_features = ['Age', 'Income', 'Number of Children']

        # Ordinal encoding
        input_data[ordinal_features] = ordinal_encoder.transform(input_data[ordinal_features])

        # One-hot encoding
        input_data = pd.get_dummies(input_data, columns=nominal_features, drop_first=True)

        # Ensure all dummy columns are present
        for col in feature_cols:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match training
        input_data = input_data[feature_cols]

        # Scaling
        input_data[numerical_features] = scaler.transform(input_data[numerical_features])

        # Predict
        risk_probability = model.predict_proba(input_data)[0, 1]
        depression_risk = int(model.predict(input_data)[0])

        return {
            "depression_risk": depression_risk,
            "risk_probability": float(risk_probability)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
def root():
    return {"message": "Mental Health Risk Prediction API"} 