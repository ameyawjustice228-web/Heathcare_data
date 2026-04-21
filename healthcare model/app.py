from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(title="Healthcare Prediction API", version="1.0")

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    model = joblib.load("../models/xgboost_model.pkl")
    print("Model loaded successfully")
except:
    model = None
    print("Model NOT found - running in demo mode")

# -----------------------------
# CATEGORICAL ENCODERS
# -----------------------------
def create_encoders():
    encoders = {}

    encoders["Blood Type"] = LabelEncoder().fit(
        ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    )
    encoders["Gender"] = LabelEncoder().fit(['Female', 'Male'])
    encoders["Medical Condition"] = LabelEncoder().fit(
        ['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension',
         'Arthritis', 'Heart Disease', 'Kidney Disease']
    )
    encoders["Admission Type"] = LabelEncoder().fit(
        ['Urgent', 'Emergency', 'Elective']
    )
    encoders["Medication"] = LabelEncoder().fit(
        ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin', 'Lipitor']
    )
    encoders["Insurance Provider"] = LabelEncoder().fit(
        ['Blue Cross', 'Medicare', 'Aetna', 'UnitedHealthcare', 'Cigna']
    )

    return encoders

encoders = create_encoders()

# -----------------------------
# INPUT SCHEMA (VERY IMPORTANT IN FASTAPI)
# -----------------------------
class PatientData(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: str
    Insurance_Provider: str
    Medication: str

# -----------------------------
# PREPROCESSING FUNCTION
# -----------------------------
def preprocess(data: dict):

    df = pd.DataFrame([data])

    # map API names → model names
    df = df.rename(columns={
        "Blood_Type": "Blood Type",
        "Medical_Condition": "Medical Condition",
        "Admission_Type": "Admission Type",
        "Insurance_Provider": "Insurance Provider"
    })

    categorical_cols = encoders.keys()

    for col in categorical_cols:
        if col in df.columns:
            try:
                df[col] = encoders[col].transform(df[col])
            except:
                df[col] = 0  # handle unseen values

    # Ensure required columns
    required = [
        "Age",
        "Gender",
        "Blood Type",
        "Medical Condition",
        "Billing_Amount",
        "Admission Type",
        "Insurance Provider",
        "Medication"
    ]

    for col in required:
        if col not in df.columns:
            df[col] = 0

    return df[required]

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "🚀 Healthcare ML API Running (FastAPI)",
        "status": "active"
    }

@app.get("/health")
def health():
    return {
        "model_loaded": model is not None
    }

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: PatientData):

    input_data = data.dict()

    processed = preprocess(input_data)

    if model:
        prediction = model.predict(processed)[0]

        return {
            "predicted_test_result": str(prediction)
        }

    else:
        return {
            "predicted_test_result": "Abnormal (demo mode)",
            "note": "Model not loaded"
        }
