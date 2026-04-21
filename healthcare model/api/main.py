from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Hospital Prediction API")

# --- Load model & preprocessor ---
los_model = joblib.load("../models/los_model.pkl")
encoders = joblib.load("../encorders/encoders.pkl")


def preprocess(data):
    df = pd.DataFrame([data])

    # Rename columns to match encoder keys
    df = df.rename(columns={
        'Blood_Type': 'Blood Type',
        'Medical_Condition': 'Medical Condition',
        'Insurance_Provider': 'Insurance Provider',
        'Admission_Type': 'Admission Type',
        'Billing_Amount': 'Billing Amount'
    })

    # Convert categorical columns to title case to match training data
    categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Admission Type', 'Medication']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].str.title()

    # Select only the features used by the model except Length of Stay
    model_features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Insurance Provider', 'Billing Amount', 'Admission Type', 'Medication']
    df = df[model_features]

    # For prediction, set Length of Stay to a default value since it's required by the model
    df['Length of Stay'] = 2  # default value

    # Encode categorical variables, handling unknown categories
    for col in df.columns:
        encoder_key = col.replace('_', ' ')
        if encoder_key in encoders:
            le = encoders[encoder_key]
            try:
                df[col] = le.transform(df[col])
            except ValueError as e:
                if 'unseen' in str(e):
                    # For unknown categories, use the first class as default
                    df[col] = le.transform([le.classes_[0]])[0]
                else:
                    raise e

    return df
class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Hospital: str
    Insurance_Provider: str
    Billing_Amount: float
    Room_Number: int
    Admission_Type: str
    Medication: str
    Test_Results: str = None


# --- Home route ---
@app.get("/")
def home():
    return {"message": "XGBoost Hospital API is running"}


from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")


# --- Prediction route ---

def predict(input_data: PredictionInput):

    try:
        # Convert input to dict
        data = input_data.dict()
        df = preprocess(data)
        prediction = los_model.predict(df)
        prediction_proba = los_model.predict_proba(df)
        
        # Decode the prediction to string
        class_names = ['Abnormal', 'Inconclusive', 'Normal']
        predicted_class = [class_names[int(p)] for p in prediction]
        
        return {"prediction": predicted_class, "prediction_probability": prediction_proba.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))