from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

from preprocessing import DataPreprocessor

app = FastAPI(title="Hospital Prediction API")

# --- Load model & preprocessor ---
xgb_model_path = "xgboost_model.pkl"
preprocessor_path = "fitted_preprocessor.pkl"

xgb_model = None
preprocessor = None

if os.path.exists(xgb_model_path):
    xgb_model = joblib.load(xgb_model_path)
    print("XGBoost model loaded successfully!")
else:
    print("Model file not found!")

if os.path.exists(preprocessor_path):
    preprocessor = DataPreprocessor.load(preprocessor_path)
    print("Preprocessor loaded successfully!")
else:
    print("Preprocessor file not found!")


# --- Request schema (important in FastAPI) ---
class PredictionInput(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Date_of_Admission: str
    Hospital: str
    Insurance_Provider: str
    Billing_Amount: float
    Room_Number: int
    Admission_Type: str
    Discharge_Date: str
    Medication: str
    Test_Results: str = None  # optional for prediction mode


# --- Home route ---
@app.get("/")
def home():
    return {"message": "XGBoost Hospital API is running"}


# --- Prediction route ---
@app.post("/predict")
def predict(input_data: PredictionInput):

    if xgb_model is None or preprocessor is None:
        raise HTTPException(status_code=500, detail="Model or preprocessor not loaded")

    try:
        # Convert input to DataFrame
        data_dict = input_data.dict()
        raw_df = pd.DataFrame([data_dict])

        # Preprocess
        processed_df, _ = preprocessor.transform_data(raw_df, predict_mode=True)

        # Predict
        prediction = xgb_model.predict(processed_df)

        # Convert numeric back to label
        label = preprocessor.target_encoder.inverse_transform(prediction.astype(int))

        return {
            "prediction": label.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))