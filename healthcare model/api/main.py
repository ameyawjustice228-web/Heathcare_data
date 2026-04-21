from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ----------------------------
# CORS SETUP (IMPORTANT)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production use specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = joblib.load("models/xgboost_model.pkl")


# ----------------------------
# INPUT SCHEMA
# ----------------------------
class PatientData(BaseModel):
    Age: int
    Gender: str
    Blood_Type: str
    Medical_Condition: str
    Billing_Amount: float
    Admission_Type: str
    Insurance_Provider: str
    Medication: str


# ----------------------------
# ROOT
# ----------------------------
@app.get("/")
def home():
    return {"message": "API running"}

# ----------------------------
# PREDICT ENDPOINT
# ----------------------------
@app.post("/predict")
def predict(data: PatientData):

    df = pd.DataFrame([data.dict()])

    # (IMPORTANT) You must preprocess here if needed
    prediction = model.predict(df)[0]

    return {
        "predicted_test_result": str(prediction)
    }
