from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import json

app = Flask(__name__)
CORS(app)

# Load the available model
try:
    model = joblib.load("../models/xgboost_model.pkl")
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found, using dummy predictions")
    model = None

# Create encoders for categorical data
def create_encoders():
    """Create label encoders for categorical variables"""
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    
    # Define possible values for each categorical feature
    blood_types = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    genders = ['Female', 'Male']
    medical_conditions = ['Cancer', 'Obesity', 'Diabetes', 'Asthma', 'Hypertension', 'Arthritis', 'Heart Disease', 'Kidney Disease']
    admission_types = ['Urgent', 'Emergency', 'Elective']
    medications = ['Paracetamol', 'Ibuprofen', 'Aspirin', 'Penicillin', 'Lipitor', 'Antibiotics', 'Painkillers']
    insurance_providers = ['Blue Cross', 'Medicare', 'Aetna', 'UnitedHealthcare', 'Cigna']
    
    # Create and fit encoders
    encoders['Blood Type'] = LabelEncoder().fit(blood_types)
    encoders['Gender'] = LabelEncoder().fit(genders)
    encoders['Medical Condition'] = LabelEncoder().fit(medical_conditions)
    encoders['Admission Type'] = LabelEncoder().fit(admission_types)
    encoders['Medication'] = LabelEncoder().fit(medications)
    encoders['Insurance Provider'] = LabelEncoder().fit(insurance_providers)
    
    return encoders

# Initialize encoders
encoders = create_encoders()

def preprocess_data(data):
    """Preprocess raw input data for model prediction"""
    df = pd.DataFrame([data])
    
    # Encode categorical variables
    categorical_cols = ['Blood Type', 'Gender', 'Medical Condition', 'Admission Type', 'Medication', 'Insurance Provider']
    
    for col in categorical_cols:
        if col in df.columns and col in encoders:
            try:
                df[col] = encoders[col].transform(df[col])
            except ValueError as e:
                # Handle unseen categories by assigning a default value
                df[col] = 0
    
    # Ensure all required columns are present and in correct order
    required_columns = [
        'Age', 'Gender', 'Blood Type', 'Medical Condition', 'Billing Amount',
        'Room Number', 'Admission Type', 'Medication', 'Length of Stay',
        'Hospital', 'Insurance Provider'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[required_columns]
    
    return df

@app.route('/')
def home():
    return jsonify({
        "message": "🚀 Healthcare Prediction API Running",
        "version": "1.0",
        "endpoints": {
            "/": "Home",
            "/predict": "Make predictions (POST)",
            "/health": "Health check"
        }
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        if isinstance(data, list):
            data = data[0]  # Take first item if it's a list
        
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        if model is not None:
            # Make prediction
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data) if hasattr(model, 'predict_proba') else None
            
            result = {
                "prediction": prediction.tolist(),
                "input_data": data,
                "processed_features": processed_data.iloc[0].to_dict()
            }
            
            if prediction_proba is not None:
                result["prediction_probability"] = prediction_proba.tolist()
                
        else:
            # Dummy prediction for testing
            result = {
                "prediction": [1],  # Dummy prediction
                "input_data": data,
                "note": "Using dummy prediction - model not loaded"
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Prediction failed"
        }), 400

if __name__ == "__main__":
    print("Starting Healthcare Prediction API...")
    print("Available endpoints:")
    print("  GET  /         - Home")
    print("  GET  /health   - Health check")
    print("  POST /predict  - Make predictions")
    print("\nAPI running on http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)