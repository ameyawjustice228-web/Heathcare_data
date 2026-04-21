from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
los_model = joblib.load("models/los_model.pkl")
test_model = joblib.load("models/xgboost_model.joblib")
encoders = joblib.load("encorders/encoders.pkl")

def preprocess(data):
    df = pd.DataFrame([data])

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

    for col, le in encoders.items():
        if col in df:
            df[col] = le.transform(df[col])

    return df

@app.route('/')
def home():
    return "🚀 Hospital Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict_los():
    data = request.json
    df = preprocess(data)
    prediction = los_model.predict(df)
    prediction_proba = los_model.predict_proba(df)
    return jsonify({"prediction": prediction.tolist(), "prediction_probability": prediction_proba.tolist()})

@app.route('/predict_test', methods=['POST'])
def predict_test():
    data = request.json
    df = preprocess(data)
    prediction = test_model.predict(df)
    return jsonify({"Test_Result": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)