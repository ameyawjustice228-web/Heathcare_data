
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.target_encoder = None
        self.categorical_cols = ['Gender', 'Blood Type', 'Medical Condition', 'Hospital', 'Insurance Provider', 'Admission Type', 'Medication']
        self.numerical_cols = ['Age', 'Billing Amount', 'Room Number', 'Length of Stay']

    def fit(self, df):
        # Fit LabelEncoders for categorical columns
        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le

        # Fit StandardScaler for numerical columns
        self.scaler = StandardScaler()
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        # Fit LabelEncoder for the target variable
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df['Test Results'])

    def transform(self, df):
        # Transform categorical columns using fitted LabelEncoders
        for col in self.categorical_cols:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col])
            else:
                raise ValueError(f"LabelEncoder for column '{col}' not found. Make sure to fit the preprocessor first.")

        # Transform numerical columns using fitted StandardScaler
        if self.scaler:
            df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])
        else:
            raise ValueError("StandardScaler not fitted. Make sure to fit the preprocessor first.")

        return df

    def fit_transform(self, df):
        self.fit(df.copy())
        return self.transform(df.copy())

    def inverse_transform_target(self, predictions):
        if self.target_encoder:
            return self.target_encoder.inverse_transform(predictions)
        else:
            raise ValueError("Target encoder not fitted. Cannot inverse transform.")

    def save(self, filepath):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        return joblib.load(filepath)

def load_and_clean_data(path):
    df = pd.read_csv(path)

    # Drop irrelevant columns
    df = df.drop(columns=[
        "Name", "Doctor", "Hospital",
        "Room Number"
    ])

    # Convert dates
    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"])
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"])

    # Create new feature
    df["Length of Stay"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

    df = df.drop(columns=["Date of Admission", "Discharge Date"])

    # Handle missing values
    df = df.dropna()

    return df

def encode_data(df):
    encoders = {}
    categorical_cols = df.select_dtypes(include='object').columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders
