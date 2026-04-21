
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
        self.features_to_drop_initial = ['Name', 'Doctor']
        self.date_cols = ['Date of Admission', 'Discharge Date']
        self.fitted_feature_columns = None # To ensure consistent feature order for model input

    def fit_transform_data(self, df):
        # Create a copy to avoid modifying the original DataFrame passed in
        df_processed = df.copy()

        # Drop initial columns (e.g., 'Name', 'Doctor')
        df_processed.drop(columns=self.features_to_drop_initial, inplace=True, errors='ignore')

        # Date conversion
        for col in self.date_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col])

        # Feature Engineering: Length of Stay
        if all(col in df_processed.columns for col in self.date_cols):
            df_processed['Length of Stay'] = (df_processed['Discharge Date'] - df_processed['Date of Admission']).dt.days
        df_processed.drop(columns=self.date_cols, inplace=True, errors='ignore')

        # Encode categorical variables
        for col in self.categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col])
                self.label_encoders[col] = le
            else:
                print(f"Warning: Categorical column '{col}' not found in DataFrame for fitting.")

        # Separate and encode target variable if present
        y_encoded = None
        if 'Test Results' in df_processed.columns:
            y_col = df_processed['Test Results']
            self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y_col)
            df_processed = df_processed.drop(columns=['Test Results']) # Drop original target column from features
        else:
            print("Warning: Target column 'Test Results' not found in DataFrame for fitting.")

        # Scale numerical features
        numerical_features_present = [col for col in self.numerical_cols if col in df_processed.columns]
        if numerical_features_present:
            self.scaler = StandardScaler()
            df_processed[numerical_features_present] = self.scaler.fit_transform(df_processed[numerical_features_present])
        else:
            print("Warning: No numerical features found in DataFrame for scaling.")

        # Store the order of features after fitting, crucial for consistent input to the model
        self.fitted_feature_columns = df_processed.columns.tolist()

        X_processed = df_processed
        return X_processed, y_encoded

    def transform_data(self, df, predict_mode=False):
        # Create a copy to avoid modifying the original DataFrame passed in
        df_transformed = df.copy()

        # Drop initial columns
        df_transformed.drop(columns=self.features_to_drop_initial, inplace=True, errors='ignore')

        # Date conversion
        for col in self.date_cols:
            if col in df_transformed.columns:
                df_transformed[col] = pd.to_datetime(df_transformed[col])

        # Feature Engineering: Length of Stay
        if all(col in df_transformed.columns for col in self.date_cols):
            df_transformed['Length of Stay'] = (df_transformed['Discharge Date'] - df_transformed['Date of Admission']).dt.days
        df_transformed.drop(columns=self.date_cols, inplace=True, errors='ignore')

        # Transform categorical variables
        for col in self.categorical_cols:
            if col in df_transformed.columns and col in self.label_encoders:
                # Handle unknown categories: LabelEncoder's transform will raise ValueError
                try:
                    df_transformed[col] = self.label_encoders[col].transform(df_transformed[col])
                except ValueError as e:
                    print(f"Error: Unknown category encountered in column '{col}'. Error: {e}")
                    # For a robust API, you might map unknown categories to a specific value or raise a custom error.
                    # For now, we let the ValueError propagate.
                    raise
            elif col in df_transformed.columns:
                 print(f"Warning: LabelEncoder for categorical column '{col}' not found. Column not transformed.")

        # Transform numerical features
        numerical_features_present = [col for col in self.numerical_cols if col in df_transformed.columns]
        if numerical_features_present and self.scaler:
            df_transformed[numerical_features_present] = self.scaler.transform(df_transformed[numerical_features_present])
        elif numerical_features_present:
            print("Warning: StandardScaler not fitted. Numerical features not scaled during transform.")

        y_encoded = None
        if not predict_mode and 'Test Results' in df_transformed.columns:
            y_col = df_transformed['Test Results']
            if self.target_encoder:
                y_encoded = self.target_encoder.transform(y_col)
            else:
                print("Warning: Target encoder not fitted. Target variable not transformed.")
            df_transformed.drop(columns=['Test Results'], inplace=True, errors='ignore')

        # Ensure feature columns are in the same order as during training
        if self.fitted_feature_columns is not None:
            # Check for missing features
            missing_in_input = [f for f in self.fitted_feature_columns if f not in df_transformed.columns]
            if missing_in_input:
                raise ValueError(f"Input data is missing expected features: {missing_in_input}. Please provide all features.")

            # Check for extra features
            extra_in_input = [f for f in df_transformed.columns if f not in self.fitted_feature_columns]
            if extra_in_input:
                print(f"Warning: Input data contains extra features not seen during training: {extra_in_input}. These will be dropped.")
                df_transformed = df_transformed.drop(columns=extra_in_input)

            # Reorder columns to match training order
            df_transformed = df_transformed[self.fitted_feature_columns]
        else:
            print("Warning: Preprocessor has not been fitted, cannot ensure consistent feature order.")

        X_transformed = df_transformed
        return X_transformed, y_encoded

    def save(self, file_path='preprocessor.pkl'):
        """Saves the fitted preprocessor object."""
        joblib.dump(self, file_path)
        print(f"Preprocessor saved to {file_path}")

    @classmethod
    def load(cls, file_path='preprocessor.pkl'):
        """Loads a fitted preprocessor object."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Preprocessor file not found at {file_path}")
        preprocessor = joblib.load(file_path)
        print(f"Preprocessor loaded from {file_path}")
        return preprocessor

if __name__ == '__main__':
    # This block demonstrates how to use the preprocessor
    print("--- Running example usage of DataPreprocessor ---")

    # Example: Create a dummy DataFrame (similar to your initial df)
    # This assumes a raw DataFrame like the one you loaded, before any preprocessing
    raw_data = {
        'Name': ['John Doe', 'Jane Smith'],
        'Age': [30, 25],
        'Gender': ['Male', 'Female'],
        'Blood Type': ['A+', 'B-'],
        'Medical Condition': ['Diabetes', 'Flu'],
        'Date of Admission': ['2023-01-01', '2023-01-05'],
        'Doctor': ['Dr. Smith', 'Dr. Jones'],
        'Hospital': ['Hospital A', 'Hospital B'],
        'Insurance Provider': ['Blue Cross', 'Aetna'],
        'Billing Amount': [1000.50, 2000.75],
        'Room Number': [101, 202],
        'Admission Type': ['Emergency', 'Elective'],
        'Discharge Date': ['2023-01-03', '2023-01-10'],
        'Medication': ['Insulin', 'Antibiotics'],
        'Test Results': ['Normal', 'Abnormal']
    }
    raw_df_example = pd.DataFrame(raw_data)

    print("
Original raw DataFrame:")
    print(raw_df_example)

    # Initialize and fit the preprocessor
    preprocessor = DataPreprocessor()
    X_train_processed, y_train_encoded = preprocessor.fit_transform_data(raw_df_example)

    print("
Processed X (features):")
    print(X_train_processed)
    print("
Encoded y (target):")
    print(y_train_encoded)
    print(f"
Fitted feature columns: {preprocessor.fitted_feature_columns}")


    # Save the preprocessor
    preprocessor.save('fitted_preprocessor.pkl')

    # Load the preprocessor
    loaded_preprocessor = DataPreprocessor.load('fitted_preprocessor.pkl')

    # Example: Transform new raw data using the loaded preprocessor
    new_raw_data = {
        'Name': ['Alice Johnson'],
        'Age': [40],
        'Gender': ['Female'],
        'Blood Type': ['O-'],
        'Medical Condition': ['Cancer'],
        'Date of Admission': ['2023-02-10'],
        'Doctor': ['Dr. White'],
        'Hospital': ['Hospital A'],
        'Insurance Provider': ['Medicare'],
        'Billing Amount': [3500.00],
        'Room Number': [303],
        'Admission Type': ['Urgent'],
        'Discharge Date': ['2023-02-15'],
        'Medication': ['Chemotherapy']
        # 'Test Results': ['Inconclusive'] # Omit target for prediction mode
    }
    new_raw_df_example = pd.DataFrame(new_raw_data)
    print("
New raw DataFrame for prediction:")
    print(new_raw_df_example)

    X_new_processed, _ = loaded_preprocessor.transform_data(new_raw_df_example, predict_mode=True)
    print("
Transformed new X for prediction:")
    print(X_new_processed)

    print("
--- End example usage ---")
