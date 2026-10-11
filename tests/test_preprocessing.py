"""
test_preprocessing.py - Unit Tests for Data Preprocessing & Feature Engineering
TY BSc Data Science Project
"""

import unittest
import numpy as np
import pandas as pd
from src.preprocessing import clean_data, build_preprocessor, get_feature_names
from src.feature_engineering import engineer_features, create_tenure_groups, count_additional_services


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "customerID": ["CUST-01", "CUST-02", "CUST-03"],
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 1, 0],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "tenure": [0, 12, 65],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["DSL", "Fiber optic", "No"],
            "OnlineSecurity": ["Yes", "No", "No internet service"],
            "OnlineBackup": ["No", "No", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No internet service"],
            "TechSupport": ["Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
            "MonthlyCharges": [29.85, 89.10, 20.05],
            "TotalCharges": [" ", "1069.20", "1303.25"],  # Note space for tenure=0
            "Churn": ["No", "Yes", "No"]
        })

    def test_total_charges_cleaned_numeric(self):
        """Verify that blank whitespace TotalCharges values are converted to 0.0 float."""
        cleaned = clean_data(self.sample_df, is_training=True)
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["TotalCharges"]))
        self.assertEqual(cleaned["TotalCharges"].iloc[0], 0.0)
        self.assertEqual(cleaned["TotalCharges"].iloc[1], 1069.20)

    def test_target_churn_binary_mapping(self):
        """Verify that Churn strings 'Yes'/'No' are correctly mapped to integer 1/0."""
        cleaned = clean_data(self.sample_df, is_training=True)
        self.assertListEqual(list(cleaned["Churn"]), [0, 1, 0])

    def test_feature_engineering_tenure_groups(self):
        """Verify that tenure groups are created according to defined lifecycle tiers."""
        engineered = engineer_features(self.sample_df)
        self.assertIn("tenure_group", engineered.columns)
        self.assertEqual(engineered["tenure_group"].iloc[0], "0-12m")
        self.assertEqual(engineered["tenure_group"].iloc[2], ">60m")

    def test_feature_engineering_additional_services(self):
        """Verify that subscribed add-on services are correctly summed."""
        engineered = engineer_features(self.sample_df)
        self.assertIn("num_additional_services", engineered.columns)
        # Row 0 has OnlineSecurity=Yes, TechSupport=Yes -> count 2
        self.assertEqual(engineered["num_additional_services"].iloc[0], 2)
        # Row 1 has DeviceProtection=Yes, StreamingTV=Yes, StreamingMovies=Yes -> count 3
        self.assertEqual(engineered["num_additional_services"].iloc[1], 3)
        # Row 2 has none -> count 0
        self.assertEqual(engineered["num_additional_services"].iloc[2], 0)

    def test_column_transformer_fit_transform(self):
        """Verify ColumnTransformer fits and transforms without errors."""
        cleaned = clean_data(self.sample_df, is_training=False)
        preprocessor = build_preprocessor()
        X_trans = preprocessor.fit_transform(cleaned)
        feature_names = get_feature_names(preprocessor)

        self.assertIsInstance(X_trans, np.ndarray)
        self.assertEqual(X_trans.shape[0], 3)
        self.assertEqual(X_trans.shape[1], len(feature_names))


if __name__ == "__main__":
    unittest.main()
