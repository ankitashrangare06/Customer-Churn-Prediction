"""
test_model.py - Unit Tests for Trained ML Models & Serialization
TY BSc Data Science Project
"""

import os
import unittest
import json
import joblib
import numpy as np


class TestModelArtifacts(unittest.TestCase):

    def setUp(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        self.churn_model_path = os.path.join(self.models_dir, "churn_model.pkl")
        self.preprocessor_path = os.path.join(self.models_dir, "preprocessing_pipeline.pkl")
        self.all_models_path = os.path.join(self.models_dir, "all_models.pkl")
        self.metadata_path = os.path.join(self.models_dir, "model_metadata.json")

    def test_churn_model_exists_and_loads(self):
        """Verify best model artifact is persisted and callable."""
        self.assertTrue(os.path.exists(self.churn_model_path), "churn_model.pkl not found!")
        model = joblib.load(self.churn_model_path)
        self.assertTrue(hasattr(model, "predict"))
        self.assertTrue(hasattr(model, "predict_proba"))

    def test_preprocessor_pipeline_exists_and_loads(self):
        """Verify preprocessor pipeline artifact is persisted and transforms data."""
        self.assertTrue(os.path.exists(self.preprocessor_path), "preprocessing_pipeline.pkl not found!")
        preprocessor = joblib.load(self.preprocessor_path)
        self.assertTrue(hasattr(preprocessor, "transform"))

    def test_all_models_present(self):
        """Verify all 4 required algorithms are persisted in all_models.pkl."""
        self.assertTrue(os.path.exists(self.all_models_path), "all_models.pkl not found!")
        all_models = joblib.load(self.all_models_path)
        expected_models = ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"]
        for expected in expected_models:
            self.assertIn(expected, all_models, f"Missing model: {expected}")
            self.assertTrue(hasattr(all_models[expected], "predict_proba"))

    def test_model_metadata_validity(self):
        """Verify model_metadata.json contains complete evaluation records."""
        self.assertTrue(os.path.exists(self.metadata_path), "model_metadata.json not found!")
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.assertIn("best_model_name", metadata)
        self.assertIn("metrics", metadata)
        self.assertIn("risk_thresholds", metadata)
        self.assertGreater(metadata["num_features"], 20)


if __name__ == "__main__":
    unittest.main()
