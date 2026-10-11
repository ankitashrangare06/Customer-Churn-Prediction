"""
test_prediction.py - Unit & Integration Tests for Prediction & SQLite Logging
TY BSc Data Science Project
"""

import os
import unittest
import pandas as pd
from src.predict import ChurnPredictor, classify_risk_tier, PRESET_PROFILES
from src.database import save_prediction, get_all_predictions, get_database_summary


class TestPrediction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = ChurnPredictor()

    def test_single_prediction_high_risk(self):
        """Verify high-risk customer profile yields high probability and risk tier."""
        high_risk_input = PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"]
        result = self.predictor.predict_single(high_risk_input)

        self.assertIn(result["prediction"], [0, 1])
        self.assertIn(result["label"], ["Churn", "No Churn"])
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertEqual(result["risk_level"], "High")
        self.assertGreaterEqual(len(result["risk_drivers"]), 1)

    def test_single_prediction_low_risk(self):
        """Verify low-risk customer profile yields low probability."""
        low_risk_input = PRESET_PROFILES["Low Risk (Loyal 5-Year Subscriber, 2-Year Contract)"]
        result = self.predictor.predict_single(low_risk_input)

        self.assertIn(result["prediction"], [0, 1])
        self.assertEqual(result["risk_level"], "Low")
        self.assertLess(result["probability"], 0.35)

    def test_batch_prediction(self):
        """Verify batch prediction processes DataFrame and returns expected columns."""
        df_batch = pd.DataFrame([
            PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"],
            PRESET_PROFILES["Low Risk (Loyal 5-Year Subscriber, 2-Year Contract)"]
        ])
        results_df = self.predictor.predict_batch(df_batch)

        self.assertEqual(len(results_df), 2)
        self.assertIn("Churn_Prediction", results_df.columns)
        self.assertIn("Churn_Probability", results_df.columns)
        self.assertIn("Risk_Level", results_df.columns)

    def test_risk_tier_classification(self):
        """Verify threshold boundary assignments for Low, Medium, and High."""
        tier, _ = classify_risk_tier(0.15)
        self.assertEqual(tier, "Low")

        tier, _ = classify_risk_tier(0.50)
        self.assertEqual(tier, "Medium")

        tier, _ = classify_risk_tier(0.85)
        self.assertEqual(tier, "High")

    def test_sqlite_database_logging(self):
        """Verify saving and querying prediction logs via SQLite."""
        sample_input = PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"]
        pred_res = self.predictor.predict_single(sample_input)

        rec_id = save_prediction(
            customer_data=sample_input,
            prediction=pred_res["prediction"],
            probability=pred_res["probability"],
            risk_level=pred_res["risk_level"],
            top_factors=pred_res["risk_drivers"][:3],
            recommendation="Test retention recommendation"
        )
        self.assertIsInstance(rec_id, int)
        self.assertGreater(rec_id, 0)

        history_df = get_all_predictions()
        self.assertFalse(history_df.empty)
        summary = get_database_summary()
        self.assertGreater(summary["total_predictions"], 0)


if __name__ == "__main__":
    unittest.main()
