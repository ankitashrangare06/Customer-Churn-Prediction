"""
predict.py - Inference & Prediction Service
TY BSc Data Science Project

Provides single-customer and batch-customer churn prediction, risk tiering
(Low, Medium, High), and pipeline orchestration.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_data
from src.explain import ChurnExplainer

# Risk thresholds
LOW_RISK_MAX = 0.35
MEDIUM_RISK_MAX = 0.65

# Predefined profiles for demonstration and testing
PRESET_PROFILES = {
    "High Risk (New Customer, Month-to-Month, High Cost)": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.85,
        "TotalCharges": 189.70
    },
    "Low Risk (Loyal 5-Year Subscriber, 2-Year Contract)": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 62,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 58.20,
        "TotalCharges": 3608.40
    },
    "Medium Risk (Mid-Tenure, 1-Year Contract, Moderate Bundle)": {
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 18,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 79.50,
        "TotalCharges": 1431.00
    }
}


def classify_risk_tier(probability: float) -> Tuple[str, str]:
    """
    Classifies churn probability into risk categories and UI badge colors.
    - Low Risk:    < 35%
    - Medium Risk: 35% - 65%
    - High Risk:   >= 65%
    """
    if probability < LOW_RISK_MAX:
        return "Low", "#28a745"     # Green
    elif probability < MEDIUM_RISK_MAX:
        return "Medium", "#ffc107"  # Yellow / Amber
    else:
        return "High", "#dc3545"    # Red


class ChurnPredictor:
    """Production predictor wrapper encapsulating model and pipeline artifacts."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        explainer: Optional[ChurnExplainer] = None
    ):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.pkl")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.explainer = explainer if explainer is not None else ChurnExplainer(model_path, preprocessor_path)

    def predict_single(self, customer_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes end-to-end prediction for an individual customer.

        Args:
            customer_input: Dictionary with customer features.

        Returns:
            Dict containing prediction label, probability, risk level, and SHAP drivers.
        """
        df_input = pd.DataFrame([customer_input])
        df_cleaned = clean_data(df_input, is_training=False)

        # Preprocess
        X_trans = self.preprocessor.transform(df_cleaned)

        # Inference
        probability = float(self.model.predict_proba(X_trans)[0, 1])
        prediction = int(probability >= 0.5)
        label = "Churn" if prediction == 1 else "No Churn"
        risk_level, badge_color = classify_risk_tier(probability)

        # Compute SHAP explanation
        explanation = self.explainer.explain_prediction(df_cleaned)

        return {
            "prediction": prediction,
            "label": label,
            "probability": round(probability, 4),
            "probability_percent": f"{probability * 100:.1f}%",
            "risk_level": risk_level,
            "badge_color": badge_color,
            "risk_drivers": explanation["risk_drivers"],
            "retention_factors": explanation["retention_factors"],
            "raw_explanation": explanation
        }

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        """
        Generates churn predictions for a batch DataFrame of customer records.

        Args:
            df_batch: Raw customer DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame augmented with prediction columns.
        """
        df_clean = clean_data(df_batch, is_training=False)
        X_trans = self.preprocessor.transform(df_clean)

        probabilities = self.model.predict_proba(X_trans)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        result_df = df_batch.copy()
        result_df["Churn_Prediction"] = predictions
        result_df["Churn_Label"] = np.where(predictions == 1, "Churn", "No Churn")
        result_df["Churn_Probability"] = np.round(probabilities, 4)
        result_df["Churn_Probability_Pct"] = [f"{p*100:.1f}%" for p in probabilities]

        risk_levels = []
        for p in probabilities:
            tier, _ = classify_risk_tier(p)
            risk_levels.append(tier)
        result_df["Risk_Level"] = risk_levels

        return result_df


if __name__ == "__main__":
    predictor = ChurnPredictor()
    print("Testing single customer prediction with High Risk profile:")
    sample = PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"]
    result = predictor.predict_single(sample)
    print(f"Prediction: {result['label']} (Prob: {result['probability_percent']}) | Risk: {result['risk_level']}")
    print("Top Risk Drivers:")
    for d in result["risk_drivers"][:3]:
        print(f"  - {d['description']} ({d['shap_value']:+.3f})")
