"""
explain.py - Explainable AI (XAI) using SHAP
TY BSc Data Science Project

Provides model interpretability and feature-level explanations using SHAP
(SHapley Additive exPlanations) for individual predictions and global insights.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import get_feature_names, clean_data

# Feature descriptions mapping for non-technical users
FEATURE_INTERPRETATION_MAP = {
    "Contract_Month-to-month": "Month-to-month contract (lack of commitment)",
    "Contract_One year": "One-year fixed contract commitment",
    "Contract_Two year": "Two-year long-term contract loyalty",
    "tenure": "Account tenure length (customer relationship age)",
    "MonthlyCharges": "Monthly recurring billing charges",
    "TotalCharges": "Cumulative lifetime spend",
    "num_additional_services": "Total bundled value-added services",
    "charges_per_tenure_ratio": "Monthly pricing burden relative to tenure",
    "TechSupport_No": "No technical support subscribed",
    "TechSupport_Yes": "Active tech support protection",
    "OnlineSecurity_No": "Lack of online security add-on",
    "OnlineSecurity_Yes": "Active online security protection",
    "InternetService_Fiber optic": "High-speed Fiber optic connection",
    "InternetService_DSL": "DSL internet service",
    "InternetService_No": "No internet service subscribed",
    "PaymentMethod_Electronic check": "Electronic check payment method",
    "PaymentMethod_Credit card (automatic)": "Automated credit card billing",
    "PaymentMethod_Bank transfer (automatic)": "Automated bank transfer billing",
    "PaperlessBilling_Yes": "Paperless electronic billing",
    "OnlineBackup_No": "No cloud backup protection",
    "DeviceProtection_No": "No device hardware warranty/protection",
    "SeniorCitizen_1": "Senior citizen subscriber demographic",
    "Partner_No": "Single account (no partner on plan)",
    "Dependents_No": "No dependents attached to household plan"
}


class ChurnExplainer:
    """Wrapper class managing SHAP explainers for churn classification models."""

    def __init__(self, model_path: Optional[str] = None, preprocessor_path: Optional[str] = None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, "models", "churn_model.pkl")
        if preprocessor_path is None:
            preprocessor_path = os.path.join(PROJECT_ROOT, "models", "preprocessing_pipeline.pkl")

        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        self.feature_names = get_feature_names(self.preprocessor)

        # Initialize TreeExplainer for tree models
        try:
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            self.explainer = shap.Explainer(self.model)

    def extract_churn_shap_values(self, X_trans: np.ndarray) -> np.ndarray:
        """
        Calculates raw SHAP values and normalizes them for the positive class (Churn).
        Handles both 2D and 3D SHAP outputs across scikit-learn and xgboost.
        """
        raw_shap = self.explainer.shap_values(X_trans)

        if isinstance(raw_shap, list):
            # Binary classification list [shap_class_0, shap_class_1]
            return np.array(raw_shap[1][0])
        elif isinstance(raw_shap, np.ndarray):
            if raw_shap.ndim == 3:
                # Shape: (1, num_features, 2)
                return raw_shap[0, :, 1]
            elif raw_shap.ndim == 2:
                # Shape: (1, num_features)
                return raw_shap[0, :]
            else:
                return raw_shap.flatten()
        else:
            return np.array(raw_shap)

    def explain_prediction(
        self,
        customer_df: pd.DataFrame,
        top_n: int = 6
    ) -> Dict[str, Any]:
        """
        Generates individual prediction explanation with human-friendly interpretations.

        Args:
            customer_df: Cleaned single-row DataFrame containing customer attributes.
            top_n: Number of primary drivers to return.

        Returns:
            Dict containing top risk-increasing factors, top retention factors,
            and data formatted for charting.
        """
        # Transform customer attributes using preprocessor
        X_trans = self.preprocessor.transform(customer_df)
        shap_values = self.extract_churn_shap_values(X_trans)

        # Pair each feature with its SHAP contribution
        contributions = []
        for feat_name, shap_val in zip(self.feature_names, shap_values):
            clean_desc = FEATURE_INTERPRETATION_MAP.get(feat_name, feat_name.replace("_", " "))
            contributions.append({
                "feature": feat_name,
                "description": clean_desc,
                "shap_value": float(shap_val),
                "abs_impact": abs(float(shap_val)),
                "direction": "Risk Driver" if shap_val > 0 else "Retention Factor"
            })

        contributions_df = pd.DataFrame(contributions)

        # Top factors increasing churn risk (positive SHAP)
        risk_drivers = contributions_df[contributions_df["shap_value"] > 0].sort_values(
            by="shap_value", ascending=False
        ).head(top_n).to_dict(orient="records")

        # Top factors mitigating churn risk (negative SHAP)
        retention_factors = contributions_df[contributions_df["shap_value"] < 0].sort_values(
            by="shap_value", ascending=True
        ).head(top_n).to_dict(orient="records")

        # Top overall influential features for waterfall/bar plot
        top_overall = contributions_df.sort_values(
            by="abs_impact", ascending=False
        ).head(top_n * 2).to_dict(orient="records")

        return {
            "risk_drivers": risk_drivers,
            "retention_factors": retention_factors,
            "top_overall": top_overall,
            "base_value": float(getattr(self.explainer, "expected_value", [0.0, 0.5])[1])
            if isinstance(getattr(self.explainer, "expected_value", None), (list, np.ndarray))
            else float(getattr(self.explainer, "expected_value", 0.5))
        }


def create_shap_waterfall_figure(explanation: Dict[str, Any], max_display: int = 8) -> go.Figure:
    """
    Builds an interactive Plotly horizontal bar chart showing SHAP feature impacts
    for a single customer prediction.
    """
    top_factors = explanation["top_overall"][:max_display]
    top_factors = list(reversed(top_factors))  # Bottom to top for horizontal bar chart

    descriptions = [f["description"] for f in top_factors]
    shap_vals = [f["shap_value"] for f in top_factors]
    colors = ["#d9534f" if v > 0 else "#28a745" for v in shap_vals]

    fig = go.Figure(go.Bar(
        x=shap_vals,
        y=descriptions,
        orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.3f}" for v in shap_vals],
        textposition="outside"
    ))

    fig.update_layout(
        title="<b>SHAP Feature Contribution to Churn Prediction</b><br><sup>Red: Increases Churn Risk | Green: Encourages Customer Retention</sup>",
        xaxis_title="SHAP Value (Impact on Prediction Log-Odds / Risk)",
        yaxis_title="Customer Feature",
        template="plotly_white",
        margin=dict(l=10, r=20, t=60, b=40),
        height=380
    )
    return fig


if __name__ == "__main__":
    from src.preprocessing import clean_data

    explainer = ChurnExplainer()
    print("ChurnExplainer initialized with", len(explainer.feature_names), "features.")

    sample_cust = pd.DataFrame([{
        "customerID": "TEST-01",
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
        "MonthlyCharges": 92.5,
        "TotalCharges": 185.0
    }])
    sample_clean = clean_data(sample_cust, is_training=False)
    exp = explainer.explain_prediction(sample_clean)
    print("\nTop Risk Drivers:")
    for d in exp["risk_drivers"]:
        print(f" - {d['description']}: {d['shap_value']:+.4f}")
    print("\nTop Retention Factors:")
    for f in exp["retention_factors"]:
        print(f" - {f['description']}: {f['shap_value']:+.4f}")
