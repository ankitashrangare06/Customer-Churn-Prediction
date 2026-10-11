"""
feature_engineering.py - Feature Engineering Module
TY BSc Data Science Project

Derives domain-specific features from raw customer attributes to enhance
machine learning model discriminatory power for churn prediction.
"""

import pandas as pd
import numpy as np
from typing import List


ADDITIONAL_SERVICES: List[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def create_tenure_groups(df: pd.DataFrame) -> pd.Series:
    """
    Groups customer tenure into business lifecycle buckets:
    - 0-12m : New / onboarding customers (highest churn vulnerability)
    - 13-24m: Early-stage customers
    - 25-48m: Mid-stage customers
    - 49-60m: Established customers
    - >60m  : Long-term loyal customers
    """
    bins = [-1, 12, 24, 48, 60, 1000]
    labels = ["0-12m", "13-24m", "25-48m", "49-60m", ">60m"]
    return pd.cut(df["tenure"], bins=bins, labels=labels).astype(str)


def count_additional_services(df: pd.DataFrame) -> pd.Series:
    """
    Counts how many optional value-add internet services a customer subscribes to.
    Customers with fewer bundled services typically show higher churn velocity.
    """
    count = pd.Series(0, index=df.index, dtype=int)
    for col in ADDITIONAL_SERVICES:
        if col in df.columns:
            count += (df[col] == "Yes").astype(int)
    return count


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering logic to the DataFrame.
    Preserves original columns while appending derived features.

    Args:
        df: Input pandas DataFrame containing customer records.

    Returns:
        pd.DataFrame: Augmented DataFrame with newly engineered features.
    """
    df_feat = df.copy()

    # 1. Tenure Group
    if "tenure" in df_feat.columns:
        df_feat["tenure_group"] = create_tenure_groups(df_feat)

    # 2. Number of Additional Services
    df_feat["num_additional_services"] = count_additional_services(df_feat)

    # 3. Internet Service Flag
    if "InternetService" in df_feat.columns:
        df_feat["has_internet_service"] = (df_feat["InternetService"] != "No").astype(int)

    # 4. Month-to-month Contract Flag
    if "Contract" in df_feat.columns:
        df_feat["is_month_to_month"] = (df_feat["Contract"] == "Month-to-month").astype(int)

    # 5. Charges Per Tenure Ratio
    if "MonthlyCharges" in df_feat.columns and "tenure" in df_feat.columns:
        df_feat["charges_per_tenure_ratio"] = df_feat["MonthlyCharges"] / (df_feat["tenure"] + 1)

    return df_feat


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "tenure": [1, 25, 65],
        "MonthlyCharges": [70.5, 80.0, 105.0],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "Yes", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"]
    })
    augmented = engineer_features(test_df)
    print("Engineered DataFrame shape:", augmented.shape)
    print(augmented[["tenure", "tenure_group", "num_additional_services", "charges_per_tenure_ratio"]])
