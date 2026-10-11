"""
preprocessing.py - Data Preprocessing & Pipeline Construction
TY BSc Data Science Project

Provides reproducible loading, cleaning, validation, and scikit-learn
ColumnTransformer construction for the Customer Churn Prediction system.
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

from src.feature_engineering import engineer_features


# Feature lists
NUMERICAL_FEATURES: List[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "num_additional_services",
    "charges_per_tenure_ratio"
]

CATEGORICAL_FEATURES: List[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group"
]

TARGET_COL: str = "Churn"
ID_COL: str = "customerID"


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw customer data from CSV file.

    Args:
        file_path: Path to customer CSV dataset.

    Returns:
        pd.DataFrame: Loaded customer dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans raw customer data:
    - Removes whitespace and converts TotalCharges to numeric
    - Fills missing TotalCharges with 0.0 (typically brand-new customers with tenure=0)
    - Converts SeniorCitizen to string or category if needed
    - Maps target Churn to binary 1/0 if present
    - Removes duplicate rows
    - Applies feature engineering

    Args:
        df: Input DataFrame
        is_training: Whether dataset is used for training (contains target column)

    Returns:
        pd.DataFrame: Cleaned and feature-engineered DataFrame
    """
    df_clean = df.copy()

    # Drop duplicate records
    df_clean = df_clean.drop_duplicates()

    # Convert TotalCharges to numeric
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen to string category representation for consistent one-hot encoding
    if "SeniorCitizen" in df_clean.columns:
        df_clean["SeniorCitizen"] = df_clean["SeniorCitizen"].astype(str)

    # Convert Churn target to binary (1 = Yes, 0 = No)
    if is_training and TARGET_COL in df_clean.columns:
        # Map strings Yes/No or coerce safely
        df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(str).str.strip().map(
            {"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0}
        )
        # Drop rows where target could not be mapped
        df_clean = df_clean.dropna(subset=[TARGET_COL])
        df_clean[TARGET_COL] = df_clean[TARGET_COL].astype(int)

    # Apply Feature Engineering
    df_clean = engineer_features(df_clean)

    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """
    Builds a reusable Scikit-learn ColumnTransformer.
    - Numerical features: Median Imputation -> StandardScaler
    - Categorical features: Most Frequent Imputation -> OneHotEncoder(drop='first' / ignore)
    """
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Extracts transformed feature names from fitted ColumnTransformer.
    Used for model interpretability, feature importance, and SHAP.
    """
    feature_names = []

    # Numerical features
    feature_names.extend(NUMERICAL_FEATURES)

    # Categorical features from OneHotEncoder
    cat_transformer = preprocessor.named_transformers_["cat"]
    encoder = cat_transformer.named_steps["encoder"]
    encoded_cat_names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(list(encoded_cat_names))

    return feature_names


def prepare_training_data(
    raw_csv_path: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str], pd.DataFrame]:
    """
    Loads raw data, cleans it, splits into stratified train/test sets,
    fits the ColumnTransformer, and returns transformed arrays and metadata.

    Returns:
        X_train_trans, X_test_trans, y_train, y_test, preprocessor, feature_names, df_clean
    """
    df_raw = load_data(raw_csv_path)
    df_clean = clean_data(df_raw, is_training=True)

    # Separate features and target
    drop_cols = [TARGET_COL]
    if ID_COL in df_clean.columns:
        drop_cols.append(ID_COL)

    X = df_clean.drop(columns=drop_cols)
    y = df_clean[TARGET_COL].values

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Fit preprocessor on training data only to avoid data leakage
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    return X_train_trans, X_test_trans, y_train, y_test, preprocessor, feature_names, df_clean


def save_preprocessor(preprocessor: ColumnTransformer, save_path: str) -> None:
    """Saves fitted ColumnTransformer pipeline using joblib."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(preprocessor, save_path)


def load_preprocessor(save_path: str) -> ColumnTransformer:
    """Loads fitted ColumnTransformer pipeline using joblib."""
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Preprocessor artifact not found at {save_path}")
    return joblib.load(save_path)


if __name__ == "__main__":
    raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "customer_churn.csv")
    if os.path.exists(raw_path):
        X_tr, X_te, y_tr, y_te, prep, f_names, df_cleaned = prepare_training_data(raw_path)
        print("Training set shape:", X_tr.shape)
        print("Testing set shape:", X_te.shape)
        print("Number of transformed features:", len(f_names))
        print("Sample transformed features:", f_names[:10])

        # Save processed data
        processed_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "cleaned_customer_churn.csv")
        df_cleaned.to_csv(processed_path, index=False)
        print("Saved cleaned data to:", processed_path)
