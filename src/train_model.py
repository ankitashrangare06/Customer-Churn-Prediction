"""
train_model.py - Model Training & Selection Pipeline
TY BSc Data Science Project

Trains, cross-validates, evaluates and compares four tabular machine learning
algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost) on the
churn dataset, selecting and persisting the top-performing model.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_training_data, save_preprocessor
from src.evaluate_model import (
    compute_evaluation_metrics,
    plot_model_comparison,
    plot_roc_curves,
    plot_confusion_matrix_heatmap,
    plot_feature_importance,
    generate_eda_figures
)

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "customer_churn.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "cleaned_customer_churn.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def build_candidate_models(y_train: np.ndarray, random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiates the 4 required classifiers with hyperparameter configurations
    addressing churn class imbalance.
    """
    # Calculate scale_pos_weight for XGBoost to balance negative to positive ratio
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = float(neg_count / max(pos_count, 1))

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=random_state
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state
        )
    }
    return models


def train_and_evaluate_all_models(
    raw_data_path: str = RAW_DATA_PATH,
    random_state: int = 42
) -> Tuple[Dict[str, Any], str, pd.DataFrame]:
    """
    Executes the complete end-to-end training, cross-validation, and evaluation cycle:
    1. Preprocesses dataset and splits into train/test sets.
    2. Generates initial EDA plots.
    3. Trains and cross-validates 4 candidate models.
    4. Evaluates test set performance (Accuracy, Precision, Recall, F1, ROC-AUC).
    5. Selects best model based on composite F1 & ROC-AUC score.
    6. Generates evaluation charts and saves serialized artifacts.

    Returns:
        models_dict, best_model_name, results_df
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Data Preparation
    print("[1/6] Loading and preprocessing dataset...")
    X_train_trans, X_test_trans, y_train, y_test, preprocessor, feature_names, df_clean = prepare_training_data(
        raw_data_path,
        test_size=0.2,
        random_state=random_state
    )

    # Save cleaned processed CSV
    df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"      Cleaned dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"      Training samples: {X_train_trans.shape[0]}, Features: {X_train_trans.shape[1]}")
    print(f"      Test samples:     {X_test_trans.shape[0]}")

    # 2. EDA Visualizations
    print("[2/6] Generating EDA visualizations for reports/figures...")
    generate_eda_figures(df_clean, FIGURES_DIR)
    print("      Saved EDA figures: churn_distribution, contract_vs_churn, monthly_charges_vs_churn, tenure_vs_churn")

    # 3. Model Definition & Cross-Validation
    print("[3/6] Initializing models and computing 5-fold Stratified Cross-Validation...")
    candidate_models = build_candidate_models(y_train, random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results_list = []
    trained_models = {}

    for name, model in candidate_models.items():
        print(f"      -> Cross-validating: {name}...")
        cv_roc = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_f1 = cross_val_score(model, X_train_trans, y_train, cv=cv, scoring="f1", n_jobs=-1)

        # Train on full training split
        model.fit(X_train_trans, y_train)
        trained_models[name] = model

        # Evaluate on holdout test set
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        metrics = compute_evaluation_metrics(y_test, y_pred, y_prob, model_name=name)
        metrics["cv_roc_auc_mean"] = round(float(np.mean(cv_roc)), 4)
        metrics["cv_roc_auc_std"] = round(float(np.std(cv_roc)), 4)
        metrics["cv_f1_mean"] = round(float(np.mean(cv_f1)), 4)
        metrics["cv_f1_std"] = round(float(np.std(cv_f1)), 4)

        results_list.append(metrics)
        print(f"         Test Acc: {metrics['accuracy']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    # 4. Results Aggregation & Model Selection
    print("[4/6] Aggregating results and selecting the best model...")
    results_df = pd.DataFrame(results_list)
    results_csv_path = os.path.join(REPORTS_DIR, "model_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"      Saved metrics summary to: {results_csv_path}")

    # Best model selection strategy (PRD Sec 10 & TRD Sec 7):
    # Customer retention prioritizes identifying real churners (Recall & F1) while maintaining strong separation (ROC-AUC)
    # Composite score = 0.5 * ROC-AUC + 0.5 * F1-Score
    results_df["selection_score"] = 0.5 * results_df["roc_auc"] + 0.5 * results_df["f1_score"]
    best_row = results_df.sort_values(by="selection_score", ascending=False).iloc[0]
    best_model_name = best_row["model_name"]
    best_model = trained_models[best_model_name]

    print(f"\n>>> BEST SELECTED MODEL: {best_model_name} <<<")
    print(f"    Accuracy:  {best_row['accuracy']:.4f}")
    print(f"    Recall:    {best_row['recall']:.4f} (High recall captures at-risk customers)")
    print(f"    Precision: {best_row['precision']:.4f}")
    print(f"    F1-Score:  {best_row['f1_score']:.4f}")
    print(f"    ROC-AUC:   {best_row['roc_auc']:.4f}\n")

    # 5. Visual Reporting
    print("[5/6] Generating evaluation visual reports...")
    plot_model_comparison(results_df, os.path.join(FIGURES_DIR, "model_comparison.png"))
    plot_roc_curves(trained_models, X_test_trans, y_test, os.path.join(FIGURES_DIR, "roc_curves.png"))

    best_cm = np.array(best_row["confusion_matrix"])
    plot_confusion_matrix_heatmap(best_cm, best_model_name, os.path.join(FIGURES_DIR, "confusion_matrix_best.png"))
    plot_feature_importance(best_model, feature_names, top_n=15, save_path=os.path.join(FIGURES_DIR, "feature_importance.png"))
    print("      Saved evaluation figures: model_comparison, roc_curves, confusion_matrix_best, feature_importance")

    # 6. Artifact Serialization
    print("[6/6] Saving model artifacts to models/ directory...")
    # Best model
    best_model_path = os.path.join(MODELS_DIR, "churn_model.pkl")
    joblib.dump(best_model, best_model_path)

    # Preprocessing pipeline
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")
    save_preprocessor(preprocessor, preprocessor_path)

    # All trained models
    all_models_path = os.path.join(MODELS_DIR, "all_models.pkl")
    joblib.dump(trained_models, all_models_path)

    # Metadata
    metadata = {
        "best_model_name": best_model_name,
        "training_timestamp": datetime.now().isoformat(),
        "random_state": random_state,
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": {row["model_name"]: {
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1_score": row["f1_score"],
            "roc_auc": row["roc_auc"],
            "tn": row["tn"],
            "fp": row["fp"],
            "fn": row["fn"],
            "tp": row["tp"]
        } for _, row in results_df.iterrows()},
        "risk_thresholds": {
            "low_risk_max": 0.35,
            "medium_risk_max": 0.65
        }
    }
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"      Saved best model to:         {best_model_path}")
    print(f"      Saved preprocessor to:       {preprocessor_path}")
    print(f"      Saved all models to:         {all_models_path}")
    print(f"      Saved metadata to:           {metadata_path}")
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return trained_models, best_model_name, results_df


if __name__ == "__main__":
    train_and_evaluate_all_models()
