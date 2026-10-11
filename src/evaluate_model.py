"""
evaluate_model.py - Model Evaluation & Reporting Module
TY BSc Data Science Project

Calculates comprehensive classification metrics (Accuracy, Precision, Recall,
F1-Score, ROC-AUC, Confusion Matrix) and generates presentation-ready
visualizations saved to reports/figures/.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless figure generation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Computes standard evaluation metrics for binary churn classification.

    Args:
        y_true: True binary labels (0 or 1)
        y_pred: Predicted binary labels (0 or 1)
        y_prob: Predicted probability for positive class (Churn)
        model_name: Name of the model

    Returns:
        Dict containing Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "classification_report": classification_report(y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True)
    }


def plot_model_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """Generates a grouped bar chart comparing all models across key metrics."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    melted = results_df.melt(
        id_vars=["model_name"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    metric_map = dict(zip(metrics, metric_labels))
    melted["Metric"] = melted["Metric"].map(metric_map)

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted")

    ax = sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="model_name",
        palette=palette
    )
    plt.title("Model Performance Comparison (All Metrics)", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12)
    plt.xlabel("Evaluation Metric", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(title="Algorithm", bbox_to_anchor=(1.02, 1), loc="upper left")

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha="center", va="bottom",
                fontsize=8, rotation=0, xytext=(0, 2),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_roc_curves(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: str
) -> None:
    """Generates overlaid ROC curves for all evaluated models."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    for name, model in models_dict.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    # Baseline diagonal
    plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=14, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_confusion_matrix_heatmap(
    cm: np.ndarray,
    model_name: str,
    save_path: str
) -> None:
    """Generates an annotated heatmap of the confusion matrix."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.set_theme(style="white")

    group_names = ["True Neg (Retained)", "False Pos (False Alarm)", "False Neg (Missed Churn)", "True Pos (Caught Churn)"]
    group_counts = [f"{val:d}" for val in cm.flatten()]
    group_percentages = [f"{val:.1%}" for val in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}\n({v3})" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted No Churn", "Predicted Churn"],
        yticklabels=["Actual No Churn", "Actual Churn"]
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Actual Customer Status", fontsize=11)
    plt.xlabel("Predicted Churn Status", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    save_path: str = ""
) -> None:
    """Extracts and plots top feature importances for tree-based models or coefficients for linear models."""
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=feat_df, x="importance", y="feature", palette="Blues_r")
    plt.title(f"Top {top_n} Predictive Features (Model Importance)", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature", fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()


def generate_eda_figures(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Generates standard Exploratory Data Analysis figures as specified in PRD/TRD:
    - churn_distribution.png
    - contract_vs_churn.png
    - monthly_charges_vs_churn.png
    - tenure_vs_churn.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Churn Distribution
    plt.figure(figsize=(6, 5))
    churn_counts = df["Churn"].value_counts()
    churn_labels = ["No Churn (Retained)", "Churn (Left)"] if 0 in churn_counts.index else ["No Churn", "Churn"]
    colors = ["#2b5c8f", "#d9534f"]
    plt.pie(
        churn_counts,
        labels=churn_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0, 0.08),
        shadow=True
    )
    plt.title("Overall Customer Churn Distribution", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "churn_distribution.png"), dpi=300)
    plt.close()

    # 2. Contract vs Churn
    if "Contract" in df.columns:
        plt.figure(figsize=(8, 5))
        # Compute churn percentage per contract
        contract_churn = df.groupby("Contract")["Churn"].value_counts(normalize=True).unstack().fillna(0)
        churn_col = 1 if 1 in contract_churn.columns else "Yes"
        if churn_col in contract_churn.columns:
            churn_rates = (contract_churn[churn_col] * 100).reset_index()
            churn_rates.columns = ["Contract", "Churn_Rate"]
            ax = sns.barplot(data=churn_rates, x="Contract", y="Churn_Rate", palette="Reds_r")
            plt.title("Churn Rate by Contract Type (%)", fontsize=13, fontweight="bold", pad=15)
            plt.ylabel("Churn Rate (%)", fontsize=11)
            plt.xlabel("Contract Type", fontsize=11)
            plt.ylim(0, 100)
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f"{height:.1f}%", (p.get_x() + p.get_width() / 2., height),
                            ha="center", va="bottom", fontsize=10, xytext=(0, 3), textcoords="offset points")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, "contract_vs_churn.png"), dpi=300)
            plt.close()

    # 3. Monthly Charges vs Churn
    if "MonthlyCharges" in df.columns:
        plt.figure(figsize=(8, 5))
        churn_mapped = df["Churn"].map({1: "Churn", 0: "No Churn"}) if df["Churn"].dtype in [int, np.int64] else df["Churn"]
        sns.kdeplot(data=df, x="MonthlyCharges", hue=churn_mapped, common_norm=False, fill=True, alpha=0.4, palette={"Churn": "#d9534f", "No Churn": "#2b5c8f"})
        plt.title("Monthly Charges Distribution by Churn Status", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Monthly Charges ($)", fontsize=11)
        plt.ylabel("Density", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "monthly_charges_vs_churn.png"), dpi=300)
        plt.close()

    # 4. Tenure vs Churn
    if "tenure" in df.columns:
        plt.figure(figsize=(8, 5))
        churn_mapped = df["Churn"].map({1: "Churn", 0: "No Churn"}) if df["Churn"].dtype in [int, np.int64] else df["Churn"]
        sns.boxplot(data=df, x=churn_mapped, y="tenure", palette={"Churn": "#d9534f", "No Churn": "#2b5c8f"})
        plt.title("Customer Tenure by Churn Status", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("Churn Status", fontsize=11)
        plt.ylabel("Tenure (Months)", fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "tenure_vs_churn.png"), dpi=300)
        plt.close()


if __name__ == "__main__":
    print("Testing evaluate_model module...")
    y_t = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    y_p = np.array([0, 1, 0, 0, 1, 0, 1, 1])
    y_pr = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.3, 0.6, 0.85])
    metrics = compute_evaluation_metrics(y_t, y_p, y_pr, "DemoModel")
    print("Computed test metrics:", metrics)
