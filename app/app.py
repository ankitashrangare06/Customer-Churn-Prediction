"""
app.py - Interactive Streamlit Application for Customer Churn Prediction
TY BSc Data Science Final-Year Project

Features:
1. Overview & System Workflow
2. Exploratory Data Analysis & Interactive Visualizations
3. Machine Learning Model Comparison & Evaluation
4. Single Customer Prediction with SHAP Explainability & AI Retention Insights
5. Batch Prediction (CSV Upload & Export)
6. Historical Prediction Audit Log (SQLite Database)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_data, load_data
from src.predict import ChurnPredictor, PRESET_PROFILES, classify_risk_tier
from src.explain import ChurnExplainer, create_shap_waterfall_figure
from src.database import (
    init_db,
    save_prediction,
    get_all_predictions,
    get_database_summary,
    delete_prediction
)
from llm.recommendation import get_retention_recommendations, check_ollama_status

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
    }
    .recommendation-box {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .driver-box {
        background-color: #fff1f2;
        border-left: 5px solid #f43f5e;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# Cached Resource Loaders
@st.cache_resource
def load_churn_predictor():
    return ChurnPredictor()


@st.cache_resource
def load_all_models():
    all_models_path = os.path.join(PROJECT_ROOT, "models", "all_models.pkl")
    if os.path.exists(all_models_path):
        return joblib.load(all_models_path)
    return {}


@st.cache_data
def load_dataset():
    data_path = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_customer_churn.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn.csv")
    return pd.read_csv(data_path)


@st.cache_data
def load_model_results():
    results_path = os.path.join(PROJECT_ROOT, "reports", "model_results.csv")
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    return pd.DataFrame()


@st.cache_data
def load_metadata():
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# Initialize database
init_db()

# Load artifacts
predictor = load_churn_predictor()
all_models = load_all_models()
df_data = load_dataset()
results_df = load_model_results()
metadata = load_metadata()

# Sidebar Navigation
st.sidebar.title("🎯 ChurnGuard AI")
st.sidebar.markdown("**TY BSc Data Science Project**")
st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Overview & Problem Statement",
        "📊 Exploratory Data Analysis",
        "🤖 Model Comparison & Evaluation",
        "🔮 Customer Churn Predictor & SHAP",
        "📁 Batch CSV Prediction",
        "🗄️ Prediction Audit Log (Database)"
    ],
    index=3  # Default to Predictor
)

st.sidebar.markdown("---")
st.sidebar.markdown("### System Status")
st.sidebar.success(f"Active Model: **{metadata.get('best_model_name', 'Random Forest')}**")
ollama_online = check_ollama_status()
if ollama_online:
    st.sidebar.info("Local LLM: **Ollama Online (Qwen2.5)**")
else:
    st.sidebar.warning("Local LLM: **Expert Rules Engine (Offline Fallback)**")

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Customer Churn Prediction System | College Project")


# ==============================================================================
# 1. OVERVIEW & PROBLEM STATEMENT
# ==============================================================================
if menu == "🏠 Overview & Problem Statement":
    st.markdown('<div class="main-header">Customer Churn Prediction & Retention System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">BSc Data Science Final-Year College Project | Machine Learning & Explainable AI</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        ### 📌 Problem Statement
        In competitive subscription businesses (such as telecommunications and SaaS), acquiring new customers is **5 to 25 times more costly** than retaining existing ones.
        Companies frequently discover customer dissatisfaction only **after** the customer has already terminated their service.

        This system applies supervised Machine Learning and Explainable AI (SHAP) to:
        1. **Proactively predict customer attrition** before contract termination.
        2. **Quantify customer defection risk** into actionable tiers (**Low, Medium, High**).
        3. **Explain the root causes** driving churn for each customer.
        4. **Generate targeted retention recommendations** to preserve customer relationships and revenue.
        """)

        st.markdown("""
        ### 🎯 Project Objectives
        - **Data Pipeline:** Reproducible data ingestion, missing-value imputation, outlier management, and feature engineering.
        - **Algorithm Benchmarking:** Compare **Logistic Regression, Decision Tree, Random Forest, and XGBoost** using Stratified 5-Fold Cross-Validation.
        - **Imbalance Handling:** Optimize for **Recall and F1-Score** so at-risk customers are not missed by naive accuracy.
        - **Explainability:** Utilize **SHAP (SHapley Additive exPlanations)** to uncover individual and global feature attributions.
        - **Local Deployment:** 100% free and open-source stack (Python, Streamlit, Scikit-learn, XGBoost, SHAP, SQLite).
        """)

    with col2:
        st.markdown("### 🏗️ Architecture & Workflow")
        st.markdown("""
        ```
        [ Customer Dataset ]
                ↓
        [ Data Cleaning & Preprocessing ]
                ↓
        [ Domain Feature Engineering ]
                ↓
        [ Train/Test Split (Stratified) ]
                ↓
        [ 4 ML Models Cross-Validation ]
                ↓
        [ Best Model Selection (F1 & ROC-AUC) ]
                ↓
        [ Prediction Engine + Risk Tiers ]
                ↓
        [ SHAP Feature Attributions ]
                ↓
        [ Retention Advisor (LLM / Rules) ]
                ↓
        [ SQLite History Audit Log ]
        ```
        """)

    st.markdown("---")
    st.subheader("🛠️ Technology Stack")
    tcol1, tcol2, tcol3, tcol4 = st.columns(4)
    with tcol1:
        st.markdown("**Data & ML**\n- Python 3\n- Pandas & NumPy\n- Scikit-learn\n- XGBoost")
    with tcol2:
        st.markdown("**Explainable AI**\n- SHAP (TreeExplainer)\n- Feature Attributions\n- Force/Waterfall Visuals")
    with tcol3:
        st.markdown("**UI & Storage**\n- Streamlit Framework\n- Plotly Interactive Charts\n- SQLite Local Database")
    with tcol4:
        st.markdown("**Retention AI**\n- Qwen2.5-3B-Instruct (Ollama)\n- Expert Rule Engine Fallback\n- Zero-Cost Architecture")


# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif menu == "📊 Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive data exploration and customer churn behavioral patterns</div>', unsafe_allow_html=True)

    # Top KPI cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_cust = len(df_data)
    churn_count = (df_data["Churn"] == 1).sum() if "Churn" in df_data.columns else (df_data["Churn"] == "Yes").sum()
    churn_rate = (churn_count / total_cust) * 100
    avg_tenure = df_data["tenure"].mean()
    avg_charges = df_data["MonthlyCharges"].mean()

    kpi1.metric("Total Customers", f"{total_cust:,}")
    kpi2.metric("Overall Churn Rate", f"{churn_rate:.1f}%", f"{churn_count:,} churned", delta_color="inverse")
    kpi3.metric("Average Tenure", f"{avg_tenure:.1f} months")
    kpi4.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Churn Distribution & Contracts", "💰 Financials & Tenure", "📋 Dataset Preview"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Target Distribution (Class Balance)")
            churn_labels = df_data["Churn"].map({1: "Churned (Left)", 0: "Retained (Active)"}) if df_data["Churn"].dtype in [int, np.int64] else df_data["Churn"]
            fig_pie = px.pie(
                names=churn_labels,
                title="Customer Churn Ratio",
                color=churn_labels,
                color_discrete_map={"Churned (Left)": "#ef4444", "Retained (Active)": "#3b82f6", "Yes": "#ef4444", "No": "#3b82f6"},
                hole=0.4
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            st.subheader("Churn Rate by Contract Type")
            contract_churn = df_data.groupby("Contract")["Churn"].apply(
                lambda s: ((s == 1).sum() / len(s) * 100) if s.dtype in [int, np.int64] else ((s == "Yes").sum() / len(s) * 100)
            ).reset_index()
            contract_churn.columns = ["Contract", "Churn_Percentage"]

            fig_bar = px.bar(
                contract_churn,
                x="Contract",
                y="Churn_Percentage",
                text="Churn_Percentage",
                color="Contract",
                title="Churn Rate % across Contract Types",
                color_discrete_sequence=["#f87171", "#60a5fa", "#34d399"]
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(yaxis=dict(range=[0, 60]), yaxis_title="Churn Rate (%)")
            st.plotly_chart(fig_bar, use_container_width=True)

        st.info("💡 **Key Finding:** Month-to-month contracts have a **42.7%** churn rate, while customers on Two-year contracts churn at only **2.8%**.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Monthly Charges Distribution by Churn Status")
            churn_str = df_data["Churn"].map({1: "Churn", 0: "No Churn"}) if df_data["Churn"].dtype in [int, np.int64] else df_data["Churn"]
            fig_charges = px.histogram(
                df_data,
                x="MonthlyCharges",
                color=churn_str,
                barmode="overlay",
                marginal="box",
                title="Monthly Charges Histogram & Boxplot",
                color_discrete_map={"Churn": "#ef4444", "No Churn": "#3b82f6"}
            )
            st.plotly_chart(fig_charges, use_container_width=True)

        with c2:
            st.subheader("Tenure Boxplot by Churn Status")
            fig_tenure = px.box(
                df_data,
                x=churn_str,
                y="tenure",
                color=churn_str,
                title="Customer Tenure (Months) vs Churn",
                color_discrete_map={"Churn": "#ef4444", "No Churn": "#3b82f6"}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

    with tab3:
        st.subheader("Cleaned Dataset Records")
        st.dataframe(df_data.head(50), use_container_width=True)
        st.caption(f"Showing sample records from {len(df_data):,} total customers.")


# ==============================================================================
# 3. MODEL COMPARISON & EVALUATION
# ==============================================================================
elif menu == "🤖 Model Comparison & Evaluation":
    st.markdown('<div class="main-header">Machine Learning Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Benchmarking Logistic Regression, Decision Tree, Random Forest, and XGBoost</div>', unsafe_allow_html=True)

    if not results_df.empty:
        # Display comparison table
        st.subheader("📋 Comprehensive Evaluation Metrics Table")

        display_df = results_df[[
            "model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "cv_roc_auc_mean", "cv_f1_mean"
        ]].copy()
        display_df.columns = [
            "Algorithm", "Accuracy", "Precision", "Recall (Sensitivity)",
            "F1-Score", "ROC-AUC", "5-Fold CV ROC-AUC", "5-Fold CV F1"
        ]

        # Highlight best model
        best_name = metadata.get("best_model_name", "Random Forest")
        st.dataframe(
            display_df.style.highlight_max(
                subset=["Accuracy", "Recall (Sensitivity)", "F1-Score", "ROC-AUC"],
                color="#bbf7d0"
            ),
            use_container_width=True
        )

        st.success(f"🏆 **Selected Production Model:** **{best_name}** | Optimized for high Recall and F1-Score to prioritize identifying customers at risk of churn.")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Metric Comparison Bar Chart")
            melted_metrics = results_df.melt(
                id_vars=["model_name"],
                value_vars=["accuracy", "recall", "f1_score", "roc_auc"],
                var_name="Metric",
                value_name="Score"
            )
            fig_comp = px.bar(
                melted_metrics,
                x="Metric",
                y="Score",
                color="model_name",
                barmode="group",
                title="Model Comparison across Key Metrics",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_comp.update_layout(yaxis=dict(range=[0, 1.05]))
            st.plotly_chart(fig_comp, use_container_width=True)

        with c2:
            st.subheader("🎯 Confusion Matrix (Test Set)")
            selected_model_for_cm = st.selectbox("Select Model for Confusion Matrix:", list(results_df["model_name"]))
            row = results_df[results_df["model_name"] == selected_model_for_cm].iloc[0]
            cm_vals = np.array(json.loads(row["confusion_matrix"])) if isinstance(row["confusion_matrix"], str) else np.array(row["confusion_matrix"])

            cm_fig = px.imshow(
                cm_vals,
                text_auto=True,
                labels=dict(x="Predicted Label", y="Actual Label", color="Count"),
                x=["Retained (0)", "Churn (1)"],
                y=["Retained (0)", "Churn (1)"],
                color_continuous_scale="Blues",
                title=f"Confusion Matrix: {selected_model_for_cm}"
            )
            st.plotly_chart(cm_fig, use_container_width=True)

            tn, fp, fn, tp = int(row["tn"]), int(row["fp"]), int(row["fn"]), int(row["tp"])
            st.markdown(f"""
            - **True Negatives (Retained correctly identified):** `{tn}`
            - **False Positives (False alarms):** `{fp}`
            - **False Negatives (Missed churners):** `{fn}`
            - **True Positives (Caught churners):** `{tp}`
            """)

        st.markdown("---")
        st.subheader("🌟 Global Feature Importance (Best Model)")
        fig_path = os.path.join(PROJECT_ROOT, "reports", "figures", "feature_importance.png")
        if os.path.exists(fig_path):
            st.image(fig_path, caption="Top 15 Predictive Features by Model Importance", use_container_width=True)
    else:
        st.warning("Model results have not been generated yet. Run `python src/train_model.py` to populate metrics.")


# ==============================================================================
# 4. CUSTOMER CHURN PREDICTOR & SHAP
# ==============================================================================
elif menu == "🔮 Customer Churn Predictor & SHAP":
    st.markdown('<div class="main-header">Individual Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter customer information to predict churn probability, risk level, and SHAP drivers</div>', unsafe_allow_html=True)

    # Preset profile loader
    st.markdown("##### ⚡ Quick Load Preset Customer Profile (Viva / Demo)")
    selected_preset = st.selectbox(
        "Choose a test profile or enter custom values below:",
        ["Custom Customer Input"] + list(PRESET_PROFILES.keys())
    )

    if selected_preset in PRESET_PROFILES:
        current_data = PRESET_PROFILES[selected_preset]
    else:
        current_data = PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"]

    st.markdown("---")

    # Input Form
    with st.form("customer_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Account & Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"], index=0 if current_data.get("gender") == "Male" else 1)
            senior = st.selectbox("Senior Citizen", [0, 1], index=int(current_data.get("SeniorCitizen", 0)), format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
            partner = st.selectbox("Has Partner", ["Yes", "No"], index=0 if current_data.get("Partner") == "Yes" else 1)
            dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=0 if current_data.get("Dependents") == "Yes" else 1)
            tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=int(current_data.get("tenure", 12)))

        with col2:
            st.markdown("#### 📑 Contract & Billing")
            contract = st.selectbox(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(current_data.get("Contract", "Month-to-month"))
            )
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if current_data.get("PaperlessBilling") == "Yes" else 1)
            payment_methods = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            payment_idx = payment_methods.index(current_data.get("PaymentMethod", "Electronic check")) if current_data.get("PaymentMethod") in payment_methods else 0
            payment = st.selectbox("Payment Method", payment_methods, index=payment_idx)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=float(current_data.get("MonthlyCharges", 70.0)), step=1.0)
            auto_total = float(round(tenure * monthly_charges, 2))
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(current_data.get("TotalCharges", auto_total)), step=10.0)

        with col3:
            st.markdown("#### 🌐 Subscribed Services")
            phone = st.selectbox("Phone Service", ["Yes", "No"], index=0 if current_data.get("PhoneService") == "Yes" else 1)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(current_data.get("MultipleLines", "No")))
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(current_data.get("InternetService", "Fiber optic")))

            # Service sub-features
            sec_idx = ["No", "Yes", "No internet service"].index(current_data.get("OnlineSecurity", "No")) if current_data.get("OnlineSecurity") in ["No", "Yes", "No internet service"] else 0
            online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=sec_idx)

            backup_idx = ["No", "Yes", "No internet service"].index(current_data.get("OnlineBackup", "No")) if current_data.get("OnlineBackup") in ["No", "Yes", "No internet service"] else 0
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=backup_idx)

            dev_idx = ["No", "Yes", "No internet service"].index(current_data.get("DeviceProtection", "No")) if current_data.get("DeviceProtection") in ["No", "Yes", "No internet service"] else 0
            device_prot = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=dev_idx)

            tech_idx = ["No", "Yes", "No internet service"].index(current_data.get("TechSupport", "No")) if current_data.get("TechSupport") in ["No", "Yes", "No internet service"] else 0
            tech_supp = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=tech_idx)

            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0 if current_data.get("StreamingTV") == "No" else 1)
            streaming_mov = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0 if current_data.get("StreamingMovies") == "No" else 1)

        submit_btn = st.form_submit_button("🔮 Predict Customer Churn Risk", use_container_width=True)

    if submit_btn:
        customer_dict = {
            "customerID": f"CUST-{datetime.now().strftime('%M%S')}",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_sec,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_supp,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        # Run Prediction
        pred_res = predictor.predict_single(customer_dict)

        st.markdown("---")
        st.subheader("🎯 Prediction & Risk Assessment")

        r1, r2, r3 = st.columns([1.2, 1.2, 1.6])

        with r1:
            st.markdown("##### Churn Classification")
            if pred_res["prediction"] == 1:
                st.error("🚨 **High Likelihood of Churn**")
                st.caption("Customer is at imminent risk of terminating service.")
            else:
                st.success("✅ **Likely to Stay (Retained)**")
                st.caption("Customer displays positive retention indicators.")

        with r2:
            st.markdown("##### Churn Probability & Risk Tier")
            risk_tier = pred_res["risk_level"]
            prob_pct = pred_res["probability"] * 100

            if risk_tier == "High":
                st.markdown(f'<span class="badge-high">Risk Level: HIGH ({prob_pct:.1f}%)</span>', unsafe_allow_html=True)
            elif risk_tier == "Medium":
                st.markdown(f'<span class="badge-medium">Risk Level: MEDIUM ({prob_pct:.1f}%)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="badge-low">Risk Level: LOW ({prob_pct:.1f}%)</span>', unsafe_allow_html=True)

            st.progress(pred_res["probability"])

        with r3:
            st.markdown("##### Risk Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e3a8a"},
                    'steps': [
                        {'range': [0, 35], 'color': "#dcfce7"},
                        {'range': [35, 65], 'color': "#fef3c7"},
                        {'range': [65, 100], 'color': "#fee2e2"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': prob_pct
                    }
                }
            ))
            fig_gauge.update_layout(height=180, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("---")

        # Explainable AI (SHAP)
        st.subheader("🔍 Explainable AI (SHAP Feature Attributions)")
        st.markdown("Understand **WHY** the model arrived at this churn prediction:")

        e_col1, e_col2 = st.columns([1, 1])

        with e_col1:
            st.markdown("##### 🚨 Primary Factors Driving Churn Risk (Positive SHAP)")
            if pred_res["risk_drivers"]:
                for driver in pred_res["risk_drivers"][:4]:
                    st.markdown(f"""
                    <div class="driver-box">
                        <b>{driver['description']}</b><br>
                        <small>Impact Score: +{driver['shap_value']:.3f} | Strongly pushes model toward Churn</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No dominant risk drivers found for this customer.")

        with e_col2:
            st.markdown("##### 🛡️ Primary Factors Supporting Retention (Negative SHAP)")
            if pred_res["retention_factors"]:
                for factor in pred_res["retention_factors"][:4]:
                    st.markdown(f"""
                    <div style="background-color: #f0fdf4; border-left: 5px solid #22c55e; padding: 12px 16px; border-radius: 4px; margin-bottom: 8px;">
                        <b>{factor['description']}</b><br>
                        <small>Impact Score: {factor['shap_value']:.3f} | Supports customer retention</small>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No dominant retention factors found for this customer.")

        # Interactive SHAP Waterfall / Bar Plot
        fig_shap = create_shap_waterfall_figure(pred_res["raw_explanation"])
        st.plotly_chart(fig_shap, use_container_width=True)

        st.markdown("---")

        # Retention Recommendations
        st.subheader("💡 Actionable Customer Retention Recommendations")
        with st.spinner("Generating tailored retention actions..."):
            rec_result = get_retention_recommendations(customer_dict, pred_res, prefer_llm=True)

        st.markdown(f"""
        <div class="recommendation-box">
            <b>Source:</b> {rec_result.get('source')} (<i>{rec_result.get('model_used')}</i>)<br>
            <b>Summary:</b> {rec_result.get('summary')}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Recommended Interventions:")
        for action in rec_result.get("actions", []):
            st.markdown(f"- {action}")

        # SQLite Database Logging
        rec_summary_text = "; ".join(rec_result.get("actions", []))
        rec_id = save_prediction(
            customer_data=customer_dict,
            prediction=pred_res["prediction"],
            probability=pred_res["probability"],
            risk_level=pred_res["risk_level"],
            top_factors=pred_res["risk_drivers"][:3],
            recommendation=rec_summary_text
        )
        st.success(f"💾 Prediction audit log saved to SQLite database (Record ID: **#{rec_id}**).")


# ==============================================================================
# 5. BATCH CSV PREDICTION
# ==============================================================================
elif menu == "📁 Batch CSV Prediction":
    st.markdown('<div class="main-header">Batch Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a CSV file containing multiple customer records for bulk scoring</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.markdown(f"**Uploaded Dataset:** `{len(batch_df):,}` rows x `{len(batch_df.columns)}` columns")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Process Batch Predictions", type="primary"):
                with st.spinner("Processing batch predictions with Random Forest & ColumnTransformer..."):
                    processed_batch = predictor.predict_batch(batch_df)

                st.success("Batch predictions completed successfully!")

                # Batch summary KPIs
                b1, b2, b3, b4 = st.columns(4)
                total_scored = len(processed_batch)
                high_count = (processed_batch["Risk_Level"] == "High").sum()
                med_count = (processed_batch["Risk_Level"] == "Medium").sum()
                low_count = (processed_batch["Risk_Level"] == "Low").sum()

                b1.metric("Scored Customers", f"{total_scored:,}")
                b2.metric("High Risk", f"{high_count:,}", f"{(high_count/total_scored)*100:.1f}%", delta_color="inverse")
                b3.metric("Medium Risk", f"{med_count:,}", f"{(med_count/total_scored)*100:.1f}%")
                b4.metric("Low Risk", f"{low_count:,}", f"{(low_count/total_scored)*100:.1f}%")

                st.markdown("---")
                filter_tier = st.multiselect("Filter by Risk Level:", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
                filtered_df = processed_batch[processed_batch["Risk_Level"].isin(filter_tier)]

                st.dataframe(filtered_df, use_container_width=True)

                # CSV Download Button
                csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Scored Customers CSV",
                    data=csv_bytes,
                    file_name=f"churn_predictions_batch_{datetime.now().strftime('%Y%m%d%H%M')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")
    else:
        st.info("ℹ️ Download the sample test batch or upload your customer dataset matching the IBM Telco schema.")
        sample_path = os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn.csv")
        if os.path.exists(sample_path):
            sample_preview = pd.read_csv(sample_path).head(10).drop(columns=["Churn"], errors="ignore")
            sample_csv = sample_preview.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Sample Customer CSV Template", data=sample_csv, file_name="sample_customers.csv", mime="text/csv")


# ==============================================================================
# 6. PREDICTION AUDIT LOG (SQLITE DATABASE)
# ==============================================================================
elif menu == "🗄️ Prediction Audit Log (Database)":
    st.markdown('<div class="main-header">Prediction History & Audit Log</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Persistent SQLite database records of past customer churn predictions</div>', unsafe_allow_html=True)

    summary = get_database_summary()
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Stored Records", f"{summary['total_predictions']:,}")
    d2.metric("High Risk Customers", f"{summary['high_risk_count']:,}", delta_color="inverse")
    d3.metric("Medium Risk Customers", f"{summary['medium_risk_count']:,}")
    d4.metric("Low Risk Customers", f"{summary['low_risk_count']:,}")

    st.markdown("---")
    history_df = get_all_predictions()

    if not history_df.empty:
        st.dataframe(
            history_df[[
                "id", "customer_id", "timestamp", "contract", "tenure", "monthly_charges",
                "churn_prediction", "churn_probability", "risk_level", "retention_recommendation"
            ]],
            use_container_width=True
        )

        # Record deletion option
        with st.expander("🗑️ Delete a Record by ID"):
            del_id = st.number_input("Record ID to Delete", min_value=1, step=1)
            if st.button("Delete Record"):
                if delete_prediction(int(del_id)):
                    st.success(f"Record #{del_id} deleted successfully.")
                    st.rerun()
                else:
                    st.error(f"Record #{del_id} not found.")

        # Export Database
        db_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export Full Database to CSV",
            data=db_csv,
            file_name="churn_prediction_database_export.csv",
            mime="text/csv"
        )
    else:
        st.info("No predictions have been logged to the database yet. Generate predictions in the Predictor tab to see records here.")
