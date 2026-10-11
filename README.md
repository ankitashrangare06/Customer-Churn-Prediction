# 🎯 Customer Churn Prediction & Retention System

> **TY (Third Year) BSc Data Science College Final-Year Project**  
> *An end-to-end Machine Learning and Explainable AI (SHAP) system with interactive Streamlit dashboard, SQLite persistence, and intelligent retention recommendations.*

---

## 📑 Table of Contents
1. [Project Title & Overview](#1-project-title--overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Key Features](#4-key-features)
5. [Technologies & Tools Used](#5-technologies--tools-used)
6. [System Architecture & Workflow](#6-system-architecture--workflow)
7. [Machine Learning Approach & Algorithms](#7-machine-learning-approach--algorithms)
8. [Dataset Information](#8-dataset-information)
9. [Project Folder Structure](#9-project-folder-structure)
10. [Environment Setup & Installation](#10-environment-setup--installation)
11. [How to Train the Models](#11-how-to-train-the-models)
12. [How to Run the Application](#12-how-to-run-the-application)
13. [How to Make Predictions](#13-how-to-make-predictions)
14. [Explainable AI (SHAP) Methodology](#14-explainable-ai-shap-methodology)
15. [Retention Recommendation Engine](#15-retention-recommendation-engine)
16. [Model Evaluation & Comparative Analysis](#16-model-evaluation--comparative-analysis)
17. [Database Implementation (SQLite)](#17-database-implementation-sqlite)
18. [Test Suite Execution](#18-test-suite-execution)
19. [Limitations](#19-limitations)
20. [Future Scope](#20-future-scope)
21. [College Viva & Interview Guide](#21-college-viva--interview-guide)

---

## 1. Project Title & Overview
**Project Title:** Customer Churn Prediction and Retention Strategy System  
**Academic Level:** Third Year (TY) BSc Data Science Final-Year Project  

This project delivers an industry-grade, locally deployable Machine Learning system that predicts whether a customer is likely to discontinue their subscription (churn). Beyond binary classification, the system classifies customers into actionable risk tiers (**Low, Medium, High**), calculates churn probability, uses **SHAP (SHapley Additive exPlanations)** to uncover the root causes of defection, and provides personalized retention suggestions.

---

## 2. Problem Statement
In competitive subscription industries (such as telecommunications and SaaS), acquiring a new customer costs between **5x to 25x more** than retaining an existing one. Most businesses only discover that a customer is unhappy after they cancel their contract, when it is already too late for preventive intervention.

By utilizing historical customer profiles, behavioral signals, and billing telemetry, this system detects early attrition patterns, allowing customer success teams to intervene with targeted commercial offers before defection occurs.

---

## 3. Objectives
- **Data Preprocessing & Validation:** Build a robust, reproducible data pipeline to handle missing values, blank entries, and categorical encodings without data leakage.
- **Feature Engineering:** Derive actionable domain features (customer lifecycle tiers, add-on service saturation, charges-to-tenure ratios).
- **Model Comparison:** Train and benchmark 4 algorithms: **Logistic Regression, Decision Tree, Random Forest, and XGBoost** using 5-Fold Stratified Cross-Validation.
- **Imbalance-Aware Optimization:** Prioritize **Recall, F1-Score, and ROC-AUC** over naive accuracy to detect at-risk customers effectively.
- **Explainable AI (XAI):** Implement **SHAP** to make black-box tree models transparent and interpretable for non-technical stakeholders.
- **Actionable Retention:** Generate commercial intervention recommendations via local open-source LLMs (Ollama with Qwen2.5-3B-Instruct) or expert rule-based logic.
- **Audit Logging:** Maintain a persistent local **SQLite** database tracking prediction history, risk levels, and timestamps.

---

## 4. Key Features
- **Interactive Multi-Tab Dashboard:** Built with Streamlit, providing EDA, Model Benchmarks, Single Prediction, Batch Scoring, and Database Audit History.
- **Dual-Model Inference:** Allows dynamic evaluation between Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- **Individual Risk Scoring:** Real-time risk probability gauge and colored risk badges:
  - 🟢 **Low Risk:** Probability < 35%
  - 🟡 **Medium Risk:** Probability between 35% and 65%
  - 🔴 **High Risk:** Probability ≥ 65%
- **Interactive SHAP Visuals:** Dynamic Plotly horizontal waterfall chart illustrating exact positive (risk driver) and negative (retention factor) attributions.
- **Batch CSV Upload:** Enables enterprise scoring of thousands of customer records with downloadable CSV output.
- **Persistent SQLite Audit Log:** Searchable and exportable history of customer predictions.
- **100% Free & Open-Source:** Operates locally on a standard student laptop with no paid API keys required.

---

## 5. Technologies & Tools Used
| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.10+ | Standard data science & ML ecosystem |
| **Data Processing** | Pandas, NumPy | Data cleaning, vectorization, transformation |
| **Machine Learning** | Scikit-Learn, XGBoost | Imbalance handling, ensemble methods, pipelines |
| **Explainable AI** | SHAP (TreeExplainer) | Game-theoretic feature attribution |
| **Web Dashboard** | Streamlit | Interactive, low-latency UI for Python |
| **Visualizations** | Plotly, Matplotlib, Seaborn | Interactive web charts and high-res report figures |
| **Storage** | SQLite | Serverless, zero-configuration local database |
| **Optional LLM** | Qwen2.5-3B-Instruct via Ollama | Local open-weight instruction-following retention assistant |
| **Testing** | unittest | Automated test suite |

---

## 6. System Architecture & Workflow
```
[ Customer Data (CSV / User Form) ]
                │
                ▼
    [ Data Cleaning & Validation ]
       ├── Handle Blank TotalCharges (tenure=0)
       ├── Drop Non-Predictive IDs (customerID)
       └── Binary Target Mapping (Yes/No -> 1/0)
                │
                ▼
       [ Feature Engineering ]
       ├── tenure_group (0-12m, 13-24m, 25-48m, 49-60m, >60m)
       ├── num_additional_services (Count of active security/streaming add-ons)
       ├── has_internet_service (Boolean flag)
       └── charges_per_tenure_ratio (MonthlyCharges / (tenure + 1))
                │
                ▼
  [ Scikit-Learn ColumnTransformer ]
       ├── Numerical: Median Imputer + StandardScaler
       └── Categorical: Mode Imputer + OneHotEncoder
                │
                ▼
   [ Stratified 5-Fold Cross-Validation ]
       ├── Logistic Regression (Balanced)
       ├── Decision Tree (Max Depth = 5)
       ├── Random Forest (150 Trees, Balanced)
       └── XGBoost (scale_pos_weight = 2.76)
                │
                ▼
   [ Best Model Selection (F1 & ROC-AUC) ]
                │
         ┌──────┴────────────────────────┐
         ▼                               ▼
 [ Churn Probability & Risk Tier ]   [ SHAP Explainability ]
 (Low < 35%, Med 35-65%, High ≥ 65%) (Top Risk & Retention Factors)
         │                               │
         └──────────────┬────────────────┘
                        ▼
       [ Retention Recommendation Engine ]
       (Local LLM via Ollama OR Rule-Based Expert Engine)
                        │
                        ▼
       [ Interactive Streamlit Dashboard ]
                        │
                        ▼
       [ Persistent SQLite Database Log ]
```

---

## 7. Machine Learning Approach & Algorithms
Four distinct machine learning algorithms were trained and compared:

1. **Logistic Regression:**
   - **Role:** Interpretable linear baseline model.
   - **Configuration:** `class_weight='balanced'`, `max_iter=1000`.
   - **Strength:** Fast inference, linear coefficient transparency.

2. **Decision Tree Classifier:**
   - **Role:** Non-linear rule-based baseline.
   - **Configuration:** `max_depth=5`, `min_samples_split=15`, `class_weight='balanced'`.
   - **Strength:** Easily human-interpretable orthogonal decision rules.

3. **Random Forest Classifier (Selected Best Model):**
   - **Role:** Bagging ensemble of de-correlated decision trees.
   - **Configuration:** `n_estimators=150`, `max_depth=8`, `class_weight='balanced'`, `n_jobs=-1`.
   - **Strength:** Exceptional resistance to overfitting, balanced sensitivity (**79.95% Recall**).

4. **XGBoost (Extreme Gradient Boosting):**
   - **Role:** Sequential gradient boosting optimizing pseudo-residuals.
   - **Configuration:** `n_estimators=150`, `max_depth=4`, `learning_rate=0.08`, `scale_pos_weight=2.76`.
   - **Strength:** High discriminatory power on tabular datasets (**0.8412 ROC-AUC**).

---

## 8. Dataset Information
- **Source:** IBM Telco Customer Churn Public Benchmark Dataset
- **Total Records:** 7,043 customer accounts
- **Target Variable:** `Churn` (Yes = 1,869 [26.54%], No = 5,174 [73.46%])
- **Attributes (21 original columns):**
  - **Identifiers:** `customerID` (excluded from predictive modeling)
  - **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - **Account/Contract:** `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
  - **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

## 9. Project Folder Structure
```
Customer Churn Prediction/
│
├── data/
│   ├── raw/
│   │   └── customer_churn.csv              # Original IBM Telco dataset
│   ├── processed/
│   │   └── cleaned_customer_churn.csv      # Cleaned and feature-engineered dataset
│   └── churn_predictions.db                # SQLite audit database
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # EDA, distributions, churn patterns
│   ├── 02_data_preprocessing.ipynb         # Data cleaning & ColumnTransformer pipeline
│   ├── 03_model_training.ipynb             # Model training & 5-fold cross-validation
│   ├── 04_model_evaluation.ipynb           # Test metrics, ROC curves & SHAP explanation
│   └── generate_notebooks.py               # Notebook generator script
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── database.py                         # SQLite database connection & CRUD operations
│   ├── feature_engineering.py              # Domain feature creation logic
│   ├── preprocessing.py                    # Preprocessing pipeline & ColumnTransformer
│   ├── train_model.py                      # Multi-model training & model selection
│   ├── evaluate_model.py                   # Classification metrics & figure plotting
│   ├── predict.py                          # Inference service & risk classification
│   └── explain.py                          # SHAP explainability & feature attributions
│
├── models/
│   ├── churn_model.pkl                     # Serialized best model (Random Forest)
│   ├── preprocessing_pipeline.pkl          # Fitted ColumnTransformer pipeline
│   ├── all_models.pkl                      # Dictionary of all 4 trained models
│   └── model_metadata.json                 # Metrics, feature names & risk thresholds
│
├── app/
│   └── app.py                              # Interactive Streamlit Web Application
│
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png          # Target class balance pie chart
│   │   ├── contract_vs_churn.png           # Churn rate across contract types
│   │   ├── monthly_charges_vs_churn.png    # Monthly charges distribution KDE
│   │   ├── tenure_vs_churn.png             # Tenure boxplot comparison
│   │   ├── model_comparison.png            # Grouped bar chart comparing 4 models
│   │   ├── roc_curves.png                  # Overlaid ROC curves with AUC scores
│   │   ├── confusion_matrix_best.png       # Annotated confusion matrix heatmap
│   │   └── feature_importance.png          # Top 15 predictive features
│   └── model_results.csv                   # Model comparison metrics summary table
│
├── llm/
│   ├── __init__.py                         # LLM package initialization
│   └── recommendation.py                   # Retention advisor (Ollama + Expert Rules)
│
├── tests/
│   ├── __init__.py                         # Test package initialization
│   ├── test_preprocessing.py               # Tests for cleaning and feature engineering
│   ├── test_model.py                       # Tests for model loading and metadata
│   └── test_prediction.py                  # Tests for predictions and SQLite logging
│
├── requirements.txt                        # Complete Python dependencies
│
├── README.md                               # Project documentation & viva guide
├── .gitignore                              # Git exclusion rules
├── LICENSE                                 # MIT open-source license
└── run.py                                  # Central project launcher CLI
```

---

## 10. Environment Setup & Installation

### Step 1: Open Terminal / PowerShell in Project Directory
```powershell
cd "c:\Users\Pragati\Downloads\Customer Churn Prediction"
```

### Step 2: (Optional but Recommended) Create Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 11. How to Train the Models
To re-train all 4 models, perform 5-fold Stratified Cross-Validation, regenerate report figures, and select the best model:

```powershell
python run.py --train
```
*Or directly via script:*
```powershell
python src/train_model.py
```

**Output artifacts created:**
- `models/churn_model.pkl` (Best classifier)
- `models/preprocessing_pipeline.pkl` (Fitted ColumnTransformer)
- `models/all_models.pkl` (All 4 models for comparisons)
- `models/model_metadata.json` (Full training metadata)
- `reports/model_results.csv` (Metrics table)
- `reports/figures/*.png` (All 8 high-resolution charts)

---

## 12. How to Run the Application
Launch the interactive Streamlit dashboard:

```powershell
python run.py --app
```
*Or directly via Streamlit:*
```powershell
streamlit run app/app.py
```

The application will launch in your browser at:
`http://localhost:8501`

---

## 13. How to Make Predictions

### Method A: Via Interactive Streamlit Web UI
1. Navigate to **🔮 Customer Churn Predictor & SHAP** in the sidebar.
2. Choose a quick preset (e.g. *High Risk*, *Low Risk*, *Medium Risk*) or enter custom customer attributes.
3. Click **🔮 Predict Customer Churn Risk**.
4. View the churn probability gauge, risk level badge, top SHAP drivers, and actionable retention recommendations.
5. The prediction is automatically logged to the SQLite database.

### Method B: Via CLI Launcher
```powershell
python run.py --predict
```

### Method C: Programmatically in Python
```python
from src.predict import ChurnPredictor, PRESET_PROFILES

predictor = ChurnPredictor()
customer = PRESET_PROFILES["High Risk (New Customer, Month-to-Month, High Cost)"]
result = predictor.predict_single(customer)

print("Prediction:", result["label"])               # Churn
print("Probability:", result["probability_percent"]) # 90.3%
print("Risk Level:", result["risk_level"])           # High
```

---

## 14. Explainable AI (SHAP) Methodology
Machine learning models are frequently criticized in business environments as "black boxes." In customer retention, business executives need to know **why** a customer is predicted to churn.

This system integrates **SHAP (SHapley Additive exPlanations)** based on cooperative game theory:
$$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \Big( f(S \cup \{i\}) - f(S) \Big)$$

- **Positive SHAP Values (+):** Attributes pushing the model toward predicting **Churn** (e.g., Month-to-month contract, short tenure, high monthly charge).
- **Negative SHAP Values (-):** Attributes pushing the model toward predicting **Retention** (e.g., Two-year contract, bundled security services, long tenure).
- **Waterfall / Horizontal Impact Chart:** Rendered in Plotly within the Streamlit UI for each prediction.

---

## 15. Retention Recommendation Engine
Per the PRD and TRD specifications:
- **Rule of Separation:** The LLM or rule engine **never** predicts whether a customer will churn. The machine learning model is the sole source of truth.
- **Input to Recommendation Engine:** Churn Probability, Risk Level, Top SHAP drivers, and Customer Attributes.
- **Two Operating Modes:**
  1. **Local LLM via Ollama (Qwen2.5-3B-Instruct):** If Ollama is running locally at `http://localhost:11434`, it generates concise, professional retention suggestions.
  2. **Domain Expert Rule Engine (Zero-Latency Fallback):** If Ollama is offline or not installed, the system automatically uses heuristic retention logic tailored to specific customer pain points (e.g., contract discounts, tech support trials, autopay incentives).

---

## 16. Model Evaluation & Comparative Analysis

Evaluation on held-out test set ($N = 1,409$ samples):

| Algorithm | Test Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | 5-Fold CV ROC-AUC | 5-Fold CV F1 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest (Best)** | **75.94%** | **0.5311** | **79.95%** | **0.6382** | **0.8428** | **0.8437** | **0.6277** |
| Decision Tree | 75.73% | 0.5283 | 77.54% | 0.6291 | 0.8350 | 0.8258 | 0.6180 |
| XGBoost | 74.66% | 0.5147 | 78.07% | 0.6206 | 0.8412 | 0.8414 | 0.6255 |
| Logistic Regression | 73.74% | 0.5034 | 77.27% | 0.6097 | 0.8459 | 0.8456 | 0.6288 |

### Why Random Forest Was Selected:
1. **Highest Recall (79.95%):** Captures nearly 80% of all genuine churners. In customer retention, the business penalty of a False Negative (losing a customer unawares) is much higher than a False Positive (offering a discount to a loyal customer).
2. **Highest F1-Score (0.6382):** Maintains the best harmonic balance between precision and recall on imbalanced tabular data.
3. **High ROC-AUC (0.8428):** Demonstrates excellent ranking separation between churners and non-churners across all possible decision thresholds.

---

## 17. Database Implementation (SQLite)
The application includes a local SQLite database at `data/churn_predictions.db`.
- **Table:** `customer_predictions`
- **Fields:** `id`, `customer_id`, `timestamp`, demographics, subscribed services, contract details, monthly charges, `churn_prediction`, `churn_probability`, `risk_level`, `top_factors` (JSON), and `retention_recommendation`.
- **UI Access:** Available in the **🗄️ Prediction Audit Log (Database)** tab with summary KPI cards, record deletion, and CSV export.

---

## 18. Test Suite Execution
To verify that all modules, preprocessors, models, and predictions are functioning:

```powershell
python run.py --test
```
*Or using Python's unittest module:*
```powershell
python -m unittest discover tests
```

**Test Coverage:**
- `test_preprocessing.py`: Validates cleaning of whitespace TotalCharges, target mapping, feature engineering, and ColumnTransformer shapes.
- `test_model.py`: Validates presence and interface compliance of all 4 serialized models, preprocessors, and metadata.
- `test_prediction.py`: Validates single prediction, batch prediction, risk tier boundaries, and SQLite database transactions.

---

## 19. Limitations
1. **Dataset Scope:** The dataset represents telecom customers; applying the model to banking or SaaS requires retraining on domain-specific features.
2. **Static Snapshot:** The dataset captures a cross-sectional snapshot rather than full time-series clickstream activity.
3. **Imbalance Constraints:** Despite class weighting, precision is around 53%, meaning approximately 47% of flagged customers might have stayed even without intervention (conservative intervention policy).

---

## 20. Future Scope
- **Real-Time Streaming Ingestion:** Integrate Apache Kafka or FastAPI Webhook endpoints for real-time CRM updates.
- **Dynamic Threshold Optimization:** Calibrate decision thresholds using customer Lifetime Value (LTV) cost-benefit matrices.
- **Deep Learning Comparison:** Experiment with TabNet or compact Multi-Layer Perceptrons (MLP) for tabular representation learning.
- **Automated Retention Tracking:** Track whether customers who received recommended interventions stayed, establishing a closed-loop causal reinforcement learning system.

---

## 21. College Viva & Interview Guide

### Common Viva Questions & Model Answers

#### Q1: Why did you not just use Accuracy to evaluate your model?
> *"The churn dataset exhibits moderate class imbalance, with only 26.5% of customers churning. A trivial dummy model that predicts 'No Churn' for every customer would achieve 73.5% accuracy but detect 0% of churners, making it useless for business retention. Therefore, we optimized for **Recall** (capturing actual churners) and **F1-Score** (harmonic mean of Precision and Recall), along with **ROC-AUC**."*

#### Q2: How did you prevent Data Leakage during preprocessing?
> *"We strictly separated the train and test splits before fitting the Scikit-learn `ColumnTransformer`. The median values for numerical imputation, standard scaler mean/variance, and one-hot categories were learned exclusively from the training split and subsequently applied to the test split."*

#### Q3: Why is SHAP better than default feature importance (MDI)?
> *"Mean Decrease in Impurity (MDI / Gini importance) is calculated across the entire tree training process and suffers from bias toward high-cardinality features. It only gives a global score and cannot explain an individual prediction. In contrast, SHAP values are rooted in game theory (Shapley values), provide local explanations for any specific customer, and tell us whether a feature pushed the prediction toward or away from churn."*

#### Q4: What are the primary business insights uncovered by your models?
> *"Three dominant patterns emerge: (1) Contract duration is the single strongest factor—Month-to-month subscribers churn at 42.7% versus 2.8% for Two-year contracts; (2) The first 12 months (tenure) represent the critical retention window; and (3) Fiber optic customers without TechSupport or OnlineSecurity churn significantly faster due to unassisted technical friction."*

---

**Project Author:** TY BSc Data Science Student  
**Academic Year:** 2025–2026  
**License:** MIT License  
