"""
database.py - SQLite Database Handler for Customer Churn Prediction
TY BSc Data Science Project

Manages local SQLite database operations to store and retrieve prediction logs,
risk scores, and customer audit histories.
"""

import os
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Default database path
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "churn_predictions.db")


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Creates and returns a connection to the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes the database schema if tables do not already exist."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customer_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            timestamp TEXT NOT NULL,
            gender TEXT,
            senior_citizen INTEGER,
            partner TEXT,
            dependents TEXT,
            tenure INTEGER,
            phone_service TEXT,
            multiple_lines TEXT,
            internet_service TEXT,
            online_security TEXT,
            online_backup TEXT,
            device_protection TEXT,
            tech_support TEXT,
            streaming_tv TEXT,
            streaming_movies TEXT,
            contract TEXT,
            paperless_billing TEXT,
            payment_method TEXT,
            monthly_charges REAL,
            total_charges REAL,
            churn_prediction INTEGER NOT NULL,
            churn_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            top_factors TEXT,
            retention_recommendation TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(
    customer_data: Dict[str, Any],
    prediction: int,
    probability: float,
    risk_level: str,
    top_factors: Optional[List[Dict[str, Any]]] = None,
    recommendation: Optional[str] = None,
    db_path: str = DB_PATH
) -> int:
    """
    Saves a customer churn prediction record into the SQLite database.

    Returns:
        int: The inserted record ID.
    """
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    cust_id = customer_data.get("customerID") or customer_data.get("customer_id") or f"CUST-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    factors_json = json.dumps(top_factors) if top_factors else "[]"

    cursor.execute("""
        INSERT INTO customer_predictions (
            customer_id, timestamp, gender, senior_citizen, partner, dependents,
            tenure, phone_service, multiple_lines, internet_service, online_security,
            online_backup, device_protection, tech_support, streaming_tv,
            streaming_movies, contract, paperless_billing, payment_method,
            monthly_charges, total_charges, churn_prediction, churn_probability,
            risk_level, top_factors, retention_recommendation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(cust_id),
        timestamp_str,
        str(customer_data.get("gender", "Unknown")),
        int(customer_data.get("SeniorCitizen", customer_data.get("senior_citizen", 0))),
        str(customer_data.get("Partner", "No")),
        str(customer_data.get("Dependents", "No")),
        int(customer_data.get("tenure", 0)),
        str(customer_data.get("PhoneService", "Yes")),
        str(customer_data.get("MultipleLines", "No")),
        str(customer_data.get("InternetService", "DSL")),
        str(customer_data.get("OnlineSecurity", "No")),
        str(customer_data.get("OnlineBackup", "No")),
        str(customer_data.get("DeviceProtection", "No")),
        str(customer_data.get("TechSupport", "No")),
        str(customer_data.get("StreamingTV", "No")),
        str(customer_data.get("StreamingMovies", "No")),
        str(customer_data.get("Contract", "Month-to-month")),
        str(customer_data.get("PaperlessBilling", "Yes")),
        str(customer_data.get("PaymentMethod", "Electronic check")),
        float(customer_data.get("MonthlyCharges", customer_data.get("monthly_charges", 0.0))),
        float(customer_data.get("TotalCharges", customer_data.get("total_charges", 0.0))),
        int(prediction),
        round(float(probability), 4),
        str(risk_level),
        factors_json,
        str(recommendation or "")
    ))

    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id


def get_all_predictions(db_path: str = DB_PATH) -> pd.DataFrame:
    """Retrieves all logged predictions as a pandas DataFrame."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    query = "SELECT * FROM customer_predictions ORDER BY id DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def delete_prediction(record_id: int, db_path: str = DB_PATH) -> bool:
    """Deletes a prediction record by its primary key ID."""
    init_db(db_path)
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM customer_predictions WHERE id = ?", (record_id,))
    rows_affected = cursor.rowcount
    conn.commit()
    conn.close()
    return rows_affected > 0


def get_database_summary(db_path: str = DB_PATH) -> Dict[str, Any]:
    """Computes summary statistics for historical logged predictions."""
    df = get_all_predictions(db_path)
    if df.empty:
        return {
            "total_predictions": 0,
            "churn_count": 0,
            "retained_count": 0,
            "high_risk_count": 0,
            "medium_risk_count": 0,
            "low_risk_count": 0,
            "avg_probability": 0.0
        }

    return {
        "total_predictions": int(len(df)),
        "churn_count": int((df["churn_prediction"] == 1).sum()),
        "retained_count": int((df["churn_prediction"] == 0).sum()),
        "high_risk_count": int((df["risk_level"] == "High").sum()),
        "medium_risk_count": int((df["risk_level"] == "Medium").sum()),
        "low_risk_count": int((df["risk_level"] == "Low").sum()),
        "avg_probability": float(df["churn_probability"].mean())
    }


if __name__ == "__main__":
    init_db()
    print("Database initialized at:", DB_PATH)
    sample_data = {
        "customerID": "DEMO-001",
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
        "MonthlyCharges": 85.5,
        "TotalCharges": 171.0
    }
    rec_id = save_prediction(
        sample_data,
        prediction=1,
        probability=0.82,
        risk_level="High",
        top_factors=[{"feature": "Contract_Month-to-month", "impact": "High Risk", "value": "Month-to-month"}],
        recommendation="Offer discounted 1-year contract"
    )
    print(f"Sample prediction saved with ID: {rec_id}")
    summary = get_database_summary()
    print("Database summary:", summary)
