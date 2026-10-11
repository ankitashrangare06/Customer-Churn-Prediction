"""
recommendation.py - Customer Retention Recommendation Module
TY BSc Data Science Project

Generates actionable customer retention recommendations based on ML churn predictions
and SHAP feature impacts. Supports local open-weight LLMs via Ollama (Qwen2.5-3B-Instruct / Phi-3.5)
with an automatic rule-based retention engine fallback when Ollama is unavailable.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
import requests

# Default Ollama configuration (Free & Open Source)
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:3b"


def check_ollama_status(ollama_url: str = DEFAULT_OLLAMA_URL) -> bool:
    """Checks whether local Ollama daemon is running and reachable."""
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=1.5)
        return response.status_code == 200
    except Exception:
        return False


def generate_rule_based_recommendation(
    customer_data: Dict[str, Any],
    prediction_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Intelligent domain-specific customer retention recommendation engine.
    Analyzes specific customer pain points and SHAP drivers to generate
    tailored commercial interventions.
    """
    risk_level = prediction_result.get("risk_level", "Medium")
    probability_pct = prediction_result.get("probability_percent", "N/A")
    contract = customer_data.get("Contract", "Month-to-month")
    tenure = int(customer_data.get("tenure", 0))
    monthly_charges = float(customer_data.get("MonthlyCharges", 0.0))
    payment = customer_data.get("PaymentMethod", "")
    internet = customer_data.get("InternetService", "")
    tech_support = customer_data.get("TechSupport", "No")
    online_sec = customer_data.get("OnlineSecurity", "No")

    actions = []

    if risk_level == "Low":
        actions.append(
            "Account Appreciation: Customer demonstrates high loyalty and low defection probability. "
            "Send an anniversary thank-you note and loyalty points rather than aggressive sales outreach."
        )
        if contract != "Two year":
            actions.append(
                "Loyalty Lock-in: Offer a VIP renewal incentive for a 2-year contract with guaranteed rate lock."
            )
        return {
            "source": "Rule-Based Expert Engine",
            "model_used": "Domain Expert Heuristics",
            "risk_level": risk_level,
            "actions": actions,
            "summary": "Customer is stable. Focus on relationship maintenance and appreciation."
        }

    # High or Medium Risk Interventions
    if contract == "Month-to-month":
        actions.append(
            "Contract Incentive: Customer is currently uncommitted on a Month-to-month contract. "
            "Propose a 1-year contract bundle with a 15% discount for the first 6 months to secure commitment."
        )

    if monthly_charges > 75.0:
        actions.append(
            f"Pricing Relief: High monthly charges (${monthly_charges:.2f}) increase price sensitivity. "
            "Audit feature usage and offer an optimized value bundle or waive equipment rental fees ($10/mo savings)."
        )

    if internet == "Fiber optic" and (tech_support != "Yes" or online_sec != "Yes"):
        actions.append(
            "Tech Support & Security Bundle: High-speed fiber customers without technical support "
            "frequently churn due to unassisted outages. Grant 3 months of complimentary 24/7 TechSupport & OnlineSecurity."
        )

    if "Electronic check" in payment:
        actions.append(
            "Payment Method Switch: Electronic check payers exhibit the highest attrition rate. "
            "Incentivize automated payment (Credit Card or Direct Bank Transfer) with a one-time $15 bill credit."
        )

    if tenure <= 12:
        actions.append(
            f"Onboarding Concierge: Customer is in the critical early lifecycle ({tenure} months). "
            "Schedule a proactive check-in call from customer success to resolve any onboarding frustrations."
        )

    if not actions:
        actions.append(
            "Targeted Outreach: Initiate proactive customer satisfaction survey with a retention discount code."
        )

    return {
        "source": "Rule-Based Expert Engine",
        "model_used": "Domain Expert Heuristics (Zero-Latency Fallback)",
        "risk_level": risk_level,
        "actions": actions[:3],  # Top 3 actions
        "summary": f"Customer churn probability is {probability_pct} ({risk_level} Risk). Immediate preventive intervention recommended."
    }


def query_ollama_retention_advisor(
    customer_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL
) -> Optional[Dict[str, Any]]:
    """
    Queries local Ollama runtime hosting Qwen2.5-3B-Instruct or Phi-3.5-mini.
    Adheres strictly to PRD/TRD rule: The LLM does NOT predict churn, only generates suggestions.
    """
    prompt = f"""You are an expert Telecom Customer Retention Specialist.
The Machine Learning classification model has ALREADY calculated the churn probability.
Do NOT recalculate or contradict the prediction.

Customer Profile:
- Churn Risk Level: {prediction_result.get('risk_level')}
- Churn Probability: {prediction_result.get('probability_percent')}
- Contract: {customer_data.get('Contract')}
- Tenure: {customer_data.get('tenure')} months
- Monthly Charges: ${customer_data.get('MonthlyCharges')}
- Internet Service: {customer_data.get('InternetService')}
- Tech Support: {customer_data.get('TechSupport')}
- Payment Method: {customer_data.get('PaymentMethod')}

Top Machine Learning Risk Factors:
{', '.join([d.get('description', '') for d in prediction_result.get('risk_drivers', [])[:4]])}

TASK:
Provide exactly 3 concise, highly practical retention actions that the customer success team should execute to retain this customer.
Format as bullet points with bold titles. Keep your answer brief and professional.
"""

    try:
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.4,
                "top_p": 0.9,
                "max_tokens": 250
            }
        }
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=12.0)
        if resp.status_code == 200:
            result = resp.json()
            response_text = result.get("response", "").strip()
            # Split response into bullets
            lines = [line.strip() for line in response_text.split("\n") if line.strip().startswith(("-", "*", "1.", "2.", "3."))]
            actions = lines if lines else [response_text]

            return {
                "source": "Local LLM",
                "model_used": model_name,
                "risk_level": prediction_result.get("risk_level"),
                "actions": actions[:3],
                "summary": response_text
            }
    except Exception:
        pass
    return None


def get_retention_recommendations(
    customer_data: Dict[str, Any],
    prediction_result: Dict[str, Any],
    prefer_llm: bool = True,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_name: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Main entry point for retention recommendations.
    Attempts local LLM generation if requested and available; otherwise uses expert rule engine.
    """
    if prefer_llm and check_ollama_status(ollama_url):
        llm_result = query_ollama_retention_advisor(customer_data, prediction_result, ollama_url, model_name)
        if llm_result:
            return llm_result

    # Fallback to Domain Expert Engine
    return generate_rule_based_recommendation(customer_data, prediction_result)


if __name__ == "__main__":
    print("Testing recommendation engine...")
    sample_cust = {
        "tenure": 2,
        "Contract": "Month-to-month",
        "MonthlyCharges": 94.85,
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check"
    }
    sample_pred = {
        "risk_level": "High",
        "probability_percent": "90.3%",
        "risk_drivers": [{"description": "Month-to-month contract"}, {"description": "High monthly charges"}]
    }

    recs = get_retention_recommendations(sample_cust, sample_pred, prefer_llm=False)
    print("\nRecommendation Source:", recs["source"])
    print("Actionable Suggestions:")
    for a in recs["actions"]:
        print(" *", a)
