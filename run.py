"""
run.py - Central Project Launcher & Automation CLI
TY BSc Data Science Final-Year Project

Usage:
    python run.py              # Default: Launches the Streamlit Web Application
    python run.py --app        # Launches the Streamlit Web Application
    python run.py --train      # Executes complete ML training & evaluation pipeline
    python run.py --test       # Runs all unit and integration tests
    python run.py --predict    # Runs CLI prediction on preset demo profiles
"""

import sys
import os
import subprocess
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_training():
    """Executes the machine learning training pipeline."""
    print(">>> Launching ML Training Pipeline...")
    script_path = os.path.join(PROJECT_ROOT, "src", "train_model.py")
    subprocess.run([sys.executable, script_path], check=True)


def run_tests():
    """Runs the unit and integration test suite."""
    print(">>> Running Unit and Integration Tests...")
    subprocess.run([sys.executable, "-m", "unittest", "discover", "tests"], check=True)


def run_cli_prediction():
    """Executes quick CLI predictions on preset demo customers."""
    print(">>> Executing CLI Churn Prediction Demonstration...")
    from src.predict import ChurnPredictor, PRESET_PROFILES

    predictor = ChurnPredictor()
    for profile_name, customer_data in PRESET_PROFILES.items():
        print("\n" + "=" * 65)
        print(f"Profile: {profile_name}")
        print("=" * 65)
        res = predictor.predict_single(customer_data)
        print(f"Prediction:          {res['label']}")
        print(f"Churn Probability:   {res['probability_percent']}")
        print(f"Risk Tier:           {res['risk_level']}")
        print("Top Risk Factors (SHAP):")
        for driver in res["risk_drivers"][:3]:
            print(f"  * {driver['description']} ({driver['shap_value']:+.4f})")


def run_app():
    """Launches the Streamlit interactive dashboard."""
    app_path = os.path.join(PROJECT_ROOT, "app", "app.py")
    print(">>> Starting Streamlit Application at http://localhost:8501 ...")
    print("Press Ctrl+C to terminate the web server.")
    subprocess.run(["streamlit", "run", app_path], check=True)


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction System Launcher")
    parser.add_argument("--train", action="store_true", help="Train and evaluate all ML models")
    parser.add_argument("--test", action="store_true", help="Run automated test suite")
    parser.add_argument("--predict", action="store_true", help="Run sample CLI prediction")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit web dashboard")

    args = parser.parse_args()

    if args.train:
        run_training()
    elif args.test:
        run_tests()
    elif args.predict:
        run_cli_prediction()
    else:
        # Default action: launch Streamlit app
        run_app()


if __name__ == "__main__":
    main()
