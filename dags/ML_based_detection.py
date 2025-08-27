from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Paths
DATA_PATH = "/usr/local/airflow/dags/data/creditcard.csv"
OUTPUT_PATH = "/usr/local/airflow/dags/data/flagged_creditcard.csv"
DB_PATH = "/usr/local/airflow/dags/data/fraud.db"
MODEL_PATH = "/usr/local/airflow/dags/data/fraud_model.pkl"

# -----------------------------
# Functions
# -----------------------------

def extract_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Extracted {len(df)} records")
    # Optionally save a copy locally
    df.to_csv("/usr/local/airflow/dags/data/extracted_data.csv", index=False)

def train_model():
    df = pd.read_csv(DATA_PATH)
    
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain 'Class' column as label")
    
    X = df.drop(columns=['Class'])
    y = df['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save trained model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Random Forest model trained on {len(X_train)} records")

def predict_fraud():
    df = pd.read_csv(DATA_PATH)
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_model first.")
    
    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)
    
    # Predict
    X = df.drop(columns=['Class'])
    df['PredictedFraud'] = clf.predict(X)
    
    fraud_df = df[df['PredictedFraud'] == 1]
    fraud_df.to_csv(OUTPUT_PATH, index=False)

    print(f"ML-based fraud detection complete: {len(fraud_df)} flagged transactions")

def load_to_db():
    df = pd.read_csv(OUTPUT_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("fraudulent_transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} records into fraud.db")

# -----------------------------
# DAG definition
# -----------------------------

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "fraud_detection_pipeline_ml",
    default_args=default_args,
    description="ML-based Fraud Detection Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2025, 8, 1),
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    task2 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    task3 = PythonOperator(
        task_id="predict_fraud",
        python_callable=predict_fraud,
    )

    task4 = PythonOperator(
        task_id="load_to_db",
        python_callable=load_to_db,
    )

    # DAG flow
    task1 >> task2 >> task3 >> task4
