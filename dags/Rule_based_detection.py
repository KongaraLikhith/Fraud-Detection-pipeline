from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sqlite3

# Paths
DATA_PATH = "/usr/local/airflow/dags/data/creditcard.csv"
OUTPUT_PATH = "/usr/local/airflow/dags/data/flagged_creditcard.csv"
DB_PATH = "/usr/local/airflow/dags/data/fraud.db"

# Functions
def extract_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Extracted {len(df)} records")

def detect_fraud():
    df = pd.read_csv(DATA_PATH)

    # Rule 1: High amount threshold
    rule1 = df['Amount'] > 10000

    # Rule 2: Statistical outlier detection (z-score)
    df['zscore'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    rule2 = np.abs(df['zscore']) > 3

    fraud_df = df[rule1 | rule2]
    fraud_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Fraud detection complete. {len(fraud_df)} suspicious transactions found.")

def load_to_db():
    df = pd.read_csv(OUTPUT_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("fraudulent_transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} records into fraud.db")

# DAG definition
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "fraud_detection_pipeline",
    default_args=default_args,
    description="Simple Fraud Detection Pipeline",
    schedule_interval="@daily",
    start_date=datetime(2025, 8, 1),
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data,
    )

    task2 = PythonOperator(
        task_id="detect_fraud",
        python_callable=detect_fraud,
    )

    task3 = PythonOperator(
        task_id="load_to_db",
        python_callable=load_to_db,
    )

    task1 >> task2 >> task3
