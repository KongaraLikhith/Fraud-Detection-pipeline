from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# -----------------------------
# Paths
# -----------------------------
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
    df.to_csv("/usr/local/airflow/dags/data/extracted_data.csv", index=False)

def train_model():
    df = pd.read_csv(DATA_PATH)
    if 'Class' not in df.columns:
        raise ValueError("Dataset must contain 'Class' column as label")
    
    X = df.drop(columns=['Class'])
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)

    print(f"Random Forest model trained on {len(X_train)} records")

def predict_fraud(**kwargs):
    df = pd.read_csv(DATA_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_model first.")

    with open(MODEL_PATH, 'rb') as f:
        clf = pickle.load(f)

    X = df.drop(columns=['Class'])
    df['PredictedFraud'] = clf.predict(X)

    fraud_df = df[df['PredictedFraud'] == 1]
    fraud_df.to_csv(OUTPUT_PATH, index=False)
    fraud_count = len(fraud_df)
    print(f"ML-based fraud detection complete: {fraud_count} flagged transactions")

    kwargs['ti'].xcom_push(key='fraud_count', value=fraud_count)

def load_to_db():
    df = pd.read_csv(OUTPUT_PATH)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("fraudulent_transactions", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Loaded {len(df)} records into fraud.db")

def fraud_alert_condition(**kwargs):
    fraud_count = kwargs['ti'].xcom_pull(task_ids='predict_fraud', key='fraud_count')
    if fraud_count > 0:
        return 'send_alert'
    else:
        return 'skip_all'

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
    "fraud_detection_pipeline_ml_alert",
    default_args=default_args,
    description="ML-based Fraud Detection Pipeline with Alerts",
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
        provide_context=True
    )

    branch_task = BranchPythonOperator(
        task_id='check_fraud',
        python_callable=fraud_alert_condition,
        provide_context=True
    )

    alert_task = EmailOperator(
        task_id='send_alert',
        to='yourmail@gmail.com',
        subject='Fraud Alert: {{ ti.xcom_pull(task_ids="predict_fraud", key="fraud_count") }} suspicious transactions detected',
        html_content="""<h3>Alert!</h3>
                        <p>Suspicious transactions were flagged by the ML model. Check flagged_creditcard.csv for details.</p>""",
    )

    db_task = PythonOperator(
        task_id="load_to_db",
        python_callable=load_to_db,
    )

    skip_task = PythonOperator(
        task_id='skip_all',
        python_callable=lambda: print("No fraud detected. Skipping alert and DB load."),
    )

    # -----------------------------
    # DAG flow
    # -----------------------------
    task1 >> task2 >> task3 >> branch_task
    branch_task >> alert_task >> db_task
    branch_task >> skip_task
