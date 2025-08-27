from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime, timedelta
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# -----------------------------
# Paths & Postgres config
# -----------------------------
DATA_PATH = "/usr/local/airflow/dags/data/creditcard.csv"
OUTPUT_PATH = "/usr/local/airflow/dags/data/flagged_creditcard.csv"
MODEL_PATH = "/usr/local/airflow/dags/data/fraud_model.pkl"

# Postgres connection details
PG_HOST = "postgres"
PG_DB = "airflow"
PG_USER = "airflow"
PG_PASSWORD = "airflow"
PG_TABLE = "fraudulent_transactions"

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

def load_to_postgres():
    if not os.path.exists(OUTPUT_PATH):
        print("No flagged transactions file found. Skipping DB load.")
        return

    df = pd.read_csv(OUTPUT_PATH)
    conn = psycopg2.connect(
        host=PG_HOST,
        database=PG_DB,
        user=PG_USER,
        password=PG_PASSWORD
    )
    cur = conn.cursor()

    # Create table if it doesn't exist
    cols = ",".join([f"{col} TEXT" for col in df.columns])
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {PG_TABLE} (
            id SERIAL PRIMARY KEY,
            {cols},
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

    # Insert data
    for i, row in df.iterrows():
        values = "','".join(map(str, row.tolist()))
        cur.execute(f"INSERT INTO {PG_TABLE} ({','.join(df.columns)}) VALUES ('{values}');")
    conn.commit()
    cur.close()
    conn.close()
    print(f"Appended {len(df)} records to Postgres table '{PG_TABLE}'")

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
    "fraud_detection_pipeline_ml_alert_postgres",
    default_args=default_args,
    description="ML-based Fraud Detection Pipeline with Alerts and Postgres storage",
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
        to='your-mail@gmail.com',
        subject='Fraud Alert: {{ ti.xcom_pull(task_ids="predict_fraud", key="fraud_count") }} suspicious transactions detected',
        html_content="""<h3>Alert!</h3>
                        <p>Suspicious transactions were flagged by the ML model. Check flagged_creditcard.csv for details.</p>""",
    )

    db_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres,
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
