# Fraud Detection Airflow Project

This repository contains an **Apache Airflow** project with **three DAGs** that demonstrate both rule-based and machine learning–based fraud detection pipelines.  
The setup runs on **Docker** with Airflow, and uses **SQLite** and **PostgreSQL** for storage.

---

## 🚀 Project Overview

### 1. Rule-Based Fraud Detection (SQLite)
- **Steps**:
  - Extract transactions  
  - Detect fraud using rule-based logic  
  - Load results into **SQLite database**

### 2. ML Model Fraud Detection (SQLite + Email Alert)
- **Steps**:
  - Extract transactions  
  - Train ML model  
  - Predict fraud  
  - Check and send **email alert** if fraud detected  
  - Load results into **SQLite database**

### 3. ML Model Fraud Detection (PostgreSQL)
- **Same as DAG 2**, but results are stored in **PostgreSQL database** instead of SQLite.

---
Screenshots & Outputs

To demonstrate the working of the pipeline, I have included a screenshots.docx file.
It contains:
DAGs running in Airflow
Graph/Tree view of each DAG
Logs for successful runs
Output tables from PostgreSQL

🚀 Future Enhancements-
Automate Tableau dashboards directly from PostgreSQL
Move ML model training to AWS Sagemaker / Lambda for scalability
Enable streaming pipeline with Kafka

## 📂 Project Structure

├── dags/ # Airflow DAGs
│ ├── rule_based_detection.py
│ ├── ml_based_detection.py
│ ├── postgre_instd_sqlite.py
  ├── data
    ├── creditcard.csv
    ├── flagged_data.csv
    ├── fraud.db(sqlite)
    ├── fraud_model.pkl
├── docker-compose-LocalExecutor.yml # Airflow + Docker Compose setup
├── .gitignore # Ignore unnecessary files
└── README.md # Project documentation

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

docker-compose -f docker-compose-LocalExecutor.yml up
Open: http://localhost:8080
Default login: airflow / airflow

-- To access fraud detection results:
docker exec -it postgres_container_name psql -U airflow -d airflow
\dt
SELECT * FROM fraud_transactions LIMIT 10;

