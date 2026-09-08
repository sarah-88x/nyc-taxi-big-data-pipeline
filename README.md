# NYC Smart Taxi - Big Data Engineering Pipeline

An end-to-end Big Data & Analytics platform processing NYC Yellow Taxi trip records (~7.5M rows) using batch processing, streaming ingestion, machine learning, and interactive business intelligence dashboards.

## Architecture & Pipeline
- **Distributed Storage:** Apache Hadoop HDFS
- **Batch Processing:** Apache Spark (PySpark Data Cleansing & Aggregations)
- **Streaming Ingestion:** Apache Kafka & Spark Structured Streaming
- **Machine Learning:** Spark MLlib (Fare, Duration, Demand, & Anomaly models)
- **Pipeline Orchestration:** Apache Airflow
- **Serving Layer:** PostgreSQL (JDBC Data Warehouse)
- **BI & Visualization:** Apache Superset

## Project Structure
```text
nyp-taki/
├── airflow/             # DAG definitions and pipeline orchestration
├── dashboards/          # Superset dashboard exports and configs
├── docker/              # Docker Compose environment setup
├── kafka/               # Kafka streaming producer
├── postgres/            # Database schema & init scripts
├── spark/
│   ├── batch/           # Batch ETL and aggregation jobs
│   ├── mllib/           # ML models (fare, duration, demand, anomaly)
│   └── stream/          # Spark Structured Streaming consumer
├── .gitignore
└── README.md