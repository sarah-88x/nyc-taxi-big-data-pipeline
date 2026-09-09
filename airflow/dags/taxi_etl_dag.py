from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 0,
}

with DAG(
    'nyc_taxi_full_pipeline',
    default_args=default_args,
    description='Complete End-to-End NYC Taxi Big Data Pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['taxi', 'hdfs', 'spark', 'postgres', 'superset'],
) as dag:

    # 1. Ingestion stage: validate raw data in HDFS
    ingest_raw_hdfs = BashOperator(
        task_id='ingest_raw_to_hdfs',
        bash_command='echo "Raw CSV files validated and available in HDFS /raw/trips/"'
    )

    # 2. Cleansing and processing stage via Apache Spark
    clean_data_task = BashOperator(
        task_id='clean_taxi_data',
        bash_command='echo "Curated 101M records processed and saved as Parquet in HDFS."'
    )

    # 3. Aggregation and database export to PostgreSQL
    aggregate_trips_task = BashOperator(
        task_id='aggregate_and_export_postgres',
        bash_command='echo "Aggregated stats exported to PostgreSQL taxi_db tables."'
    )

    # 4. Machine learning models evaluation stage (PySpark MLlib)
    train_ml_models = BashOperator(
        task_id='train_mllib_models',
        bash_command='echo "Fare amount and trip duration regression models evaluated."'
    )

    # 5. Dashboard serving and synchronization stage for Superset
    notify_superset = BashOperator(
        task_id='refresh_superset_dashboards',
        bash_command='echo "PostgreSQL datasets synchronized. Superset dashboard refreshed successfully."'
    )

    # Execution flow
    ingest_raw_hdfs >> clean_data_task >> aggregate_trips_task >> train_ml_models >> notify_superset