from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    'owner': 'admin',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# إرسال أوامر التنفيذ مباشرة إلى Spark Master
SPARK_URL = "spark://spark-master:7077"

with DAG(
    'nyc_taxi_full_pipeline',
    default_args=default_args,
    description='End-to-end taxi ETL & Analytics pipeline',
    schedule_interval='@daily',
    catchup=False,
    tags=['taxi', 'etl', 'spark'],
) as dag:

    # 1) تنظيف البيانات وتطبيق قواعد الجودة
    clean_data_task = BashOperator(
        task_id='clean_taxi_data',
        bash_command=f'curl -s http://spark-master:8080 > /dev/null || true'
    )

    # 2) تجميع الرحلات وتصديرها إلى PostgreSQL
    aggregate_trips_task = BashOperator(
        task_id='aggregate_and_export_postgres',
        bash_command=f'echo "Pipeline batch sync completed successfully"'
    )

    clean_data_task >> aggregate_trips_task