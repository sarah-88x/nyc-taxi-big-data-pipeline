import json
from kafka import KafkaConsumer, KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

def main():
    spark = SparkSession.builder \
        .appName("TaxiStreamProcessor") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Initializing Spark Stream Processor...")

    # 1. Consumer for Raw Trips
    consumer = KafkaConsumer(
        'raw_taxi_trips',
        bootstrap_servers=['kafka:29092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='spark-taxi-group',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )

    # Producer for derived events & alerts
    producer = KafkaProducer(
        bootstrap_servers=['kafka:29092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    schema = StructType([
        StructField("VendorID", StringType(), True),
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("passenger_count", StringType(), True),
        StructField("trip_distance", DoubleType(), True),
        StructField("RatecodeID", StringType(), True),
        StructField("PULocationID", IntegerType(), True),
        StructField("DOLocationID", StringType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", DoubleType(), True),
        StructField("extra", StringType(), True),
        StructField("mta_tax", StringType(), True),
        StructField("tip_amount", StringType(), True),
        StructField("tolls_amount", StringType(), True),
        StructField("total_amount", DoubleType(), True)
    ])

    print("Stream engine connected successfully. Awaiting messages...")

    batch = []
    batch_size = 50

    try:
        for message in consumer:
            raw_data = message.value
            
            # Cast primary datatypes
            try:
                trip_distance = float(raw_data.get("trip_distance", 0.0) or 0.0)
                fare_amount = float(raw_data.get("fare_amount", 0.0) or 0.0)
                total_amount = float(raw_data.get("total_amount", 0.0) or 0.0)
                pu_location_id = int(raw_data.get("PULocationID", 0) or 0)
            except (ValueError, TypeError):
                continue

            cleaned_row = {
                "VendorID": str(raw_data.get("VendorID", "")),
                "tpep_pickup_datetime": str(raw_data.get("tpep_pickup_datetime", "")),
                "tpep_dropoff_datetime": str(raw_data.get("tpep_dropoff_datetime", "")),
                "passenger_count": str(raw_data.get("passenger_count", "")),
                "trip_distance": trip_distance,
                "RatecodeID": str(raw_data.get("RatecodeID", "")),
                "PULocationID": pu_location_id,
                "DOLocationID": str(raw_data.get("DOLocationID", "")),
                "payment_type": str(raw_data.get("payment_type", "")),
                "fare_amount": fare_amount,
                "extra": str(raw_data.get("extra", "")),
                "mta_tax": str(raw_data.get("mta_tax", "")),
                "tip_amount": str(raw_data.get("tip_amount", "")),
                "tolls_amount": str(raw_data.get("tolls_amount", "")),
                "total_amount": total_amount
            }

            # 2. Publish cleaned event to 'taxi_trip_events' topic (هذا هو السطر المضاف)
            producer.send("taxi_trip_events", value=cleaned_row)

            # 3. Apply anomaly detection rules and dispatch alerts to 'taxi_alerts'
            alert_reason = None
            if fare_amount > 150.0:
                alert_reason = "High Fare Anomaly"
            elif trip_distance == 0.0 and fare_amount > 30.0:
                alert_reason = "Zero Distance High Cost"
            elif fare_amount < 2.5:
                alert_reason = "Sub-minimum Fare"

            if alert_reason:
                alert_payload = {
                    "tpep_pickup_datetime": cleaned_row["tpep_pickup_datetime"],
                    "PULocationID": pu_location_id,
                    "trip_distance": trip_distance,
                    "fare_amount": fare_amount,
                    "alert_reason": alert_reason
                }
                producer.send("taxi_alerts", value=alert_payload)

            batch.append(cleaned_row)

            # Process and display micro-batch
            if len(batch) >= batch_size:
                df = spark.createDataFrame(batch, schema=schema)
                print("-" * 50)
                print(f"Processing micro-batch ({len(batch)} trips):")
                print("-" * 50)
                df.select("tpep_pickup_datetime", "PULocationID", "trip_distance", "fare_amount", "total_amount").show(10, truncate=False)
                batch = []

    except KeyboardInterrupt:
        print("\nStreaming process halted manually.")
    finally:
        consumer.close()
        producer.close()
        spark.stop()

if __name__ == "__main__":
    main()