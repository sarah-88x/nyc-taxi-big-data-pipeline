from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, to_json, struct, when, lit, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

def main():
    spark = SparkSession.builder \
        .appName("TaxiStreamProcessor") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(">>> 🚀 بدء تشغيل Spark Structured Streaming...")

    # 1) تعريف الـ Schema لرسائل الـ Taxi القادمة من كافكا
    schema = StructType([
        StructField("VendorID", StringType(), True),
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("passenger_count", StringType(), True),
        StructField("trip_distance", StringType(), True),
        StructField("RatecodeID", StringType(), True),
        StructField("PULocationID", StringType(), True),
        StructField("DOLocationID", StringType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", StringType(), True),
        StructField("extra", StringType(), True),
        StructField("mta_tax", StringType(), True),
        StructField("tip_amount", StringType(), True),
        StructField("tolls_amount", StringType(), True),
        StructField("total_amount", StringType(), True)
    ])

    # 2) القراءة المستمرة من Kafka Topic: raw_taxi_trips
    kafka_raw = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("subscribe", "raw_taxi_trips") \
        .option("startingOffsets", "latest") \
        .load()

    # 3) استخراج محتوى الرسائل وتحويل الأنواع
    parsed_stream = kafka_raw.selectExpr("CAST(value AS STRING) as json_payload") \
        .select(from_json(col("json_payload"), schema).alias("data")) \
        .select("data.*") \
        .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
        .withColumn("fare_amount", col("fare_amount").cast(DoubleType())) \
        .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
        .withColumn("PULocationID", col("PULocationID").cast(IntegerType()))

    # 4) فلترة الرحلات الشاذة (Anomaly Rules) لإرسالها كتنبيهات فورية
    # شذوذ الأسعار أو المسافات الصفرية ذات السعر المرتفع
    alerts_stream = parsed_stream.filter(
        (col("fare_amount") > 150.0) | 
        ((col("trip_distance") == 0.0) & (col("fare_amount") > 30.0)) |
        (col("fare_amount") < 2.5)
    ).withColumn("alert_reason", 
        when(col("fare_amount") > 150.0, "High Fare Anomaly")
        .when((col("trip_distance") == 0.0) & (col("fare_amount") > 30.0), "Zero Distance High Cost")
        .otherwise("Sub-minimum Fare")
    )

    # 5) تجهيز تيار التنبيهات لإعادة إرساله إلى Kafka Topic: taxi_alerts
    kafka_alerts_payload = alerts_stream.select(
        to_json(struct(
            col("tpep_pickup_datetime"),
            col("PULocationID"),
            col("trip_distance"),
            col("fare_amount"),
            col("alert_reason")
        )).alias("value")
    )

    query_kafka_alerts = kafka_alerts_payload.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "kafka:9092") \
        .option("topic", "taxi_alerts") \
        .option("checkpointLocation", "/tmp/spark_checkpoint_alerts") \
        .outputMode("append") \
        .start()

    # 6) طباعة عينة من الرحلات المعالجة لحظياً في الـ Console للمراقبة
    query_console = parsed_stream.select(
        "tpep_pickup_datetime", "PULocationID", "trip_distance", "fare_amount", "total_amount"
    ).writeStream \
     .format("console") \
     .outputMode("append") \
     .option("truncate", "false") \
     .start()

    print(">>> 🟢 محرك التدفق يعمل الآن وينتظر تدفق الرسائل...")
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()