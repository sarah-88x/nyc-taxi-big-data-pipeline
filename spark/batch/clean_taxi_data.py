import builtins
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, when, lit

spark = SparkSession.builder \
    .appName("TaxiDataQualityPipeline") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 1. Ingest raw dataset from HDFS (All CSV files)
raw_data_path = "hdfs://namenode:9000/raw/trips/yellow_tripdata_*.csv"
df = spark.read.csv(raw_data_path, header=True, inferSchema=True)
total_raw = df.count()
print(f"Total raw records loaded: {total_raw}")

# 2. Standardize timestamps and compute trip duration
df = df.withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"))) \
       .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"))) \
       .withColumn("trip_duration_seconds",
                   unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))

# 3. Drop nulls in essential columns
essential_cols = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "fare_amount",
    "PULocationID", "DOLocationID", "RatecodeID"
]
df = df.na.drop(subset=essential_cols)
after_null_drop = df.count()
print(f"Records after dropping nulls: {after_null_drop} (Removed: {total_raw - after_null_drop})")

# 4. Remove negative financial values
df = df.filter(
    (col("fare_amount") >= 0) &
    (col("extra") >= 0) &
    (col("mta_tax") >= 0) &
    (col("tip_amount") >= 0) &
    (col("tolls_amount") >= 0) &
    (col("improvement_surcharge") >= 0) &
    (col("total_amount") >= 0)
)
after_negative_drop = df.count()
print(f"Records after removing negative amounts: {after_negative_drop}")

# 5. Filter zero fare and zero total anomalies, flag non-zero totals with zero fare
df = df.filter(~((col("fare_amount") == 0) & (col("total_amount") == 0)))
df = df.withColumn(
    "flag_zero_fare_nonzero_total",
    when((col("fare_amount") == 0) & (col("total_amount") > 0), lit(True)).otherwise(lit(False))
)

# 6. Apply business rules (valid passenger counts and RatecodeIDs)
df = df.filter(col("passenger_count") > 0)
df = df.filter(col("RatecodeID").isin([1, 2, 3, 4, 5, 6]))

# 7. Remove extreme statistical outliers
df = df.filter(
    (col("fare_amount") <= 500) &
    (col("total_amount") <= 500) &
    (col("trip_distance") <= 100)
)

# 8. Cross-analysis filtering and flagging
# Drop inconsistent records: zero duration with positive distance
df = df.filter(~((col("trip_duration_seconds") == 0) & (col("trip_distance") > 0)))

# Flag positive duration with zero distance
df = df.withColumn(
    "flag_positive_duration_zero_distance",
    when((col("trip_duration_seconds") > 0) & (col("trip_distance") == 0), lit(True)).otherwise(lit(False))
)

# Flag zero duration with zero distance
df = df.withColumn(
    "flag_zero_duration_zero_distance",
    when((col("trip_duration_seconds") == 0) & (col("trip_distance") == 0), lit(True)).otherwise(lit(False))
)

# Filter out invalid negative durations
df = df.filter(col("trip_duration_seconds") >= 0)

# 9. Summary metrics computation
final_count = df.count()
dropped_count = total_raw - final_count
dropped_percentage = builtins.round((dropped_count / total_raw) * 100, 2)

print(f"Final curated record count: {final_count}")
print(f"Total dropped records: {dropped_count} ({dropped_percentage}%)")

flagged_zero_fare = df.filter(col("flag_zero_fare_nonzero_total") == True).count()
flagged_case_b = df.filter(col("flag_positive_duration_zero_distance") == True).count()
flagged_case_c = df.filter(col("flag_zero_duration_zero_distance") == True).count()

print(f"Flagged (zero fare, positive total): {flagged_zero_fare}")
print(f"Flagged (positive duration, zero distance): {flagged_case_b}")
print(f"Flagged (zero duration, zero distance): {flagged_case_c}")

# 10. Persist curated data to HDFS in Parquet format
output_processed_path = "hdfs://namenode:9000/processed/yellow_trips_clean.parquet"
df.write.mode("overwrite").parquet(output_processed_path)
print("Curated dataset persisted successfully to HDFS.")

spark.stop()