from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_date, count, sum as _sum, avg, round

def main():
    spark = SparkSession.builder \
        .appName("TaxiCuratedAggregation") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    lookup_path = "hdfs://namenode:9000/raw/taxi_zones/taxi_zone_lookup.csv"

    print("Reading processed dataset and zones lookup from HDFS...")
    df_trips = spark.read.parquet(processed_path)
    df_zones = spark.read.option("header", "true").option("inferSchema", "true").csv(lookup_path)

    print("Calculating hourly peak statistics...")
    df_hourly = df_trips.withColumn("trip_date", to_date(col("tpep_pickup_datetime"))) \
                        .withColumn("trip_hour", hour(col("tpep_pickup_datetime"))) \
                        .groupBy("trip_date", "trip_hour") \
                        .agg(
                            count("*").alias("total_trips"),
                            round(_sum("total_amount"), 2).alias("total_revenue"),
                            round(avg("trip_distance"), 2).alias("avg_distance")
                        )

    print("Aggregating pickup statistics by zone...")
    df_zone_summary = df_trips.join(
        df_zones,
        df_trips["PULocationID"] == df_zones["LocationID"],
        how="inner"
    ).groupBy("Borough", "Zone") \
     .agg(
         count("*").alias("pickup_count"),
         round(_sum("total_amount"), 2).alias("total_revenue"),
         round(avg("fare_amount"), 2).alias("avg_fare")
     ).orderBy(col("pickup_count").desc())

    print("Writing curated datasets to HDFS...")
    df_hourly.write.mode("overwrite").parquet("hdfs://namenode:9000/curated/hourly_trends")
    df_zone_summary.write.mode("overwrite").parquet("hdfs://namenode:9000/curated/zone_summary")

    postgres_url = "jdbc:postgresql://postgres:5432/taxi_db"
    db_properties = {
        "user": "taxi_admin",
        "password": "taxi_pass123",
        "driver": "org.postgresql.Driver"
    }

    try:
        print("Exporting tables to PostgreSQL...")
        df_hourly.write.jdbc(url=postgres_url, table="daily_hourly_stats", mode="overwrite", properties=db_properties)
        df_zone_summary.write.jdbc(url=postgres_url, table="zone_pickup_stats", mode="overwrite", properties=db_properties)
        print("Data export to PostgreSQL completed successfully.")
    except Exception as e:
        print(f"PostgreSQL export error: {e}")

    print("Batch aggregation process finished.")
    spark.stop()

if __name__ == "__main__":
    main()