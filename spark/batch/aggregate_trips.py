from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, to_date, count, sum as _sum, avg, round

def main():
    # 1. تهيئة جلسة Spark
    spark = SparkSession.builder \
        .appName("TaxiCuratedAggregation") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(">>> بدء قراءة البيانات النظيفة من HDFS...")

    # قراءة البيانات النظيفة وملف المناطق
    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    lookup_path = "hdfs://namenode:9000/raw/taxi_zones/taxi_zone_lookup.csv"

    df_trips = spark.read.parquet(processed_path)
    df_zones = spark.read.option("header", "true").option("inferSchema", "true").csv(lookup_path)

    # 2. تجميع ساعات الذروة (Peak Hours)
    print(">>> حساب إحصائيات ساعات الذروة...")
    df_hourly = df_trips.withColumn("trip_date", to_date(col("tpep_pickup_datetime"))) \
                        .withColumn("trip_hour", hour(col("tpep_pickup_datetime"))) \
                        .groupBy("trip_date", "trip_hour") \
                        .agg(
                            count("*").alias("total_trips"),
                            round(_sum("total_amount"), 2).alias("total_revenue"),
                            round(avg("trip_distance"), 2).alias("avg_distance")
                        )

    # 3. تجميع الرحلات حسب المناطق بعد الربط (Zone Aggregates)
    print(">>> ربط الرحلات بالمناطق وحساب الأحجام...")
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

    # 4. الحفظ في مسار curated في HDFS
    print(">>> حفظ النتائج في /curated داخل HDFS...")
    df_hourly.write.mode("overwrite").parquet("hdfs://namenode:9000/curated/hourly_trends")
    df_zone_summary.write.mode("overwrite").parquet("hdfs://namenode:9000/curated/zone_summary")

    # 5. التصدير إلى PostgreSQL (Serving Layer)
    postgres_url = "jdbc:postgresql://postgres:5432/taxi_db"
    properties = {
        "user": "taxi_admin",
        "password": "taxi_pass123",
        "driver": "org.postgresql.Driver"
    }

    try:
        print(">>> تصدير الجداول إلى PostgreSQL...")
        df_hourly.write.jdbc(url=postgres_url, table="daily_hourly_stats", mode="overwrite", properties=properties)
        df_zone_summary.write.jdbc(url=postgres_url, table="zone_pickup_stats", mode="overwrite", properties=properties)
        print("✅ تم التصدير إلى PostgreSQL بنجاح!")
    except Exception as e:
        print(f"⚠️ خطأ أثناء التصدير إلى Postgres: {e}")

    print("✅ تم إنهاء التجميع بنجاح!")
    spark.stop()

if __name__ == "__main__":
    main()