from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, unix_timestamp, when, lit

spark = SparkSession.builder \
    .appName("TaxiDataQualityPipeline") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# 1) قراءة الداتا الخام
# ============================================================
df = spark.read.csv("hdfs://namenode:9000/raw/trips/yellow_tripdata_2019-01.csv", header=True, inferSchema=True)
total_raw = df.count()
print(f"📥 عدد الصفوف الخام: {total_raw}")

# ============================================================
# 2) توحيد صيغة التواريخ + اشتقاق trip_duration_seconds
# ============================================================
df = df.withColumn("tpep_pickup_datetime", to_timestamp(col("tpep_pickup_datetime"))) \
       .withColumn("tpep_dropoff_datetime", to_timestamp(col("tpep_dropoff_datetime"))) \
       .withColumn("trip_duration_seconds",
                    unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime"))

# ============================================================
# 3) شيل الصفوف اللي فيها null في الأعمدة الأساسية
# ============================================================
essential_cols = ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                  "passenger_count", "trip_distance", "fare_amount",
                  "PULocationID", "DOLocationID", "RatecodeID"]
df = df.na.drop(subset=essential_cols)
after_null_drop = df.count()
print(f"🧹 بعد شيل الـ nulls الأساسية: {after_null_drop}  (اتشال {total_raw - after_null_drop})")

# ============================================================
# 4) حذف: negative financial values
# ============================================================
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
print(f"💰 بعد شيل القيم المالية السالبة: {after_negative_drop}")

# ============================================================
# 5) حذف: zero fare + zero total (مش رحلة حقيقية)
# ============================================================
df = df.filter(~((col("fare_amount") == 0) & (col("total_amount") == 0)))

df = df.withColumn("flag_zero_fare_nonzero_total",
                    when((col("fare_amount") == 0) & (col("total_amount") > 0), lit(True)).otherwise(lit(False)))

# ============================================================
# 6) حذف: passenger_count = 0 (قاعدة عمل)
# ============================================================
df = df.filter(col("passenger_count") > 0)

# ============================================================
# 7) حذف: RatecodeID = 99 (مش قيمة رسمية موثقة)
# ============================================================
df = df.filter(col("RatecodeID").isin([1, 2, 3, 4, 5, 6]))

# ============================================================
# 8) حذف: extreme outliers في fare_amount, total_amount, trip_distance
# ============================================================
df = df.filter(
    (col("fare_amount") <= 500) &
    (col("total_amount") <= 500) &
    (col("trip_distance") <= 100)
)

# ============================================================
# 9) Cross-Analysis: التعامل مع الحالات المتضاربة
# ============================================================

# Case A: zero duration + positive distance → حذف
df = df.filter(~((col("trip_duration_seconds") == 0) & (col("trip_distance") > 0)))

# Case B: positive duration + zero distance → احتفاظ + flag
df = df.withColumn("flag_positive_duration_zero_distance",
                    when((col("trip_duration_seconds") > 0) & (col("trip_distance") == 0), lit(True)).otherwise(lit(False)))

# Case C: zero duration + zero distance → احتفاظ + flag
df = df.withColumn("flag_zero_duration_zero_distance",
                    when((col("trip_duration_seconds") == 0) & (col("trip_distance") == 0), lit(True)).otherwise(lit(False)))

# حذف: negative duration
df = df.filter(col("trip_duration_seconds") >= 0)

# ============================================================
# 10) الإحصائيات النهائية
# ============================================================
final_count = df.count()
print(f"✅ عدد الصفوف النهائي بعد كل الفلترة: {final_count}")
print(f"📊 إجمالي المحذوف: {total_raw - final_count} ({round((total_raw - final_count) / total_raw * 100, 2)}%)")

flagged_zero_fare = df.filter(col("flag_zero_fare_nonzero_total") == True).count()
flagged_case_b = df.filter(col("flag_positive_duration_zero_distance") == True).count()
flagged_case_c = df.filter(col("flag_zero_duration_zero_distance") == True).count()

print(f"🏷️ صفوف معلّمة (zero fare, total>0): {flagged_zero_fare}")
print(f"🏷️ صفوف معلّمة (Case B - duration>0, distance=0): {flagged_case_b}")
print(f"🏷️ صفوف معلّمة (Case C - duration=0, distance=0): {flagged_case_c}")

# ============================================================
# 11) الحفظ في /processed كـ Parquet
# ============================================================
df.write.mode("overwrite").parquet("hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet")
print("✅ تم الحفظ بنجاح في /processed")

spark.stop()