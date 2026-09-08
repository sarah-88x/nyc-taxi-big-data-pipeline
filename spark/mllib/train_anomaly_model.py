from pyspark.sql import SparkSession
from pyspark.sql.functions import col, round as sql_round, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
import numpy as np

def main():
    spark = SparkSession.builder \
        .appName("TaxiAnomalyModel") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print(">>> 📥 بدء تدريب وحفظ نموذج Anomaly Detection في /models...")

    # 1) قراءة البيانات المنقحة من HDFS
    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    df = spark.read.parquet(processed_path)
    df_data = df.withColumn("trip_duration_minutes", sql_round(col("trip_duration_seconds") / 60.0, 2)) \
                .select("trip_distance", "fare_amount", "trip_duration_minutes") \
                .na.drop()

    # 2) بناء الـ Pipeline (Assembler + Scaler + KMeans)
    assembler = VectorAssembler(inputCols=["trip_distance", "fare_amount", "trip_duration_minutes"], outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="cluster")
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])

    print(">>> 🧠 جارٍ تدريب الـ Pipeline...")
    model = pipeline.fit(df_data)
    result = model.transform(df_data)

    # 3) جلب مراكز الـ clusters من الموديل المدرب
    kmeans_model = model.stages[-1]
    centers = kmeans_model.clusterCenters()

    # 4) حساب المسافة الإقليدية من كل نقطة لمركز الـ cluster بتاعها
    def distance_to_center(features, cluster):
        center = centers[cluster]
        return float(np.linalg.norm(np.array(features) - np.array(center)))

    distance_udf = udf(distance_to_center, DoubleType())
    result = result.withColumn("distance_to_center", distance_udf(col("features"), col("cluster")))

    # 5) تحديد threshold: أي نقطة بعيدة عن المركز بأكتر من الـ 95th percentile تتعتبر anomaly
    threshold = result.approxQuantile("distance_to_center", [0.95], 0.01)[0]
    result = result.withColumn("is_anomaly", col("distance_to_center") > threshold)

    # 6) الإحصائيات
    total = result.count()
    anomalies = result.filter(col("is_anomaly") == True).count()
    print(f">>> 📈 إجمالي الرحلات: {total}")
    print(f">>> 🚨 عدد الرحلات الشاذة (Anomalies): {anomalies} ({round(anomalies/total*100, 2)}%)")
    print(f">>> 📏 Threshold المسافة: {round(threshold, 3)}")

    # 7) عينة من أبعد الرحلات (الأكثر شذوذاً)
    print(">>> 🔍 عينة من الرحلات الشاذة (أبعد 10 عن مراكز الـ clusters):")
    result.filter(col("is_anomaly") == True) \
          .select("trip_distance", "fare_amount", "trip_duration_minutes", "cluster", "distance_to_center") \
          .orderBy(col("distance_to_center").desc()) \
          .show(10, truncate=False)

    # 8) الحفظ - الموديل نفسه
    model_path = "hdfs://namenode:9000/models/anomaly_detection_pipeline"
    model.write().overwrite().save(model_path)

    # 9) الحفظ - نتائج الـ anomalies (الداتا كاملة مع علامة is_anomaly)
    result.select("trip_distance", "fare_amount", "trip_duration_minutes", "cluster", "distance_to_center", "is_anomaly") \
          .write.mode("overwrite").parquet("hdfs://namenode:9000/curated/taxi_trips_kmeans_anomalies.parquet")

    print(f"✅ تم حفظ نموذج اكتشاف الشذوذ بنجاح في {model_path}")
    print("✅ تم حفظ نتائج الـ anomalies في /curated/taxi_trips_kmeans_anomalies.parquet")

    spark.stop()

if __name__ == "__main__":
    main()