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
    print("Initiating training and persistence for Anomaly Detection pipeline...")

    # 1. Load curated records from HDFS
    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    df = spark.read.parquet(processed_path)
    df_data = df.withColumn("trip_duration_minutes", sql_round(col("trip_duration_seconds") / 60.0, 2)) \
                .select("trip_distance", "fare_amount", "trip_duration_minutes") \
                .na.drop()

    # 2. Build feature engineering and clustering pipeline
    assembler = VectorAssembler(inputCols=["trip_distance", "fare_amount", "trip_duration_minutes"], outputCol="raw_features")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    kmeans = KMeans(k=4, seed=42, featuresCol="features", predictionCol="cluster")
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])

    print("Fitting KMeans pipeline model...")
    model = pipeline.fit(df_data)
    result = model.transform(df_data)

    # 3. Retrieve cluster centroids from the fitted model
    kmeans_model = model.stages[-1]
    centers = kmeans_model.clusterCenters()

    # 4. Compute Euclidean distance from each point to assigned cluster centroid
    def distance_to_center(features, cluster):
        center = centers[cluster]
        return float(np.linalg.norm(np.array(features) - np.array(center)))

    distance_udf = udf(distance_to_center, DoubleType())
    result = result.withColumn("distance_to_center", distance_udf(col("features"), col("cluster")))

    # 5. Define anomaly threshold using the 95th percentile
    threshold = result.approxQuantile("distance_to_center", [0.95], 0.01)[0]
    result = result.withColumn("is_anomaly", col("distance_to_center") > threshold)

    # 6. Evaluation metrics
    total = result.count()
    anomalies = result.filter(col("is_anomaly") == True).count()
    print(f"Total evaluated trips: {total}")
    print(f"Identified anomalies: {anomalies} ({round(anomalies / total * 100, 2)}%)")
    print(f"Distance threshold (95th percentile): {round(threshold, 3)}")

    # 7. Display top outlying instances
    print("Top 10 highest anomaly distance records:")
    result.filter(col("is_anomaly") == True) \
          .select("trip_distance", "fare_amount", "trip_duration_minutes", "cluster", "distance_to_center") \
          .orderBy(col("distance_to_center").desc()) \
          .show(10, truncate=False)

    # 8. Persist trained pipeline model to HDFS
    model_path = "hdfs://namenode:9000/models/anomaly_detection_pipeline"
    model.write().overwrite().save(model_path)

    # 9. Persist flagged anomaly dataset to HDFS
    result.select("trip_distance", "fare_amount", "trip_duration_minutes", "cluster", "distance_to_center", "is_anomaly") \
          .write.mode("overwrite").parquet("hdfs://namenode:9000/curated/taxi_trips_kmeans_anomalies.parquet")

    print(f"Anomaly detection pipeline saved to: {model_path}")
    print("Curated anomaly records exported to: hdfs://namenode:9000/curated/taxi_trips_kmeans_anomalies.parquet")

    spark.stop()

if __name__ == "__main__":
    main()