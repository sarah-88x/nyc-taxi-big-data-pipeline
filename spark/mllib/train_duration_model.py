from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, round as sql_round
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    spark = SparkSession.builder \
        .appName("TaxiDurationRandomForest") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Loading curated dataset and taxi zones lookup from HDFS...")

    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    lookup_path = "hdfs://namenode:9000/raw/taxi_zones/taxi_zone_lookup.csv"

    df = spark.read.parquet(processed_path)
    df_zones = spark.read.option("header", "true").option("inferSchema", "true").csv(lookup_path)

    # Filter anomaly-flagged trips to improve training quality
    df_filtered = df.filter(
        (col("flag_positive_duration_zero_distance") == False) &
        (col("flag_zero_duration_zero_distance") == False)
    )

    # Feature engineering: trip duration (mins), pickup hour, and day of week
    df_features = df_filtered.withColumn("trip_duration_minutes", sql_round(col("trip_duration_seconds") / 60.0, 2)) \
                             .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
                             .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))

    # Join with zones table to attach pickup borough
    df_model = df_features.join(
        df_zones.select(col("LocationID"), col("Borough").alias("pickup_borough")),
        df_features["PULocationID"] == df_zones["LocationID"],
        how="inner"
    )

    # Filter realistic duration boundaries (between 1 minute and 3 hours)
    df_model = df_model.filter((col("trip_duration_minutes") >= 1.0) & (col("trip_duration_minutes") < 180.0)) \
                       .select("trip_distance", "passenger_count", "pickup_hour", "day_of_week", "pickup_borough", "trip_duration_minutes") \
                       .na.drop()

    # Machine Learning Pipeline stages
    indexer = StringIndexer(inputCol="pickup_borough", outputCol="borough_index", handleInvalid="keep")

    assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "pickup_hour", "day_of_week", "borough_index"],
        outputCol="features"
    )

    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="trip_duration_minutes",
        numTrees=25,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    print("Splitting dataset into train (80%) and test (20%) sets...")
    train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

    print("Training Random Forest regression model on Spark cluster...")
    model = pipeline.fit(train_data)

    print("Evaluating model performance on test set...")
    predictions = model.transform(test_data)

    evaluator_rmse = RegressionEvaluator(labelCol="trip_duration_minutes", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="trip_duration_minutes", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print("-" * 50)
    print(f"RMSE (Root Mean Squared Error): {round(rmse, 2)} minutes")
    print(f"R2 Score (Coefficient of Determination): {round(r2, 4)}")
    print("-" * 50)

    model_output_path = "hdfs://namenode:9000/models/trip_duration_rf_pipeline"
    print(f"Saving fitted pipeline to: {model_output_path}")
    model.write().overwrite().save(model_output_path)
    print("Random Forest pipeline persisted successfully to HDFS.")

    spark.stop()

if __name__ == "__main__":
    main()