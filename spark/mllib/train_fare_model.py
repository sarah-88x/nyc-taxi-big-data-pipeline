from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    spark = SparkSession.builder \
        .appName("TaxiFarePrediction") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("Initiating dataset ingestion for Fare Prediction model...")

    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    df = spark.read.parquet(processed_path)

    # Feature selection and filtering boundaries
    df_features = df.filter((col("fare_amount") >= 2.5) & (col("fare_amount") <= 250.0)) \
                    .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
                    .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime"))) \
                    .select("trip_distance", "passenger_count", "pickup_hour", "day_of_week", 
                            "PULocationID", "DOLocationID", "fare_amount") \
                    .na.drop()

    assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "pickup_hour", "day_of_week", "PULocationID", "DOLocationID"],
        outputCol="features"
    )

    rf = RandomForestRegressor(featuresCol="features", labelCol="fare_amount", numTrees=25, maxDepth=8, seed=42)
    pipeline = Pipeline(stages=[assembler, rf])

    print("Splitting data into train (80%) and test (20%) sets...")
    train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=42)

    print("Training Random Forest Fare Prediction model on Spark cluster...")
    model = pipeline.fit(train_data)

    print("Evaluating model performance on test set...")
    predictions = model.transform(test_data)
    rmse = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="rmse").evaluate(predictions)
    r2 = RegressionEvaluator(labelCol="fare_amount", predictionCol="prediction", metricName="r2").evaluate(predictions)

    print("-" * 50)
    print(f"Fare RMSE: ${round(rmse, 2)}")
    print(f"Fare R2 Score: {round(r2, 4)}")
    print("-" * 50)

    output_path = "hdfs://namenode:9000/models/fare_prediction_rf_pipeline"
    model.write().overwrite().save(output_path)
    print(f"Fare prediction pipeline model persisted successfully to {output_path}")

    spark.stop()

if __name__ == "__main__":
    main()