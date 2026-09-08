from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("TripDurationInference") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 1. Load fitted pipeline model from HDFS
model_path = "hdfs://namenode:9000/models/trip_duration_rf_pipeline"
print("Loading pipeline model from HDFS...")
loaded_model = PipelineModel.load(model_path)

# 2. Define sample test inferences (Manhattan peak, Queens late night, Brooklyn morning)
sample_data = [
    (2.1, 1, 17, 3, "Manhattan"),
    (14.5, 2, 23, 6, "Queens"),
    (0.8, 1, 11, 2, "Brooklyn")
]

columns = ["trip_distance", "passenger_count", "pickup_hour", "day_of_week", "pickup_borough"]
df_new = spark.createDataFrame(sample_data, columns)

# 3. Perform batch inference through the loaded pipeline
predictions = loaded_model.transform(df_new)

print("Inference results for test samples:")
predictions.select("trip_distance", "pickup_borough", "pickup_hour", "prediction").show(truncate=False)

spark.stop()