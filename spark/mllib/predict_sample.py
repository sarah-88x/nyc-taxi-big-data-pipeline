from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
    .appName("TripDurationInference") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# 1) تحميل الـ Pipeline المحفوظ بالكامل من HDFS
model_path = "hdfs://namenode:9000/models/trip_duration_rf_pipeline"
print(">>> تحميل النموذج من HDFS...")
loaded_model = PipelineModel.load(model_path)

# 2) محاكاة رحلات تجريبية جديدة (رحلة قصيرة في مانهاتن vs رحلة طويلة من المطار)
sample_data = [
    # trip_distance, passenger_count, pickup_hour, day_of_week, pickup_borough
    (2.1, 1, 17, 3, "Manhattan"),   # رحلة 2 ميل ساعة الذروة في مانهاتن
    (14.5, 2, 23, 6, "Queens"),      # رحلة 14.5 ميل منتصف الليل في كوينز
    (0.8, 1, 11, 2, "Brooklyn")      # رحلة قصيرة صباحاً في بروكلين
]

columns = ["trip_distance", "passenger_count", "pickup_hour", "day_of_week", "pickup_borough"]
df_new = spark.createDataFrame(sample_data, columns)

# 3) التوقع المباشر عبر الـ Pipeline
predictions = loaded_model.transform(df_new)

print(">>> نتائج التوقع للرحلات التجريبية:")
predictions.select("trip_distance", "pickup_borough", "pickup_hour", "prediction").show(truncate=False)

spark.stop()