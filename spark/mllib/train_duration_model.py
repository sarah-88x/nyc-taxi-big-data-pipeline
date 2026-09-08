from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, round as sql_round
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

def main():
    # 1) تهيئة جلسة Spark
    spark = SparkSession.builder \
        .appName("TaxiDurationRandomForest") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print(">>> 📥 بدء قراءة البيانات النظيفة وملف المناطق من HDFS...")

    # 2) قراءة البيانات المنقحة وملف المناطق
    processed_path = "hdfs://namenode:9000/processed/yellow_tripdata_2019-01_clean.parquet"
    lookup_path = "hdfs://namenode:9000/raw/taxi_zones/taxi_zone_lookup.csv"

    df = spark.read.parquet(processed_path)
    df_zones = spark.read.option("header", "true").option("inferSchema", "true").csv(lookup_path)

    # 3) فلترة الحالات المعلمة (Outliers / Flags) لتحسين جودة التدريب
    df_filtered = df.filter(
        (col("flag_positive_duration_zero_distance") == False) &
        (col("flag_zero_duration_zero_distance") == False)
    )

    # استخراج الميزات الزمنية: مدة الرحلة بالدقائق، ساعة الركوب، ويوم الأسبوع
    df_features = df_filtered.withColumn("trip_duration_minutes", sql_round(col("trip_duration_seconds") / 60.0, 2)) \
                             .withColumn("pickup_hour", hour(col("tpep_pickup_datetime"))) \
                             .withColumn("day_of_week", dayofweek(col("tpep_pickup_datetime")))

    # ربط الرحلات لجلب اسم الحي (Borough)
    df_model = df_features.join(
        df_zones.select(col("LocationID"), col("Borough").alias("pickup_borough")),
        df_features["PULocationID"] == df_zones["LocationID"],
        how="inner"
    )

    # استبعاد الرحلات الشاذة (أقل من دقيقة أو أطول من 3 ساعات)
    df_model = df_model.filter((col("trip_duration_minutes") >= 1.0) & (col("trip_duration_minutes") < 180.0)) \
                       .select("trip_distance", "passenger_count", "pickup_hour", "day_of_week", "pickup_borough", "trip_duration_minutes") \
                       .na.drop()

    # 4) مراحل الـ Machine Learning Pipeline
    indexer = StringIndexer(inputCol="pickup_borough", outputCol="borough_index", handleInvalid="keep")

    assembler = VectorAssembler(
        inputCols=["trip_distance", "passenger_count", "pickup_hour", "day_of_week", "borough_index"],
        outputCol="features"
    )

    # نموذج Random Forest
    rf = RandomForestRegressor(
        featuresCol="features",
        labelCol="trip_duration_minutes",
        numTrees=25,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(stages=[indexer, assembler, rf])

    # 5) تقسيم البيانات (80% Train, 20% Test)
    print(">>> 🔄 تقسيم البيانات لـ Train و Test...")
    train_data, test_data = df_model.randomSplit([0.8, 0.2], seed=42)

    # 6) تدريب الموديل
    print(">>> 🌲 جارٍ تدريب نموذج Random Forest عبر الـ Cluster...")
    model = pipeline.fit(train_data)

    # 7) التقييم
    print(">>> 📊 تقييم أداء النموذج على بيانات الاختبار...")
    predictions = model.transform(test_data)

    evaluator_rmse = RegressionEvaluator(labelCol="trip_duration_minutes", predictionCol="prediction", metricName="rmse")
    evaluator_r2 = RegressionEvaluator(labelCol="trip_duration_minutes", predictionCol="prediction", metricName="r2")

    rmse = evaluator_rmse.evaluate(predictions)
    r2 = evaluator_r2.evaluate(predictions)

    print("--------------------------------------------------")
    print(f"✅ RMSE (متوسط الخطأ بالدقائق): {round(rmse, 2)} دقيقة")
    print(f"✅ R² Score (معامل التحديد): {round(r2, 4)}")
    print("--------------------------------------------------")

    # 8) حفظ الـ Pipeline كاملاً في HDFS
    model_output_path = "hdfs://namenode:9000/models/trip_duration_rf_pipeline"
    print(f">>> 💾 حفظ النموذج في {model_output_path}...")
    model.write().overwrite().save(model_output_path)
    print("✅ تم حفظ الـ Random Forest Pipeline بنجاح في HDFS!")

    spark.stop()

if __name__ == "__main__":
    main()