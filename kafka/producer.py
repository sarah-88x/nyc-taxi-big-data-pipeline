import csv
import json
import time
import os
from kafka import KafkaProducer

def main():
    # 1) الاتصال بـ Kafka Broker
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    topic_name = 'raw_taxi_trips'
    print(f">>> 🚀 بدء بث الرحلات اللحظية إلى Kafka Topic: {topic_name}")

    # 2) مسار ملف البيانات الصحيح
    csv_file_path = os.path.join("data", "raw", "yellow_tripdata_2019-01.csv")

    if not os.path.exists(csv_file_path):
        # مسار بديل في حال تشغيل الملف من داخل مجلد kafka مباشرة
        csv_file_path = os.path.join("..", "data", "raw", "yellow_tripdata_2019-01.csv")

    print(f">>> 📂 قراءة الملف من: {csv_file_path}")

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            count = 0
            for row in reader:
                producer.send(topic_name, value=row)
                count += 1
                
                # طباعة تقرير كل 100 رحلة
                if count % 100 == 0:
                    print(f"📡 تم إرسال {count} رحلة...")
                    producer.flush()
                
                # تأخير بسيط لمحاكاة البث اللحظي
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n⏹️ تم إيقاف البث يدوياً.")
    except Exception as e:
        print(f"❌ خطأ: {e}")
    finally:
        producer.flush()
        producer.close()
        print("✅ تم إغلاق الـ Producer بنجاح.")

if __name__ == "__main__":
    main()