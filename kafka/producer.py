import csv
import json
import time
import os
from kafka import KafkaProducer

def main():
    producer = KafkaProducer(
        bootstrap_servers=['localhost:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    topic_name = 'raw_taxi_trips'
    print(f"Starting trip stream to Kafka topic: {topic_name}")

    csv_file_path = os.path.join("data", "raw", "yellow_tripdata_2019-01.csv")

    if not os.path.exists(csv_file_path):
        csv_file_path = os.path.join("..", "data", "raw", "yellow_tripdata_2019-01.csv")

    print(f"Reading dataset from: {csv_file_path}")

    try:
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            count = 0
            for row in reader:
                producer.send(topic_name, value=row)
                count += 1
                
                if count % 100 == 0:
                    print(f"Dispatched {count} records...")
                    producer.flush()
                
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStreaming process halted manually.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        producer.flush()
        producer.close()
        print("Producer connection closed successfully.")

if __name__ == "__main__":
    main()