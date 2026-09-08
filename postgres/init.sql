-- Initialization script for NYC Taxi Serving Layer

CREATE TABLE IF NOT EXISTS daily_hourly_stats (
    trip_date DATE,
    trip_hour INT,
    total_trips BIGINT,
    total_revenue DOUBLE PRECISION,
    avg_distance DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS zone_pickup_stats (
    "PULocationID" INT,
    pickup_count BIGINT,
    total_revenue DOUBLE PRECISION,
    avg_fare DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_hourly_date_hour ON daily_hourly_stats (trip_date, trip_hour);
CREATE INDEX IF NOT EXISTS idx_zone_pulocation ON zone_pickup_stats ("PULocationID");