# 02: Prepare Temperature Forecasting Data

"""
MMA3001 Smart Infrastructure Project
Module 02: Forecasting Data Preparation

This module prepares the cleaned environmental sensor data
for one-hour-ahead indoor temperature forecasting.

Input:
    data/clean_environmental_sensor.csv

Output:
    data/temperature_forecasting_data.csv

Main operations:
    - Load and organise the cleaned sensor observations.
    - Extract hour-of-day and day-of-week features.
    - Generate target timestamps one hour after each observation.
    - Match future temperature measurements to target timestamps
      using a maximum tolerance of two minutes.
    - Remove observations without valid future-temperature matches.
    - Save the processed dataset for model training and evaluation.

The prepared dataset contains five input features and the
target variable temperature_1h_ahead_C.

Assumptions:
    - Timestamps can be converted to valid datetimes.
    - Sensor measurements represent the selected indoor location.
    - Future temperature targets are based on recorded measurements,
      not interpolated values.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import pandas as pd

# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "clean_environmental_sensor.csv"
output_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"

# 3. Load cleaned environmental data
df = pd.read_csv(input_path)

# Parse timestamps in UTC to handle daylight saving correctly
df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)

print("Original observations:", len(df))

# 4. Check sampling intervals
intervals = df["timestamp"].diff().dt.total_seconds() / 60

print("\nSampling intervals (minutes):")
print(intervals.describe())

# 5. Define one-hour prediction times
df["target_timestamp"] = (
    df["timestamp"] + pd.Timedelta(hours=1))

# Prepare actual temperatures for matching
future = df[["timestamp", "temperature_C"]].copy()

future = future.rename(columns={
    "timestamp": "matched_timestamp",
    "temperature_C": "temperature_1h_ahead_C"})

# 6. Match with actual future measurements
forecast = pd.merge_asof(
    df.sort_values("target_timestamp"),
    future.sort_values("matched_timestamp"),
    left_on="target_timestamp",
    right_on="matched_timestamp",
    direction="nearest",
    tolerance=pd.Timedelta(minutes=2))

# Remove rows without a suitable future measurement
forecast = forecast.dropna(
    subset=["temperature_1h_ahead_C"]).copy()

# 7. Create time-based input features
local_time = forecast["timestamp"].dt.tz_convert(
    "Australia/Melbourne")

forecast["hour"] = local_time.dt.hour
forecast["day_of_week"] = local_time.dt.dayofweek

# 8. Select input features and target
columns = [
    "timestamp",
    "target_timestamp",
    "matched_timestamp",
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "hour",
    "day_of_week",
    "temperature_1h_ahead_C"]

forecast = forecast[columns].sort_values("timestamp")

# 9. Save forecasting dataset
forecast.to_csv(output_path, index=False)

# 10. Display results
print("\nForecasting observations:", len(forecast))

print("\nFirst five rows:")
print(forecast.head().to_string(index=False))

print("\nMissing values:")
print(forecast.isna().sum())

print("\nForecasting dataset saved to:")
print(output_path)