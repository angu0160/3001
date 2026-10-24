# 01: Environmental Data Exploration

"""
MMA3001 Smart Infrastructure Project
Module 01: Data Exploration and Cleaning

This module extracts and processes environmental sensor
measurements from the Monash Smart Infrastructure dataset.

Input:
    Raw environmental sensor CSV data.

Outputs:
    Cleaned environmental sensor CSV.
    Temperature visualisation.

Main operations:
    - Filter measurements using the selected sensor ID.
    - Parse JSON measurement records.
    - Convert and organise timestamps.
    - Consolidate repeated timestamps.
    - Examine temperature, humidity and CO2 measurements.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import csv
import json
import pandas as pd
import matplotlib.pyplot as plt


# 2. Define dataset paths
BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "5EnvSensor_MayToDec2024_180kRows.csv"
output_path = BASE_DIR / "data" / "clean_environmental_sensor.csv"

# Select one sensor for our temperature prediction project
sensor_id = "6012002000869"

# 3. Load the original CSV
print("Loading environmental sensor data...")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

# 4. Extract selected sensor measurements
rows = []
wanted = {"Temperature", "Humidity", "Carbon dioxide"}

with file_path.open(encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    for record in reader:
        if record["sensorid"] != sensor_id:
            continue

        measurements = json.loads(record["jsondata"])

        for item in measurements:
            name = item["variable"]["name"]

            if name not in wanted:
                continue

            for reading in item["values"]:
                rows.append({
                    "timestamp_unix": reading["time"],
                    "measurement": name,
                    "value": reading["value"]})

print("Measurements extracted:", len(rows))

# 5. Clean and organise the data
long_df = pd.DataFrame(rows)

if long_df.empty:
    raise ValueError("No matching sensor measurements were found.")

long_df["value"] = pd.to_numeric(
    long_df["value"], errors="coerce")

df = long_df.pivot_table(
    index="timestamp_unix",
    columns="measurement",
    values="value",
    aggfunc="median")

# Convert timestamps to Melbourne local time
df.index = pd.to_datetime(
    df.index, unit="s", utc=True
).tz_convert("Australia/Melbourne")

df = df.sort_index()

df = df.rename(columns={
    "Temperature": "temperature_C",
    "Humidity": "humidity_pct",
    "Carbon dioxide": "co2_ppm"})

df.index.name = "timestamp"

print("\nData cleaning completed!")

# 6. Display summary statistics
print("\nFirst five observations:")
print(df.head())

print("\nNumber of observations:", len(df))
print("Start date:", df.index.min())
print("End date:", df.index.max())

print("\nMissing values:")
print(df.isna().sum())

print("\nSummary statistics:")
print(df.describe())

# 7. Save the cleaned dataset
df.to_csv(output_path)
print("\nCleaned dataset saved to:", output_path)

# 8. Plot indoor temperature against time
plt.figure(figsize=(12, 5))

plt.plot(
    df.index,
    df["temperature_C"],
    linewidth=0.7)

plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Indoor Temperature - Building 60, Room G.38")
plt.grid(True)
plt.tight_layout()

# Save the figure
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

plt.savefig(
    results_dir / "temperature_over_time.png",
    dpi=300)

plt.show()

print("\nData exploration completed successfully!")