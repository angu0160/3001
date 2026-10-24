# 03: Baseline Predictive Models

"""
MMA3001 Smart Infrastructure Project
Module 03: Baseline Forecasting Models

This module implements and evaluates two baseline approaches
for predicting indoor temperature one hour ahead.

Input:
    data/temperature_forecasting_data.csv

Outputs:
    results/baseline_validation_metrics.csv
    results/baseline_validation_predictions.csv
    results/baseline_validation_plot.png

Models:
    1. Persistence Forecasting:
       Predicts that the future temperature equals the
       current measured temperature.

    2. Linear Regression:
       Uses temperature, humidity, CO2 concentration,
       hour of day and day of week to predict future temperature.

Main operations:
    - Divide the dataset chronologically into training,
      validation and testing periods.
    - Prevent future-temperature targets from crossing
      relevant dataset boundaries.
    - Train a linear regression model on the training data.
    - Generate persistence and linear regression predictions.
    - Calculate MAE, RMSE and R-squared on validation data.
    - Save model comparisons and visualisations.

The final test dataset is reserved for independent evaluation.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"
results_dir = BASE_DIR / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# 3. Load the forecasting dataset
df = pd.read_csv(
    input_path,
    parse_dates=["timestamp", "matched_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Total forecasting observations:", len(df))

# 4. Define input features and prediction target
features = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "hour",
    "day_of_week"]

target = "temperature_1h_ahead_C"

# 5. Split data chronologically
# 60% training, 20% validation, 20% reserved testing

train_cutoff = df.loc[int(len(df) * 0.60), "timestamp"]
val_cutoff = df.loc[int(len(df) * 0.80), "timestamp"]

# Ensure both input and future target remain
# inside the same data period.
train = df[
    (df["timestamp"] < train_cutoff) &
    (df["matched_timestamp"] < train_cutoff)].copy()

val = df[
    (df["timestamp"] >= train_cutoff) &
    (df["timestamp"] < val_cutoff) &
    (df["matched_timestamp"] < val_cutoff)].copy()

# Keep the final period untouched for later testing
test = df[df["timestamp"] >= val_cutoff].copy()

print("\nTraining observations:", len(train))
print("Validation observations:", len(val))
print("Reserved test observations:", len(test))

# Verify chronological data splitting
print("\n===== Chronological Split Verification =====")

for name, subset in [
    ("Training", train),
    ("Validation", val),
    ("Testing", test)
]:
    print(f"\n{name} dataset:")
    print("Number of observations:", len(subset))
    print("First input:", subset["timestamp"].min())
    print("Last input:", subset["timestamp"].max())
    print("First target:", subset["matched_timestamp"].min())
    print("Last target:", subset["matched_timestamp"].max())

# Verify that measurements are chronologically ordered
assert train["timestamp"].is_monotonic_increasing
assert val["timestamp"].is_monotonic_increasing
assert test["timestamp"].is_monotonic_increasing

# Verify that earlier training targets do not
# overlap with later validation input times
assert train["matched_timestamp"].max() < val["timestamp"].min()

# Verify that validation targets do not
# overlap with test input times
assert val["matched_timestamp"].max() < test["timestamp"].min()

print("\nChronological split verified successfully!")

# 6. Separate inputs and targets
X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]


# 7. Method 1: Persistence forecasting
# Future temperature = current temperature

persistence_pred = val["temperature_C"].to_numpy()


# 8. Method 2: Linear regression
model = LinearRegression()

start = time.perf_counter()
model.fit(X_train, y_train)
fit_time = time.perf_counter() - start

start = time.perf_counter()
linear_pred = model.predict(X_val)
predict_time = time.perf_counter() - start

# 9. Define evaluation function
def evaluate_model(name, predictions):
    """Evaluate forecast accuracy on validation data."""
    return {
        "Model": name,
        "MAE_C": mean_absolute_error(y_val, predictions),
        "RMSE_C": np.sqrt(
            mean_squared_error(y_val, predictions)
        ),
        "R2": r2_score(y_val, predictions)}


# 10. Compare model performance
metrics = pd.DataFrame([
    evaluate_model("Persistence", persistence_pred),
    evaluate_model("Linear Regression", linear_pred)
])

print("\nValidation performance:")
print(metrics.to_string(index=False))

print(f"\nLinear regression fit time: {fit_time:.4f} s")
print(f"Prediction time: {predict_time:.4f} s")

print("\nLinear regression intercept:")
print(model.intercept_)

print("\nLinear regression coefficients:")
print(pd.Series(model.coef_, index=features))

# 11. Save validation results
metrics.to_csv(
    results_dir / "baseline_validation_metrics.csv",
    index=False)

predictions = val[
    ["matched_timestamp", target]].copy()

predictions["persistence_prediction_C"] = persistence_pred
predictions["linear_prediction_C"] = linear_pred

predictions.to_csv(
    results_dir / "baseline_validation_predictions.csv",
    index=False)

# 12. Plot predictions against actual temperature
# Show the first 200 validation observations
plot_data = predictions.head(200)

plt.figure(figsize=(12, 5))

plt.plot(
    plot_data["matched_timestamp"],
    plot_data[target],
    label="Measured Temperature",
    linewidth=1.5)

plt.plot(
    plot_data["matched_timestamp"],
    plot_data["persistence_prediction_C"],
    label="Persistence Forecast",
    alpha=0.8)

plt.plot(
    plot_data["matched_timestamp"],
    plot_data["linear_prediction_C"],
    label="Linear Regression",
    alpha=0.8)

plt.xlabel("Time (UTC)")
plt.ylabel("Temperature (°C)")
plt.title("One-Hour Temperature Forecast: Validation Period")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "baseline_validation_plot.png",
    dpi=300)

plt.show()

print("\nBaseline modelling completed successfully!")