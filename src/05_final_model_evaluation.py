# 05: Final Model Evaluation

"""
MMA3001 Smart Infrastructure Project
Module 05: Final Model Evaluation

This module evaluates the selected forecasting models using
a previously unseen chronological test dataset.

Input:
    data/temperature_forecasting_data.csv

Outputs:
    results/final_test_metrics.csv
    results/final_test_predictions.csv
    results/final_model_comparison.png
    results/final_mae_comparison.png

Models evaluated:
    - Persistence Forecasting
    - Linear Regression
    - Decision Tree Regression

Main operations:
    - Separate the development and held-out test datasets
      using chronological timestamps.
    - Check that development targets do not overlap
      the test period.
    - Train Linear Regression on development observations.
    - Retrain the selected Decision Tree Regression model
      using the development dataset.
    - Generate predictions for all test observations.
    - Calculate MAE, RMSE and R-squared for each method.
    - Measure model fitting and prediction times.
    - Compare model performance using numerical results
      and visualisations.

The same held-out test observations are used for all models
to ensure a consistent comparison.

The test dataset is not used to select tree hyperparameters.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"
results_dir = BASE_DIR / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# 3. Load forecasting data
df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True)

df["matched_timestamp"] = pd.to_datetime(
    df["matched_timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)

print("Total forecasting observations:", len(df))

# 4. Define input features and output
features = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "hour",
    "day_of_week"]

target = "temperature_1h_ahead_C"

# 5. Create final development and testing sets
# Use the same 80% test cutoff as previous scripts

test_cutoff = df.loc[
    int(len(df) * 0.80), "timestamp"]

# Combine earlier training and validation data.
# Do not include future targets crossing the test boundary.
development = df[
    (df["timestamp"] < test_cutoff) &
    (df["matched_timestamp"] < test_cutoff)].copy()

# Final 20% of observations
test = df[
    df["timestamp"] >= test_cutoff].copy()

print("\n===== Final Dataset Split =====")
print("Development observations:", len(development))
print("Test observations:", len(test))

print("\nDevelopment period:")
print(
    development["timestamp"].min(),
    "to",
    development["timestamp"].max())

print("\nTest period:")
print(
    test["timestamp"].min(),
    "to",
    test["timestamp"].max())

# Verify the chronological split
assert not development.empty
assert not test.empty

assert (
    development["matched_timestamp"].max()
    < test["timestamp"].min())

print("\nChronological split verified!")

# 6. Prepare inputs and targets
X_dev = development[features]
y_dev = development[target]

X_test = test[features]
y_test = test[target]

# 7. Method 1: Persistence forecasting
# One-hour forecast equals current temperature

persistence_predictions = (
    test["temperature_C"].to_numpy())

# 8. Method 2: Linear Regression
linear_model = LinearRegression()

start = time.perf_counter()
linear_model.fit(X_dev, y_dev)
linear_fit_time = time.perf_counter() - start

start = time.perf_counter()
linear_predictions = linear_model.predict(X_test)
linear_prediction_time = time.perf_counter() - start


# 9. Method 3: Optimised Decision Tree
# Settings selected previously from validation data

tree_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_leaf=20,
    random_state=42)

start = time.perf_counter()
tree_model.fit(X_dev, y_dev)
tree_fit_time = time.perf_counter() - start

start = time.perf_counter()
tree_predictions = tree_model.predict(X_test)
tree_prediction_time = time.perf_counter() - start


# 10. Define evaluation function
def evaluate_model(name, predictions):
    """Calculate final test-set forecasting metrics."""

    return {
        "Model": name,
        "MAE_C": mean_absolute_error(
            y_test, predictions),
        "RMSE_C": np.sqrt(
            mean_squared_error(y_test, predictions)),
        "R2": r2_score(
            y_test, predictions)}


# 11. Evaluate all three methods on the same test data
results = pd.DataFrame([
    evaluate_model("Persistence", persistence_predictions),
    evaluate_model("Linear Regression", linear_predictions),
    evaluate_model("Decision Tree", tree_predictions)])

results = results.sort_values("MAE_C")

print("\n===== Final Test Results =====")
print(results.to_string(index=False))


# 12. Calculate improvement over the baselines
persistence_mae = results.loc[
    results["Model"] == "Persistence", "MAE_C"
].iloc[0]

linear_mae = results.loc[
    results["Model"] == "Linear Regression", "MAE_C"
].iloc[0]

tree_mae = results.loc[
    results["Model"] == "Decision Tree", "MAE_C"
].iloc[0]

persistence_improvement = (
    (persistence_mae - tree_mae)
    / persistence_mae) * 100

linear_improvement = (
    (linear_mae - tree_mae)
    / linear_mae) * 100

print("\n===== Decision Tree Improvements =====")

print(
    f"MAE improvement vs persistence: "
    f"{persistence_improvement:.2f}%")

print(
    f"MAE improvement vs linear regression: "
    f"{linear_improvement:.2f}%")

# 13. Compare computational performance
print("\n===== Computational Performance =====")

print(
    f"Linear Regression fit time: "
    f"{linear_fit_time:.6f} s")

print(
    f"Linear Regression prediction time: "
    f"{linear_prediction_time:.6f} s")

print(
    f"Decision Tree fit time: "
    f"{tree_fit_time:.6f} s")

print(
    f"Decision Tree prediction time: "
    f"{tree_prediction_time:.6f} s")


# 14. Save final test metrics
results.to_csv(
    results_dir / "final_test_metrics.csv",
    index=False)


# 15. Save final predictions
predictions = test[
    ["timestamp", "matched_timestamp", target]
].copy()

predictions["persistence_C"] = persistence_predictions
predictions["linear_regression_C"] = linear_predictions
predictions["decision_tree_C"] = tree_predictions

predictions.to_csv(
    results_dir / "final_test_predictions.csv",
    index=False)

# 16. Plot model predictions against actual temperature
plot_data = predictions.head(200)

plt.figure(figsize=(12, 5))

plt.plot(
    plot_data["matched_timestamp"],
    plot_data[target],
    label="Actual Temperature",
    linewidth=2)

plt.plot(
    plot_data["matched_timestamp"],
    plot_data["persistence_C"],
    label="Persistence",
    alpha=0.8)

plt.plot(
    plot_data["matched_timestamp"],
    plot_data["linear_regression_C"],
    label="Linear Regression",
    alpha=0.8)

plt.plot(
    plot_data["matched_timestamp"],
    plot_data["decision_tree_C"],
    label="Decision Tree",
    alpha=0.8)

plt.xlabel("Time (UTC)")
plt.ylabel("Temperature (°C)")
plt.title("Final Test: One-Hour Temperature Predictions")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "final_model_comparison.png",
    dpi=300)

plt.show()

# 17. Plot final MAE comparison
plt.figure(figsize=(8, 5))

plt.bar(
    results["Model"],
    results["MAE_C"])

plt.ylabel("Mean Absolute Error (°C)")
plt.title("Final Test MAE Comparison")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "final_mae_comparison.png",
    dpi=300)

plt.show()

print("\nFinal model evaluation completed successfully!")