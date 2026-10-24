# 06: Sensitivity and Overfitting Analysis

"""
MMA3001 Smart Infrastructure Project
Module 06: Decision Tree Sensitivity Analysis

This module investigates how decision-tree depth affects
training and validation performance.

Input:
    data/temperature_forecasting_data.csv

Outputs:
    results/sensitivity_analysis.csv
    results/tree_overfitting_analysis.png

Main operations:
    - Load the prepared forecasting dataset.
    - Apply chronological training and validation splitting.
    - Train decision trees using different maximum depths.
    - Keep the minimum leaf size fixed at 20 observations.
    - Calculate training and validation MAE for each model.
    - Calculate validation RMSE.
    - Compare training and validation errors.
    - Investigate potential underfitting and overfitting.
    - Save numerical results and visualisations.

Parameters investigated:
    max_depth = [2, 4, 6, 8, 10]
    min_samples_leaf = 20
    random_state = 42

The analysis examines the trade-off between model complexity
and prediction accuracy.

The test dataset is not used in this sensitivity analysis.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error)


# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"
results_dir = BASE_DIR / "results"
results_dir.mkdir(parents=True, exist_ok=True)


# 3. Load the forecasting dataset
df = pd.read_csv(input_path)

df["timestamp"] = pd.to_datetime(
    df["timestamp"], utc=True)

df["matched_timestamp"] = pd.to_datetime(
    df["matched_timestamp"], utc=True)

df = df.sort_values("timestamp").reset_index(drop=True)


# 4. Define features and target
features = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "hour",
    "day_of_week"]

target = "temperature_1h_ahead_C"


# 5. Use original training and validation splits
train_cutoff = df.loc[
    int(len(df) * 0.60), "timestamp"]

val_cutoff = df.loc[
    int(len(df) * 0.80), "timestamp"]

train = df[
    (df["timestamp"] < train_cutoff) &
    (df["matched_timestamp"] < train_cutoff)].copy()

val = df[
    (df["timestamp"] >= train_cutoff) &
    (df["timestamp"] < val_cutoff) &
    (df["matched_timestamp"] < val_cutoff)].copy()

assert not train.empty
assert not val.empty
assert train["matched_timestamp"].max() < val["timestamp"].min()

X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]


# 6. Investigate decision-tree complexity
depth_values = [2, 4, 6, 8, 10]

# Fix leaf size at the previously selected value
leaf_size = 20

results = []

for depth in depth_values:

    model = DecisionTreeRegressor(
        max_depth=depth,
        min_samples_leaf=leaf_size,
        random_state=42)

    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)

    train_mae = mean_absolute_error(
        y_train, train_predictions)

    val_mae = mean_absolute_error(
        y_val, val_predictions)

    val_rmse = mean_squared_error(
        y_val, val_predictions) ** 0.5

    results.append({
        "Max_Depth": depth,
        "Training_MAE": train_mae,
        "Validation_MAE": val_mae,
        "Validation_RMSE": val_rmse,
        "MAE_Gap": val_mae - train_mae})


# 7. Display and save results
results_df = pd.DataFrame(results)

print("\n===== Sensitivity Analysis =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    results_dir / "sensitivity_analysis.csv",
    index=False)


# 8. Plot training vs validation error
plt.figure(figsize=(9, 5))

plt.plot(
    results_df["Max_Depth"],
    results_df["Training_MAE"],
    marker="o",
    label="Training MAE")

plt.plot(
    results_df["Max_Depth"],
    results_df["Validation_MAE"],
    marker="s",
    label="Validation MAE")

plt.xlabel("Maximum Decision Tree Depth")
plt.ylabel("Mean Absolute Error (°C)")
plt.title("Decision Tree Complexity and Prediction Error")
plt.xticks(depth_values)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "tree_overfitting_analysis.png",
    dpi=300)

plt.show()


# 9. Identify best depth for this leaf size
best_index = results_df["Validation_MAE"].idxmin()
best = results_df.loc[best_index]

print("\n===== Best Depth at Leaf Size 20 =====")

print("Best depth:", int(best["Max_Depth"]))
print(
    "Training MAE:",
    round(best["Training_MAE"], 4))
print(
    "Validation MAE:",
    round(best["Validation_MAE"], 4))
print(
    "Training-validation gap:",
    round(best["MAE_Gap"], 4))

print("\nSensitivity analysis completed successfully!")
