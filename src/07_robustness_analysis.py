# 07: Robustness Analysis

"""
MMA3001 Smart Infrastructure Project
Module 07: Forecasting Robustness Analysis

This module investigates the robustness of the forecasting
models under different magnitudes of temperature change.

Input:
    results/final_test_predictions.csv

Outputs:
    results/robustness_analysis.csv
    results/robustness_comparison.png

Models analysed:
    - Persistence Forecasting
    - Linear Regression
    - Decision Tree Regression

Main operations:
    - Load final test predictions and recorded temperatures.
    - Calculate the absolute one-hour temperature change.
    - Determine the 75th percentile temperature-change threshold.
    - Separate observations into smaller-change and
      larger-change groups.
    - Calculate MAE for each model within each group.
    - Compare model accuracy across operating conditions.
    - Save numerical results and comparison plots.

The robustness groups are determined retrospectively
using actual future temperatures.

They are used only to analyse model performance and
are not available for selecting models during live forecasting.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

results_dir = BASE_DIR / "results"
input_path = results_dir / "final_test_predictions.csv"

# 3. Load final test predictions
df = pd.read_csv(input_path)

print("Test observations:", len(df))

# 4. Define actual temperature and predictions
actual = "temperature_1h_ahead_C"

models = {
    "Persistence": "persistence_C",
    "Linear Regression": "linear_regression_C",
    "Decision Tree": "decision_tree_C"}

# 5. Calculate actual one-hour temperature changes
# Persistence equals the current measured temperature.
df["absolute_change_C"] = abs(
    df[actual] - df["persistence_C"])

# 6. Find the 75th percentile of temperature changes
threshold = df["absolute_change_C"].quantile(0.75)

print("\n75th percentile of temperature change:")
print(f"{threshold:.3f} °C")

# 7. Group the observations
df["change_group"] = np.where(
    df["absolute_change_C"] >= threshold,
    "Larger changes",
    "Smaller changes")

print("\nObservations in each group:")
print(df["change_group"].value_counts())

# 8. Evaluate each model in each group
results = []

for group_name, subset in df.groupby("change_group"):

    for model_name, prediction_column in models.items():

        mae = mean_absolute_error(
            subset[actual],
            subset[prediction_column])

        results.append({
            "Change_Group": group_name,
            "Model": model_name,
            "Observations": len(subset),
            "MAE_C": mae})

# 9. Display and save results
results_df = pd.DataFrame(results)

print("\n===== Robustness Analysis =====")
print(results_df.to_string(index=False))

results_df.to_csv(
    results_dir / "robustness_analysis.csv",
    index=False)

# 10. Plot model comparison
plot_data = results_df.pivot(
    index="Change_Group",
    columns="Model",
    values="MAE_C")

plot_data = plot_data.reindex([
    "Smaller changes",
    "Larger changes"])

plot_data.plot(
    kind="bar",
    figsize=(10, 5))

plt.ylabel("Mean Absolute Error (°C)")
plt.xlabel("Actual One-Hour Temperature Change")
plt.title("Forecast Accuracy Under Different Conditions")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    results_dir / "robustness_comparison.png",
    dpi=300)

plt.show()

print("\nRobustness analysis completed successfully!")