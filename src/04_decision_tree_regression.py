# 04: Decision Tree Regression

"""
MMA3001 Smart Infrastructure Project
Module 04: Decision Tree Regression

This module develops and optimises a Decision Tree Regression
model for one-hour-ahead indoor temperature forecasting.

Input:
    data/temperature_forecasting_data.csv

Main outputs:
    results/tree_tuning_results.csv
    results/tree_validation_metrics.csv
    results/tree_feature_importance.csv
    results/decision_tree_validation_plot.png
    results/decision_tree_structure.png

Main operations:
    - Load the prepared forecasting dataset.
    - Apply chronological training and validation splitting.
    - Train DecisionTreeRegressor models using scikit-learn.
    - Evaluate 15 combinations of tree hyperparameters.
    - Compare candidate models using validation MAE.
    - Select the model with the lowest validation MAE.
    - Calculate model accuracy using MAE, RMSE and R-squared.
    - Examine feature importance and visualise tree structure.
    - Save numerical results and figures.

Hyperparameters investigated:
    max_depth = [2, 4, 6, 8, 10]
    min_samples_leaf = [5, 20, 50]

Selected configuration:
    max_depth = 10
    min_samples_leaf = 20
    random_state = 42

The final test dataset is not used during hyperparameter selection.
"""
__all__ = []

# 1. Import libraries
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)

# 2. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"
results_dir = BASE_DIR / "results"

results_dir.mkdir(parents=True, exist_ok=True)

# 3. Load forecasting dataset
df = pd.read_csv(
    input_path,
    parse_dates=["timestamp", "matched_timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Total observations:", len(df))

# 4. Define input features and output
features = [
    "temperature_C",
    "humidity_pct",
    "co2_ppm",
    "hour",
    "day_of_week"]

target = "temperature_1h_ahead_C"

# 5. Split data chronologically
# Use the same 60/20/20 approach as baseline models

train_cutoff = df.loc[int(len(df) * 0.60), "timestamp"]
val_cutoff = df.loc[int(len(df) * 0.80), "timestamp"]

train = df[
    (df["timestamp"] < train_cutoff) &
    (df["matched_timestamp"] < train_cutoff)].copy()

val = df[
    (df["timestamp"] >= train_cutoff) &
    (df["timestamp"] < val_cutoff) &
    (df["matched_timestamp"] < val_cutoff)].copy()

# Reserve test data for final evaluation
test = df[df["timestamp"] >= val_cutoff].copy()

# Check that prediction targets do not cross split boundaries
assert train["matched_timestamp"].max() < val["timestamp"].min()
assert val["matched_timestamp"].max() < test["timestamp"].min()

print("\nTraining observations:", len(train))
print("Validation observations:", len(val))
print("Reserved test observations:", len(test))

# 6. Prepare model inputs
X_train = train[features]
y_train = train[target]

X_val = val[features]
y_val = val[target]

# 7. Train the initial decision tree
initial_tree = DecisionTreeRegressor(
    max_depth=4,
    min_samples_leaf=20,
    random_state=42)

initial_tree.fit(X_train, y_train)

initial_predictions = initial_tree.predict(X_val)

print("\n===== Initial Decision Tree =====")

print(
    "MAE:",
    mean_absolute_error(y_val, initial_predictions))

print(
    "RMSE:",
    np.sqrt(mean_squared_error(y_val, initial_predictions)))

print(
    "R2:",
    r2_score(y_val, initial_predictions))

# 8. Optimise tree parameters
depth_values = [2, 4, 6, 8, 10]
leaf_values = [5, 20, 50]

tuning_results = []

best_model = None
best_mae = float("inf")
best_depth = None
best_leaf = None

for depth in depth_values:
    for leaf in leaf_values:

        tree = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=leaf,
            random_state=42)

        start = time.perf_counter()
        tree.fit(X_train, y_train)
        fit_time = time.perf_counter() - start

        start = time.perf_counter()
        predictions = tree.predict(X_val)
        prediction_time = time.perf_counter() - start

        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_val, predictions))
        r2 = r2_score(y_val, predictions)

        tuning_results.append({
            "max_depth": depth,
            "min_samples_leaf": leaf,
            "MAE_C": mae,
            "RMSE_C": rmse,
            "R2": r2,
            "fit_time_s": fit_time,
            "prediction_time_s": prediction_time})

        # Select the model with the lowest validation MAE
        if mae < best_mae:
            best_mae = mae
            best_model = tree
            best_depth = depth
            best_leaf = leaf

# 9. Display optimisation results
tuning_df = pd.DataFrame(tuning_results)
tuning_df = tuning_df.sort_values("MAE_C")

print("\n===== Decision Tree Optimisation =====")
print(tuning_df.to_string(index=False))

print("\nBest maximum depth:", best_depth)
print("Best minimum leaf size:", best_leaf)


# 10. Evaluate selected tree on validation data
tree_predictions = best_model.predict(X_val)

tree_metrics = {
    "Model": "Decision Tree",
    "MAE_C": mean_absolute_error(
        y_val, tree_predictions
    ),
    "RMSE_C": np.sqrt(
        mean_squared_error(y_val, tree_predictions)
    ),
    "R2": r2_score(y_val, tree_predictions)}

print("\n===== Selected Tree Validation =====")
print(pd.DataFrame([tree_metrics]).to_string(index=False))

# 11. Compare with previous baseline models
baseline_path = results_dir / "baseline_validation_metrics.csv"

if baseline_path.exists():
    baseline_metrics = pd.read_csv(baseline_path)

    comparison = pd.concat([
        baseline_metrics,
        pd.DataFrame([tree_metrics])
    ], ignore_index=True)

    comparison = comparison.sort_values("MAE_C")

    print("\n===== All Model Comparison =====")
    print(comparison.to_string(index=False))

    comparison.to_csv(
        results_dir / "all_model_validation_metrics.csv",
        index=False)
else:
    print("\nBaseline metrics not found.")
    print("Run 03_baseline_models.py to compare all models.")


# 12. Save optimisation results
tuning_df.to_csv(
    results_dir / "tree_tuning_results.csv",
    index=False)

pd.DataFrame([tree_metrics]).to_csv(
    results_dir / "tree_validation_metrics.csv",
    index=False)

# 13. Plot measured vs predicted temperatures
plot_count = min(200, len(val))

plt.figure(figsize=(12, 5))

plt.plot(
    val["matched_timestamp"].iloc[:plot_count],
    y_val.iloc[:plot_count],
    label="Measured Temperature",
    linewidth=1.5)

plt.plot(
    val["matched_timestamp"].iloc[:plot_count],
    tree_predictions[:plot_count],
    label="Decision Tree Prediction",
    alpha=0.8)

plt.xlabel("Time (UTC)")
plt.ylabel("Temperature (°C)")
plt.title("Decision Tree: One-Hour Temperature Forecast")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    results_dir / "decision_tree_validation_plot.png",
    dpi=300)

plt.show()

# 14. Feature importance analysis
importance = pd.DataFrame({
    "Feature": features,
    "Importance": best_model.feature_importances_})

importance = importance.sort_values(
    "Importance", ascending=False)

print("\n===== Feature Importance =====")
print(importance.to_string(index=False))

importance.to_csv(
    results_dir / "tree_feature_importance.csv",
    index=False)

# 15. Visualise the first levels of the decision tree
plt.figure(figsize=(16, 7))

plot_tree(
    best_model,
    feature_names=features,
    filled=True,
    rounded=True,
    max_depth=2,
    fontsize=9)

plt.title("Decision Tree Structure - First Three Levels")
plt.tight_layout()

plt.savefig(
    results_dir / "decision_tree_structure.png",
    dpi=300,
    bbox_inches="tight")

plt.show()

print("\nDecision Tree Regression completed successfully!")