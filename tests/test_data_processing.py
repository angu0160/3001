# Automated Testing Using Pytest

from pathlib import Path
import numpy as np
import pandas as pd


# 1. Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent

clean_path = BASE_DIR / "data" / "clean_environmental_sensor.csv"
forecast_path = BASE_DIR / "data" / "temperature_forecasting_data.csv"

results_dir = BASE_DIR / "results"
predictions_path = results_dir / "final_test_predictions.csv"
metrics_path = results_dir / "final_test_metrics.csv"


# Helper function: load data and standardise timestamps
def load_data(path, date_columns):
    """Load a project CSV and convert timestamps to UTC."""
    assert path.exists(), f"Missing required file: {path}"

    df = pd.read_csv(path)

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column], utc=True
        )

    return df


# TEST 1: Check cleaned dataset integrity
def test_cleaned_data_integrity():

    df = load_data(clean_path, ["timestamp"])

    assert len(df) == 13800
    assert df["timestamp"].is_unique
    assert df["timestamp"].is_monotonic_increasing
    assert not df.isna().any().any()


# TEST 2: Check physically plausible sensor readings
def test_sensor_measurements():

    df = load_data(clean_path, ["timestamp"])

    assert df["temperature_C"].between(-20, 60).all()
    assert df["humidity_pct"].between(0, 100).all()
    assert (df["co2_ppm"] >= 0).all()


# TEST 3: Check one-hour forecast timestamp alignment
def test_forecast_timestamp_matching():

    df = load_data(
        forecast_path,
        ["timestamp", "target_timestamp", "matched_timestamp"]
    )

    # Every target time must be one hour after input
    expected_target = (
        df["timestamp"] + pd.Timedelta(hours=1)
    )

    assert (
        df["target_timestamp"] == expected_target
    ).all()

    # Actual matched readings must be within 2 minutes
    matching_error = (
        df["matched_timestamp"] - df["target_timestamp"]
    ).abs()

    assert (
        matching_error <= pd.Timedelta(minutes=2)
    ).all()

    assert (
        df["matched_timestamp"] > df["timestamp"]
    ).all()



# TEST 4: Check future temperature values
def test_future_temperature_values():

    clean = load_data(clean_path, ["timestamp"])

    forecast = load_data(
        forecast_path,
        ["timestamp", "target_timestamp", "matched_timestamp"])

    # Rename columns BEFORE merging
    future_measurements = clean[
        ["timestamp", "temperature_C"]
    ].rename(columns={
        "timestamp": "matched_timestamp",
        "temperature_C": "measured_future_C"})

    # Match future targets with original measurements
    comparison = forecast.merge(
        future_measurements,
        on="matched_timestamp",
        how="left",
        validate="many_to_one")

    # Check that all rows are matched
    assert len(comparison) == len(forecast)

    assert not comparison["measured_future_C"].isna().any()

    # Compare predicted target data with actual future readings
    np.testing.assert_allclose(
        comparison["temperature_1h_ahead_C"],
        comparison["measured_future_C"],
        atol=1e-9,
        rtol=0)


# TEST 5: Verify chronological splitting
def test_chronological_split():

    df = load_data(
        forecast_path,
        ["timestamp", "target_timestamp", "matched_timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    train_cutoff = df.loc[int(len(df) * 0.60), "timestamp"]
    val_cutoff = df.loc[int(len(df) * 0.80), "timestamp"]

    train = df[
        (df["timestamp"] < train_cutoff) &
        (df["matched_timestamp"] < train_cutoff)]

    val = df[
        (df["timestamp"] >= train_cutoff) &
        (df["timestamp"] < val_cutoff) &
        (df["matched_timestamp"] < val_cutoff)]

    test = df[df["timestamp"] >= val_cutoff]

    assert not train.empty
    assert not val.empty
    assert not test.empty

    assert train["matched_timestamp"].max() < val["timestamp"].min()
    assert val["matched_timestamp"].max() < test["timestamp"].min()

    assert len(test) == 2482


# TEST 6: Check final model prediction integrity
def test_final_predictions():

    df = load_data(
        predictions_path,
        ["timestamp", "matched_timestamp"])

    prediction_columns = [
        "persistence_C",
        "linear_regression_C",
        "decision_tree_C"]

    assert len(df) == 2482

    for column in prediction_columns:
        assert not df[column].isna().any()
        assert np.isfinite(df[column].to_numpy()).all()

    # Persistence must equal current measured temperature
    forecast = load_data(forecast_path, ["timestamp"])

    comparison = df.merge(
        forecast[["timestamp", "temperature_C"]],
        on="timestamp",
        how="left",
        validate="one_to_one")

    np.testing.assert_allclose(
        comparison["persistence_C"],
        comparison["temperature_C"],
        atol=1e-9,
        rtol=0)


# TEST 7: Recalculate and verify final evaluation metrics
def test_final_metrics():

    predictions = pd.read_csv(predictions_path)
    metrics = pd.read_csv(metrics_path)

    actual = predictions[
        "temperature_1h_ahead_C"
    ].to_numpy()

    models = {
        "Persistence": "persistence_C",
        "Linear Regression": "linear_regression_C",
        "Decision Tree": "decision_tree_C"}

    for model_name, column in models.items():

        predicted = predictions[column].to_numpy()

        errors = actual - predicted

        # Calculate metrics independently
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))

        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual - actual.mean()) ** 2)

        r2 = 1 - ss_res / ss_tot

        # Retrieve saved model results
        saved = metrics.loc[
            metrics["Model"] == model_name]

        assert len(saved) == 1

        saved = saved.iloc[0]

        assert np.isclose(
            mae, saved["MAE_C"], atol=1e-6)

        assert np.isclose(
            rmse, saved["RMSE_C"], atol=1e-6)

        assert np.isclose(
            r2, saved["R2"], atol=1e-6)