# TEST 8: End-to-End Reproducibility

from pathlib import Path
import subprocess
import sys
import os
import pandas as pd


# 1. Define paths
BASE_DIR = Path(__file__).resolve().parent.parent

src_dir = BASE_DIR / "src"
data_dir = BASE_DIR / "data"
results_dir = BASE_DIR / "results"


# 2. List scripts in execution order
scripts = [
    "01_data_exploration.py",
    "02_prepare_forecasting_data.py",
    "03_baseline_models.py",
    "04_decision_tree_regression.py",
    "05_final_model_evaluation.py",
    "06_sensitivity_analysis.py",
    "07_robustness_analysis.py"]


# 3. List numerical outputs to compare
output_files = [
    data_dir / "clean_environmental_sensor.csv",
    data_dir / "temperature_forecasting_data.csv",
    results_dir / "baseline_validation_metrics.csv",
    results_dir / "tree_validation_metrics.csv",
    results_dir / "final_test_metrics.csv",
    results_dir / "final_test_predictions.csv",
    results_dir / "sensitivity_analysis.csv",
    results_dir / "robustness_analysis.csv"]


# 4. Function to run entire project
def run_project():
    """Run all project scripts in the required order."""

    environment = os.environ.copy()

    # Prevent graph windows interrupting automated testing
    environment["MPLBACKEND"] = "Agg"

    for script in scripts:

        script_path = src_dir / script

        print(f"\nRunning {script}...", flush=True)

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=BASE_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180)

        # Stop if a script encounters an error
        assert result.returncode == 0, (
            f"\nScript failed: {script}\n"
            f"STDOUT:\n{result.stdout[-3000:]}\n"
            f"STDERR:\n{result.stderr[-3000:]}")

    print("\nAll scripts executed successfully!", flush=True)


# 5. Function to load numerical results
def load_results():
    """Load generated CSV files for comparison."""

    saved_results = {}

    for file_path in output_files:

        assert file_path.exists(), (
            f"Missing output file: {file_path}")

        saved_results[file_path.name] = pd.read_csv(file_path)

    return saved_results


# 6. Reproducibility test
def test_reproducibility():

    print("\n===== FIRST EXECUTION =====", flush=True)

    run_project()
    first_results = load_results()

    print("\n===== SECOND EXECUTION =====", flush=True)

    run_project()
    second_results = load_results()

    print("\n===== COMPARING RESULTS =====", flush=True)

    for filename in first_results:

        first = first_results[filename]
        second = second_results[filename]

        # Check shapes, values and column names
        # Allow very small floating-point differences
        pd.testing.assert_frame_equal(
            first,
            second,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10)

        print(f"Verified: {filename}", flush=True)

    print(
        "\nReproducibility test passed successfully!",
        flush=True)