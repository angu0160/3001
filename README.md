# MMA3001 — Smart Infrastructure Temperature Forecasting

**Monash University | MMA3001: Numerical Methods and Machine Learning**

## Project Overview

This project develops and evaluates machine-learning models for predicting indoor temperature approximately one hour in advance using environmental sensor measurements from the Monash Smart Infrastructure dataset.

Short-term temperature forecasting can support building environmental monitoring and potentially inform heating, ventilation and air-conditioning (HVAC) management.

Three prediction approaches are investigated:

- **Persistence Forecasting:** Predicts that the future temperature will equal the current temperature.
- **Linear Regression:** Models future temperature using a linear relationship between environmental and time-related features.
- **Decision Tree Regression:** Uses nonlinear decision rules to predict future temperature.

The models are compared through chronological validation and testing, hyperparameter optimisation, sensitivity analysis and robustness analysis.

### Key Results

The Decision Tree Regression model achieved the best overall performance on the held-out test dataset.

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---:|---:|---:|
| Persistence | 0.378 | 0.686 | 0.596 |
| Linear Regression | 0.388 | 0.648 | 0.639 |
| **Decision Tree Regression** | **0.297** | **0.472** | **0.809** |

The selected Decision Tree Regression model reduced test MAE by approximately **21.4% compared with the persistence baseline**.

These results apply to the selected sensor location and evaluation period; performance in other buildings has not been established.

## 1. Getting Started

### Requirements

The project is implemented in Python and uses:

- NumPy — numerical computing
- Pandas — data processing
- Matplotlib — visualisation
- Scikit-learn — regression models and evaluation
- Pytest — automated testing
- pdoc — HTML source-code documentation

Install the dependencies from the project root:

```powershell
py -m pip install -r requirements.txt
```

The commands below use the Windows Python launcher. On systems without `py`, use the appropriate Python interpreter.

### Dataset Setup

The environmental sensor dataset is provided through the MMA3001 teaching materials.

To run the project, create a `data` folder in the repository root if it does not already exist, then place the following original file inside:

```text
data/5EnvSensor_MayToDec2024_180kRows.csv
```

The original and processed datasets are excluded from the public repository as a precaution because their redistribution permissions have not been confirmed.

Users must obtain the dataset through authorised university channels.

### Running the Project

Execute the scripts in the following order from the repository's root directory:

```powershell
py src/01_data_exploration.py
py src/02_prepare_forecasting_data.py
py src/03_baseline_models.py
py src/04_decision_tree_regression.py
py src/05_final_model_evaluation.py
py src/06_sensitivity_analysis.py
py src/07_robustness_analysis.py
```

The scripts generate the cleaned datasets, prediction results, evaluation metrics and figures required for the analysis.

Processed datasets are saved in `data/`, while evaluation results and plots are saved in `results/`.

## 2. Project Structure

```text
MMA3001-Smart-Infrastructure/
│
├── data/                      # Local sensor datasets
│
├── src/
│   ├── 01_data_exploration.py
│   ├── 02_prepare_forecasting_data.py
│   ├── 03_baseline_models.py
│   ├── 04_decision_tree_regression.py
│   ├── 05_final_model_evaluation.py
│   ├── 06_sensitivity_analysis.py
│   └── 07_robustness_analysis.py
│
├── tests/
│   ├── test_data_processing.py
│   └── test_reproducibility.py
│
├── results/                   # Evaluation outputs and figures
├── docs/                      # Generated HTML documentation
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

### Source Code Description

| Script | Function |
|---|---|
| `01_data_exploration.py` | Extracts, cleans and visualises environmental measurements |
| `02_prepare_forecasting_data.py` | Generates one-hour-ahead prediction targets |
| `03_baseline_models.py` | Implements persistence and linear regression models |
| `04_decision_tree_regression.py` | Trains and tunes decision-tree models |
| `05_final_model_evaluation.py` | Evaluates the selected models using unseen test observations |
| `06_sensitivity_analysis.py` | Investigates the influence of tree depth on model performance |
| `07_robustness_analysis.py` | Evaluates performance during different levels of temperature change |

## 3. Dataset and Preprocessing

**Dataset:** Monash Smart Infrastructure — Dataset 2

**Selected location:** Building 60, Room G.38 (Compactus_upper)

**Selected sensor ID:** `6012002000869`

The original environmental sensor records contain timestamped measurements of temperature, relative humidity and carbon dioxide concentration.

Following data cleaning, 13,800 distinct timestamp observations were obtained. Of these, 12,408 observations could be matched with an approximately one-hour-ahead temperature reading.

The preprocessing procedure includes:

1. Extracting measurements from the selected sensor.
2. Parsing the JSON-formatted measurement records.
3. Converting and organising timestamps chronologically.
4. Consolidating repeated readings at identical timestamps.
5. Extracting environmental measurements.
6. Generating hour-of-day and day-of-week features.
7. Matching each observation with a temperature reading approximately one hour later, using a two-minute matching tolerance.
8. Excluding observations without valid future-temperature matches.

### Input Features

| Feature | Description | Unit |
|---|---|---|
| `temperature_C` | Current indoor temperature | °C |
| `humidity_pct` | Relative humidity | % |
| `co2_ppm` | CO₂ concentration | ppm |
| `hour` | Local hour of day | 0–23 |
| `day_of_week` | Day of week | 0–6 |

**Prediction target:** `temperature_1h_ahead_C` — recorded indoor temperature approximately one hour later, in °C.

## 4. Computational Methodology

### Training and Validation

The forecasting observations are divided chronologically into approximately:

- 60% training
- 20% validation
- 20% testing

Chronological splitting is used instead of random splitting to reflect the intended forecasting application.

Observations whose future-temperature targets cross the relevant partition boundaries are excluded where necessary to avoid target leakage.

### Model Selection

The persistence and linear regression models provide benchmark performance.

Decision Tree Regression is optimised using 15 hyperparameter combinations:

- `max_depth`: 2, 4, 6, 8 and 10
- `min_samples_leaf`: 5, 20 and 50

The final configuration, selected using validation MAE, is:

```python
DecisionTreeRegressor(
    max_depth=10,
    min_samples_leaf=20,
    random_state=42
)
```

The selected model is retrained using the development dataset before final evaluation on 2,482 previously withheld test observations.

### Performance Evaluation

Model accuracy is assessed using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

Additional analyses investigate the sensitivity of model performance to tree depth and robustness during different levels of temperature variation.

## 5. Results and Outputs

The `results/` directory contains the generated numerical results and visualisations.

Important output files include:

| File | Description |
|---|---|
| `final_test_metrics.csv` | Final test performance for all three models |
| `final_model_comparison.png` | Comparison of recorded and predicted temperatures |
| `final_mae_comparison.png` | Comparison of model MAE |
| `tree_tuning_results.csv` | Decision-tree hyperparameter search results |
| `sensitivity_analysis.csv` | Model complexity and error analysis |
| `tree_overfitting_analysis.png` | Training and validation error comparison |
| `robustness_analysis.csv` | Performance during smaller and larger temperature changes |
| `robustness_comparison.png` | Visual comparison of robustness results |
| `test_report.txt` | Automated testing results |

### Key Findings

Decision Tree Regression achieved the lowest MAE, RMSE and highest R² on the held-out test dataset.

Sensitivity analysis showed that greater tree depth improved validation MAE over the tested range, although the improvements decreased at higher depths.

Robustness analysis indicated that persistence forecasting performed better during relatively stable conditions, whereas Decision Tree Regression provided better predictions during larger temperature changes.

The detailed interpretation and engineering implications are discussed in the accompanying technical report.

## 6. Automated Testing and Verification

The project uses `pytest` to verify important implementation and numerical properties.

The testing suite includes checks for:

- Dataset completeness and timestamp uniqueness
- Plausible environmental measurement ranges
- One-hour forecast target alignment
- Correct matching of future-temperature measurements
- Chronological dataset partitioning
- Prediction validity and consistency
- Independent recalculation of evaluation metrics
- Reproducibility of the complete computational workflow

### Running the Tests

After generating the project outputs, execute:

```powershell
py -m pytest tests -v
```

The final testing results are documented in:

```text
results/test_report.txt
```

The reproducibility test executes the seven scripts twice and compares the generated numerical outputs.

Passing these tests supports the correctness and consistency of the implementation, but does not independently establish sensor calibration accuracy or performance on other buildings.

## 7. Documentation

HTML source-code documentation is included in the `docs/` directory.

The documentation provides information about the implemented Python modules and their functions.

The accompanying technical report contains detailed explanations of the engineering problem, methodology, validation, optimisation, results, limitations and lessons learned.

## 8. Limitations and Future Work

This project focuses on forecasting temperature at a single indoor sensor location.

Important limitations include:

- Limited demonstrated generalisability to other buildings or rooms
- Missing and irregularly spaced sensor observations
- Potential sensor measurement uncertainty
- Limited information about HVAC operating conditions and external weather
- Reduced prediction accuracy during some abrupt or unusual temperature changes

Possible improvements include incorporating historical temperature features, evaluating additional sensor locations and comparing other machine-learning algorithms.

The current implementation is an offline forecasting study rather than a deployed real-time HVAC control system.

## 9. AI Use Acknowledgement

OpenAI ChatGPT was used extensively to support the development of this project, including programming guidance, code generation and organisation, debugging, testing implementation, interpretation of numerical results and preparation of documentation.

The generated computational outputs were examined using validation comparisons, automated consistency checks and reproducibility testing.

The project's technical report provides a more detailed account of AI contributions, verification activities, limitations and lessons learned.

## 10. Licence and Acknowledgements

### MIT License

The original source code in this repository is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for the complete licence terms.

The Monash Smart Infrastructure dataset is not covered by this MIT License and remains subject to the original data provider's ownership and usage conditions.

### Acknowledgements

- Monash University — MMA3001: Numerical Methods and Machine Learning
- Monash Smart Infrastructure — Environmental Sensor Dataset
- The open-source Python communities supporting NumPy, Pandas, Matplotlib, Scikit-learn and Pytest
