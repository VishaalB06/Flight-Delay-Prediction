# D2 — Flight Delay Prediction

**Course:** Foundations of Machine Learning  
**Due:** 10 June  
**Notebook:** `D2_Flight_Delay_Prediction.ipynb`

---

## Overview

This project builds and evaluates machine learning models to predict US domestic flight arrival delays using 2023 flight operations data merged with airport weather observations. Both **binary classification** (delayed vs. on-time, threshold > 15 min) and **regression** (exact arrival delay in minutes) tasks are addressed.

Five model families are benchmarked across three data regimes (full dataset, random 50% coreset, stratified coreset), with dimensionality reduction via PCA and hyperparameter tuning for the MLP.

---

## Project Structure

```
.
├── D2_Flight_Delay_Prediction.ipynb   # Main notebook (all code, analysis, plots)
├── US_flights_2023.csv                # Raw flight operations dataset (500 k rows used)
├── weather_meteo_by_airport.csv       # Daily weather observations keyed by airport & date
├── README.md                          # This file
```

---

## Datasets

| File | Description |
|------|-------------|
| `US_flights_2023.csv` | Flight records: airline, departure airport, scheduled/actual times, distance type, arrival delay, etc. Up to 500,000 rows are loaded (`NROWS = 500_000`). |
| `weather_meteo_by_airport.csv` | Daily meteorological data per airport: temperature, precipitation, wind speed, snowfall, pressure. Merged on `(Dep_Airport, FlightDate)`. |

Both files must be present in the working directory before running the notebook.

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib
```

Python 3.9+ is recommended. The notebook installs dependencies automatically via the first cell.

---

## Features Used

| Feature | Source | Description |
|---------|--------|-------------|
| `Day_Of_Week` | Flight | Day of the week (1–7) |
| `AIRLINE_ENC` | Flight | Label-encoded airline carrier |
| `AIRPORT_ENC` | Flight | Label-encoded departure airport |
| `DEPTIME_ENC` | Flight | Label-encoded departure time bucket |
| `DIST_ENC` | Flight | Label-encoded distance category |
| `Flight_Duration` | Flight | Scheduled flight duration (min) |
| `Dep_Delay` | Flight | Departure delay (min) |
| `W_TEMP` | Weather | Average daily temperature |
| `W_PRCP` | Weather | Daily precipitation |
| `W_WIND` | Weather | Average wind speed |
| `W_SNOW` | Weather | Daily snowfall |
| `W_PRES` | Weather | Atmospheric pressure |

Missing weather values are imputed with the column median.

**Target (classification):** `IS_DELAYED` = 1 if `Arr_Delay > 15` minutes, else 0.  
**Target (regression):** `Arr_Delay` (continuous, minutes).

---

## Models

| Model | Task | Notes |
|-------|------|-------|
| Logistic Regression | Classification | `max_iter=500`, `class_weight=balanced` |
| SVC / SVR | Both | RBF kernel, `C=1.0`; trained on 5,000-row sub-sample for speed |
| Random Forest | Both | 100 trees, `max_depth=12`, `class_weight=balanced` |
| MLP | Both | Architecture (128→64), `lr=0.001`, Adam, early stopping |
| MLP + PCA | Classification | Same MLP on PCA-reduced input (95% variance retained) |

---

## Data Regimes (Coreset Selection)

| Regime | Description | Approx. Size |
|--------|-------------|-------------|
| Full (100%) | Entire pre-processed dataset | ~500 k rows |
| Random 50% | Random sample, 50% of full | ~250 k rows |
| Stratified (45% + 5%) | 45% non-delayed + 5% delayed rows | ~250 k rows |

---

## Notebook Sections

| Section | Content |
|---------|---------|
| 1 | Imports & setup |
| 2 | Load datasets |
| 3 | Preprocessing & feature engineering |
| 4 | Coreset selection (random & stratified) |
| 5 | Model training across all regimes |
| 6 | Performance comparison table (classification + regression) |
| 7 | Coreset analysis plots (AUC & RMSE by regime) |
| 8 | PCA dimensionality reduction analysis |
| 9 | Random Forest feature importances |
| 10 | MLP training loss curve |
| 11 | Hyperparameter tuning (learning rate & architecture grid) |
| 12 | Final results summary |
| 13 | D3 roadmap |

---

## Evaluation Metrics

**Classification:** Accuracy, Precision, Recall, F1-score, AUC-ROC  
**Regression:** RMSE (minutes), MAE (minutes), R²

---

## Reproducibility

All random operations use `SEED = 42`. Train/test split is 80/20, stratified by `IS_DELAYED`. Results are fully deterministic given the same input data.

---

## Key Findings

- **Random Forest** achieves the highest AUC-ROC on the full dataset.
- **MLP (128→64)** with `lr=0.001` is the best-tuned neural architecture.
- **MLP + PCA** retains competitive accuracy with fewer input dimensions.
- **Coreset sampling** preserves most predictive performance at 50% data size.
- **Departure delay (`Dep_Delay`)** is the dominant feature by Gini importance.
- Weather features contribute measurably but are secondary to operational features.

---

## Next Steps (D3)

- Compare results against two published flight-delay prediction studies.
- Implement one targeted improvement (e.g. better coreset strategy or stacking ensemble).
- Add error bars via multiple random seeds to quantify result variance.

---

## License

For academic use only (course assignment).
