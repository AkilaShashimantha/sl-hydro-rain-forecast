# 💧⚡ Sri Lanka Rain & Energy Forecasting

Forecast Sri Lanka’s monthly rainfall, hydroelectric output, and total generation using a clean ML pipeline and an attractive Streamlit app. Includes training notebooks, CLI tools, scenario analysis, and model loading.

## Highlights
- Clean daily → monthly data pipeline with robust cleaning and aggregation
- Feature engineering for time series (lags, rolling stats, seasonal encodings)
- Models: Gradient Boosting (default), Random Forest, Linear Regression
- Streamlit app with:
  - Model source: Train now or Load trained model (.joblib from `artifacts/`)
  - Scenario controls: multipliers + manual overrides for rainfall/temperature/hydro
  - KPIs, line chart (actual vs forecast), full forecast table, feature importances
  - Dark theme + custom CSS
- CLI training: forecast from a specific month (uses actuals up to previous month)
- Jupyter notebook for experiments

## Repository structure
```
Energy Prediction App/
├─ app.py                      # Streamlit app entrypoint
├─ requirements.txt           # Dependencies for local & Streamlit Cloud
├─ environment.yml            # Conda environment (optional)
├─ .gitignore
├─ .streamlit/
│  └─ config.toml             # App theme/config
├─ assets/
│  └─ styles.css              # Custom CSS for a polished UI
├─ data/
│  └─ sri_lanka_weather_energy_daily.csv  # Daily dataset
├─ src/
│  └─ pipeline.py             # Cleaning, features, modeling, forecast utilities
├─ notebooks/
│  └─ 01_model_training.ipynb # Training & evaluation notebook
├─ scripts/
│  └─ train_model.py          # CLI training + forecast export
└─ artifacts/                 # Saved models & forecasts (gitignored)
```

## Quick start
1) Create env and install deps
```bash
pip install -r requirements.txt
```
2) Run the app
```bash
streamlit run app.py
```
3) Open the notebook (optional)
```bash
jupyter notebook notebooks/01_model_training.ipynb
```

## Using the Streamlit app
- Data source: By default uses the CSV in [`data/`](file:///d:/Energy%20Prediction%20App/data). You can upload another CSV from the sidebar.
- Target: Choose one of
  - Monthly Rainfall (mm)
  - Hydroelectric Production (GWh)
  - Total Generation (GWh)
- Model source:
  - Train now: trains on current data; shows CV MAE/RMSE
  - Load trained model: pick a `.joblib` in [`artifacts/`](file:///d:/Energy%20Prediction%20App/artifacts). If the selection doesn’t match the chosen target or features, the app shows a friendly message and falls back to training.
- Scenario: Adjust rainfall/temperature/hydro with multipliers; optionally set constant manual values across the horizon.
- Results shown:
  - KPIs: Training months, Avg absolute error (CV MAE), Root mean squared error (CV RMSE), Last actual, Next forecast (+delta), Avg forecast (horizon)
  - Chart: Actual vs forecast line
  - Forecast table: Future months with the target and key exogenous/production columns
  - Top features (if available)

Main app file: [`app.py`](file:///d:/Energy%20Prediction%20App/app.py)

## Data pipeline
- Input: Daily CSV with columns like Year, Month, Day, rainfall, temperature, humidity, wind, precipitation hours, daylight, and production (hydro/solar/coal/fuel).
- Cleaning: Coerce numerics (e.g., strip units like "-2.0 GWh"), clip impossible negatives for rainfall/production, deduplicate dates.
- Aggregation: Resample to month-start (MS): sum rainfall/precip/hydro/solar/coal/fuel; mean temperature/humidity/wind/daylight; compute `Total_Generation_GWh`.
- Feature engineering: time-based features, lags (1/3/6/12), rolling mean/std, exogenous lags.

Core pipeline utilities: [`pipeline.py`](file:///d:/Energy%20Prediction%20App/src/pipeline.py)

## Training via CLI
Use the CLI to train and optionally forecast from a specific start month (actuals are used up to the prior month).

Examples (PowerShell on Windows):
```powershell
cd "D:\Energy Prediction App"
$env:PYTHONPATH="D:\Energy Prediction App"

# Rainfall: start Aug 2025, 6 months
python scripts/train_model.py --target "Average_Rainfall_mm" --start 2025-08-01 --months 6

# Hydro: start Aug 2025, 6 months
python scripts/train_model.py --target "Hydroelectric_Production_GWh" --start 2025-08-01 --months 6

# Total: start Aug 2025, 6 months
python scripts/train_model.py --target "Total_Generation_GWh" --start 2025-08-01 --months 6
```
Outputs
- Models saved in [`artifacts/`](file:///d:/Energy%20Prediction%20App/artifacts) as `model_<target>_<algo>.joblib`
- Forecast CSVs in `artifacts/forecast_<target>_<YYYYMM>_<months>m.csv`

CLI script: [`train_model.py`](file:///d:/Energy%20Prediction%20App/scripts/train_model.py)

## Notebook workflow
Use the notebook for experimental work (feature ablations, model comparisons, error analysis).
- Notebook: [`01_model_training.ipynb`](file:///d:/Energy%20Prediction%20App/notebooks/01_model_training.ipynb)
- Recommended: add your EDA and evaluation plots here; save models to [`artifacts/`](file:///d:/Energy%20Prediction%20App/artifacts)

## Team workflows (5 members)
1) Data acquisition & preprocessing
   - Own cleaning/aggregation and data dictionary; QA the dataset.
2) Rainfall forecasting
   - Build/tune rainfall models; compare algorithms & features.
3) Hydro & total modeling
   - Map rainfall → hydro → total; add demand + risk flag (if demand added).
4) App & UX
   - Polish UI; ensure scenario, model loading, forecast table, and KPIs work great.
5) Experiments, artifacts & deployment
   - Reproducibility, docs, and Streamlit Cloud deploy.

For detailed workflow guidance, see the sections and code references inside:
- [`app.py`](file:///d:/Energy%20Prediction%20App/app.py)
- [`pipeline.py`](file:///d:/Energy%20Prediction%20App/src/pipeline.py)
- [`train_model.py`](file:///d:/Energy%20Prediction%20App/scripts/train_model.py)

## Deployment (Streamlit Cloud)
- Push this repo to GitHub (e.g., `main` branch).
- Create a new app in Streamlit Cloud and set the main file to [`app.py`](file:///d:/Energy%20Prediction%20App/app.py).
- The theme and dependencies are provided in:
  - [`requirements.txt`](file:///d:/Energy%20Prediction%20App/requirements.txt)
  - [`.streamlit/config.toml`](file:///d:/Energy%20Prediction%20App/.streamlit/config.toml)

## Reproducibility
- Python 3.10 recommended. Optional: create a conda env via [`environment.yml`](file:///d:/Energy%20Prediction%20App/environment.yml).
- Keep large binaries in [`artifacts/`](file:///d:/Energy%20Prediction%20App/artifacts) (already gitignored).
- Use consistent commands (documented above) and pin requirements.

## Data & files
- Default daily CSV: [`sri_lanka_weather_energy_daily.csv`](file:///d:/Energy%20Prediction%20App/data/sri_lanka_weather_energy_daily.csv)
- You can replace or upload your own CSV with the same column semantics.

---
Want help adding a “start at month” date picker to the app, or filtering the model dropdown to only show compatible targets? I can add those next.
