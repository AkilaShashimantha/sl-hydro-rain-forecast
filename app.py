import os
import re
import io
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------
# Page config and styling
# ------------------------------
st.set_page_config(
    page_title="Sri Lanka Rain & Energy Forecasting",
    page_icon="💧⚡",
    layout="wide",
)

# Load custom CSS if present
CSS_PATH = os.path.join(os.getcwd(), "assets", "styles.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ------------------------------
# Helpers
# ------------------------------
NUMERIC_COLS_BASE = [
    "Average_Rainfall_mm",
    "Average_Temperature_C",
    "Average_Humidity_%",
    "Average_Wind_Speed_kmh",
    "Precipitation_Hours_h",
    "Daylight_Duration_s",
    "Hydroelectric_Production_GWh",
    "Solar_Production_GWh",
    "Coal_Production_GWh",
    "Fuel_Production_GWh",
]

PROD_COLS = [
    "Hydroelectric_Production_GWh",
    "Solar_Production_GWh",
    "Coal_Production_GWh",
    "Fuel_Production_GWh",
]

@st.cache_data(show_spinner=False)
def load_csv(default_path: str, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(default_path)


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    # Strip any non-numeric chars like ' GWh'
    s = series.astype(str).str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names if there are surrounding spaces
    df.columns = [c.strip() for c in df.columns]

    # Build date column
    # Some rows may miss Day/Month. Drop rows without complete date parts.
    if not {"Year", "Month", "Day"}.issubset(df.columns):
        raise ValueError("Expected columns Year, Month, Day are missing from the dataset.")

    # Coerce month/day
    # If Month is e.g., 'April' or 'Aug', pandas can parse when combined
    parts = df[["Year", "Month", "Day"]].astype(str).agg(" ".join, axis=1)
    df["date"] = pd.to_datetime(parts, errors="coerce")
    df = df.dropna(subset=["date"])  # keep rows with valid dates

    # Coerce numeric columns, handling strings like '-2.0 GWh'
    for col in NUMERIC_COLS_BASE:
        if col in df.columns:
            df[col] = _to_numeric_clean(df[col])

    # Clip physically impossible negatives for rainfall and productions
    if "Average_Rainfall_mm" in df.columns:
        df["Average_Rainfall_mm"] = df["Average_Rainfall_mm"].clip(lower=0)
    for c in PROD_COLS:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    # Drop exact duplicate dates keeping the first occurrence
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")

    return df


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    # Define how to aggregate daily -> monthly
    agg_map = {}
    if "Average_Rainfall_mm" in daily.columns:
        agg_map["Average_Rainfall_mm"] = "sum"
    for mean_col in ["Average_Temperature_C", "Average_Humidity_%", "Average_Wind_Speed_kmh", "Daylight_Duration_s"]:
        if mean_col in daily.columns:
            agg_map[mean_col] = "mean"
    if "Precipitation_Hours_h" in daily.columns:
        agg_map["Precipitation_Hours_h"] = "sum"
    for p in PROD_COLS:
        if p in daily.columns:
            agg_map[p] = "sum"

    m = daily.set_index("date").resample("MS").agg(agg_map)
    m.index.name = "date"

    # Derive total generation if possible
    present_prod_cols = [c for c in PROD_COLS if c in m.columns]
    if present_prod_cols:
        m["Total_Generation_GWh"] = m[present_prod_cols].sum(axis=1)

    # Interpolate small gaps, then forward/backward fill remaining
    m = m.sort_index()
    m = m.interpolate(limit_direction="both")

    return m


def add_time_features(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["month"] = out.index.month
    out["year"] = out.index.year
    # Cyclical encoding for month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def make_supervised(m: pd.DataFrame, target: str, lags=(1, 3, 6, 12), roll_windows=(3, 6, 12)) -> pd.DataFrame:
    df = add_time_features(m)

    # Lag features of the target
    for L in lags:
        df[f"{target}_lag{L}"] = df[target].shift(L)

    # Rolling statistics on the target
    for W in roll_windows:
        df[f"{target}_roll{W}_mean"] = df[target].rolling(W).mean()
        df[f"{target}_roll{W}_std"] = df[target].rolling(W).std()

    # Also include exogenous features' recent values (lags) to help prediction
    exog_cols = [c for c in df.columns if c not in {target, "month", "year", "month_sin", "month_cos"} and not c.startswith(target)]
    for c in exog_cols:
        df[f"{c}_lag1"] = df[c].shift(1)
        df[f"{c}_roll3_mean"] = df[c].rolling(3).mean()

    df = df.dropna()
    return df


def train_model(X: pd.DataFrame, y: pd.Series, algo: str = "Gradient Boosting"):
    if algo == "Linear Regression":
        model = LinearRegression()
    elif algo == "Random Forest":
        model = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    else:
        # HistGradientBoosting is fast and robust
        model = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=6,
            max_iter=600,
            l2_regularization=1.0,
            random_state=42,
        )

    model.fit(X, y)
    return model


def ts_backtest(X: pd.DataFrame, y: pd.Series, algo: str = "Gradient Boosting", n_splits: int = 4):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        model = train_model(X_tr, y_tr, algo)
        pred = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, pred))
        rmses.append(math.sqrt(mean_squared_error(y_te, pred)))
    return float(np.mean(maes)), float(np.mean(rmses))


def recursive_forecast(base_m: pd.DataFrame, feature_m: pd.DataFrame, target: str, model, horizon: int, scenario: dict = None, overrides: dict = None, feature_cols: list | None = None):
    """Robust recursive multi-step forecast.

    Ensures we always return a frame indexed by the requested future months,
    even if some steps cannot be computed due to insufficient history. In such
    cases, we fallback to the last known target to proceed, and mark the output
    as NaN for that step (so downstream code can handle gracefully).
    """
    # Determine forecast start based on the last month present in the base matrix
    last_hist = pd.to_datetime(base_m.index.max())
    if pd.isna(last_hist):
        # Empty history – return an empty frame with the requested index
        future_index = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return pd.DataFrame(index=future_index, columns=[target])

    start_month = (last_hist + pd.offsets.MonthBegin(1)).normalize()
    future_index = pd.date_range(start=start_month, periods=horizon, freq="MS")

    m = base_m.copy().sort_index()

    preds = {}
    rows = []
    present_cols = [c for c in NUMERIC_COLS_BASE if c in m.columns]
    include_total = "Total_Generation_GWh" in m.columns
    prod_present = [c for c in PROD_COLS if c in m.columns]

    # Precompute simple monthly climatology for exogenous columns
    m_with_month = m.copy()
    m_with_month["__mo__"] = m_with_month.index.month
    clim_cols = [c for c in present_cols if c != target]
    monthly_clim = m_with_month.groupby("__mo__")[clim_cols].mean(numeric_only=True)

    for dt in future_index:
        # Build a single-row feature frame for dt using last available values
        tmp = m.copy()
        if dt not in tmp.index:
            tmp = tmp.reindex(tmp.index.union([dt]))
        # Fill new row with a blend of last value and monthly climatology
        mo = dt.month
        for col in tmp.columns:
            if pd.isna(tmp.loc[dt, col]):
                last_val = tmp[col].ffill().iloc[-1] if not tmp[col].ffill().empty else np.nan
                clim_val = np.nan
                if col in monthly_clim.columns and mo in monthly_clim.index:
                    try:
                        clim_val = float(monthly_clim.loc[mo, col])
                    except Exception:
                        clim_val = np.nan
                if not np.isnan(clim_val) and not np.isnan(last_val):
                    tmp.loc[dt, col] = 0.5 * float(last_val) + 0.5 * float(clim_val)
                elif not np.isnan(clim_val):
                    tmp.loc[dt, col] = float(clim_val)
                else:
                    tmp.loc[dt, col] = last_val

        # Apply scenario multipliers on exog for the future step
        for k, mult in (scenario or {}).items():
            if k in tmp.columns and not pd.isna(tmp.loc[dt, k]):
                tmp.loc[dt, k] = float(tmp.loc[dt, k]) * float(mult)
        # Apply absolute overrides after multipliers (if provided)
        for k, val in (overrides or {}).items():
            if k in tmp.columns:
                tmp.loc[dt, k] = float(val)

        # Build supervised row for this dt
        sup = make_supervised(tmp, target)
        if dt not in sup.index:
            # Relaxed feature build to allow recursive multi-step forecasting
            df_feat = add_time_features(tmp.copy())
            for L in (1, 3, 6, 12):
                df_feat[f"{target}_lag{L}"] = df_feat[target].shift(L)
            for W in (3, 6, 12):
                df_feat[f"{target}_roll{W}_mean"] = df_feat[target].rolling(W, min_periods=1).mean()
                df_feat[f"{target}_roll{W}_std"] = df_feat[target].rolling(W, min_periods=2).std().fillna(0.0)
            exog_cols_rel = [c for c in df_feat.columns if c not in {target, "month", "year", "month_sin", "month_cos"} and not c.startswith(target)]
            for c in exog_cols_rel:
                df_feat[f"{c}_lag1"] = df_feat[c].shift(1)
                df_feat[f"{c}_roll3_mean"] = df_feat[c].rolling(3, min_periods=1).mean()
            # Impute missing with forward-fill then zeros
            df_feat_imputed = df_feat.fillna(method="ffill").fillna(0.0)
            row_feat = df_feat_imputed.loc[dt]
            # Build X in the correct order if provided
            if feature_cols:
                X_row = row_feat.reindex(feature_cols, fill_value=0.0)
            else:
                X_row = row_feat.drop(labels=[target])
            y_pred = float(model.predict(pd.DataFrame([X_row]))[0])

            # Apply scenario/overrides to target if specified
            if overrides and target in overrides and overrides[target] is not None:
                try:
                    y_pred = float(overrides[target])
                except Exception:
                    pass
            elif scenario and target in scenario:
                try:
                    y_pred = float(y_pred) * float(scenario[target])
                except Exception:
                    pass

            preds[dt] = y_pred
            # Persist exogenous values to enable next steps
            for c in tmp.columns:
                if c != target:
                    m.loc[dt, c] = tmp.loc[dt, c]
            m.loc[dt, target] = y_pred

            # Prepare row snapshot
            row_dict = {col: m.loc[dt, col] if col in m.columns else np.nan for col in present_cols}
            if include_total and prod_present:
                row_dict["Total_Generation_GWh"] = float(np.nansum([m.loc[dt, c] for c in prod_present]))
                if "Total_Generation_GWh" in m.columns:
                    m.loc[dt, "Total_Generation_GWh"] = row_dict["Total_Generation_GWh"]
            row_dict[target] = y_pred
            row_dict["date"] = dt
            rows.append(row_dict)
            continue

        row = sup.loc[[dt]]
        if feature_cols:
            # Ensure correct feature order for loaded models
            X_row = row[feature_cols]
        else:
            X_row = row.drop(columns=[target])
        y_pred = float(model.predict(X_row)[0])

        # If user provided override for the target, use it. Else, apply multiplier for target if specified.
        if overrides and target in overrides and overrides[target] is not None:
            try:
                y_pred = float(overrides[target])
            except Exception:
                pass
        elif scenario and target in scenario:
            try:
                y_pred = float(y_pred) * float(scenario[target])
            except Exception:
                pass

        preds[dt] = y_pred
        # Persist exogenous scenario/override values for future lags
        for k in set(list((scenario or {}).keys()) + list((overrides or {}).keys())):
            if k != target and k in tmp.columns:
                m.loc[dt, k] = tmp.loc[dt, k]
        # Write prediction into base matrix for next iterations
        m.loc[dt, target] = y_pred

        # Keep row snapshot including exogenous values used
        row_dict = {col: m.loc[dt, col] if col in m.columns else np.nan for col in present_cols}
        if include_total and prod_present:
            # Recompute total using current production columns
            row_dict["Total_Generation_GWh"] = float(np.nansum([m.loc[dt, c] for c in prod_present]))
            # Also keep m consistent
            if "Total_Generation_GWh" in m.columns:
                m.loc[dt, "Total_Generation_GWh"] = row_dict["Total_Generation_GWh"]
        row_dict[target] = y_pred
        row_dict["date"] = dt
        rows.append(row_dict)

    # Return a DataFrame with future values for target and key exogenous columns
    out = pd.DataFrame(rows).set_index("date")
    # Ensure index covers the full requested range (fill missing with NaN)
    out = out.reindex(future_index)
    return out

# ------------------------------
# UI
# ------------------------------
def kpi_card(label: str, value: float, suffix: str = "", delta: float | None = None):
    try:
        delta_str = None if (delta is None or (isinstance(delta, float) and (np.isnan(delta) or np.isinf(delta)))) else f"{delta:,.2f}{suffix}"
    except Exception:
        delta_str = None
    st.metric(label, f"{value:,.2f}{suffix}", delta_str)


def main():
    st.markdown(
        """
        <div class="app-header">
            <div>
                <h1>💧⚡ Sri Lanka Rain & Energy Forecasting</h1>
                <p>Forecast monthly rainfall and energy production, explore climate scenarios, and plan resilient hybrid strategies.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Data & Model")
        uploaded = st.file_uploader("Upload CSV (optional)", type=["csv"])
        default_path = os.path.join(os.getcwd(), "data", "sri_lanka_weather_energy_daily.csv")

        horizon = st.slider("Forecast horizon (months)", 1, 12, 6)
        target_choice = st.selectbox(
            "Target to forecast",
            [
                "Monthly Rainfall (mm)",
                "Hydroelectric Production (GWh)",
                "Total Generation (GWh)",
            ],
        )
        algo = st.selectbox("Model", ["Gradient Boosting", "Random Forest", "Linear Regression"], index=0)

        model_source = st.radio("Model source", ["Train now", "Load trained model"], index=0, horizontal=True)
        loaded = None
        feature_cols_loaded = None
        loaded_target = None
        if model_source == "Load trained model":
            art_dir = os.path.join(os.getcwd(), "artifacts")
            files = []
            try:
                files = [f for f in os.listdir(art_dir) if f.endswith(".joblib")]
            except Exception:
                pass
            if files:
                chosen = st.selectbox("Select trained model", files)
                if chosen:
                    payload = joblib.load(os.path.join(art_dir, chosen))
                    loaded = payload.get("model")
                    feature_cols_loaded = payload.get("features")
                    loaded_target = payload.get("target")
                    if loaded_target:
                        st.caption(f"Loaded model target: {loaded_target}")
            else:
                st.warning("No .joblib models found in artifacts/. Switch to 'Train now' or upload a model.")

        st.header("Scenario")
        st.caption("Choose multipliers or manually set future values (overrides apply to all forecast months).")
        rain_mult = st.slider("Rainfall ×", 0.5, 1.5, 1.0, 0.05)
        temp_mult = st.slider("Temperature ×", 0.95, 1.10, 1.0, 0.01)
        hydro_mult = st.slider("Hydro Prod ×", 0.5, 1.5, 1.0, 0.05)

        use_manual = st.checkbox("Manually set constant future values", value=False)
        overrides = {}
        if use_manual:
            # Basic defaults (if monthly not yet available on first render)
            default_rain = 0.0
            default_temp = 25.0
            default_hydro = 0.0
            rain_override = st.number_input("Future monthly rainfall (mm)", min_value=0.0, value=default_rain, step=1.0)
            temp_override = st.number_input("Future avg temperature (°C)", min_value=0.0, value=default_temp, step=0.1)
            hydro_override = st.number_input("Future hydro production (GWh)", min_value=0.0, value=default_hydro, step=1.0)
            overrides = {
                "Average_Rainfall_mm": rain_override,
                "Average_Temperature_C": temp_override,
                "Hydroelectric_Production_GWh": hydro_override,
            }

        scenario = {
            "Average_Rainfall_mm": rain_mult,
            "Average_Temperature_C": temp_mult,
            "Hydroelectric_Production_GWh": hydro_mult,
        }

    # Load and prepare data
    try:
        raw = load_csv(default_path, uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    try:
        daily = clean_daily(raw)
        monthly = aggregate_monthly(daily)
    except Exception as e:
        st.error(f"Data cleaning/aggregation error: {e}")
        return

    # Select target mapping
    if target_choice == "Monthly Rainfall (mm)":
        target = "Average_Rainfall_mm"
        y_label = "Rainfall (mm)"
        unit = " mm"
    elif target_choice == "Hydroelectric Production (GWh)":
        target = "Hydroelectric_Production_GWh"
        y_label = "Hydroelectric (GWh)"
        unit = " GWh"
        if target not in monthly.columns:
            st.warning("Hydroelectric production column not found in data.")
    else:
        target = "Total_Generation_GWh"
        y_label = "Total Generation (GWh)"
        unit = " GWh"
        if target not in monthly.columns:
            st.warning("Total generation not available (missing production columns).")

    # Build supervised dataset
    if target not in monthly.columns:
        st.stop()

    sup = make_supervised(monthly, target)
    if sup.empty:
        st.warning("Not enough historical data after feature engineering.")
        st.stop()

    X = sup.drop(columns=[target])
    y = sup[target]

    # Train or load model
    use_loaded = False
    if loaded is not None and feature_cols_loaded:
        # Check target compatibility
        if loaded_target and loaded_target != target:
            st.info(f"Selected model was trained for '{loaded_target}'. Please select a model trained for '{target}', or switch to 'Train now'. Using training instead.")
        else:
            # Align features
            missing = [c for c in feature_cols_loaded if c not in X.columns]
            if missing:
                st.info("The selected model expects features not present in the current dataset. Using training instead.")
            else:
                model = loaded
                use_loaded = True
                # Quick holdout metrics on available history with loaded model
                try:
                    split = int(0.8 * len(X))
                    pred_tmp = model.predict(X.iloc[split:][feature_cols_loaded])
                    mae = mean_absolute_error(y.iloc[split:], pred_tmp)
                    rmse = math.sqrt(mean_squared_error(y.iloc[split:], pred_tmp))
                except Exception:
                    mae, rmse = float("nan"), float("nan")
                feature_cols = feature_cols_loaded
    if not use_loaded:
        # Backtest metrics
        with st.spinner("Training and validating model..."):
            try:
                mae, rmse = ts_backtest(X, y, algo=algo, n_splits=min(5, max(2, len(X)//24)))
            except Exception:
                # Fallback to simple split if cross-val fails on very short series
                split = int(0.8 * len(X))
                model_tmp = train_model(X.iloc[:split], y.iloc[:split], algo)
                pred_tmp = model_tmp.predict(X.iloc[split:])
                mae = mean_absolute_error(y.iloc[split:], pred_tmp)
                rmse = math.sqrt(mean_squared_error(y.iloc[split:], pred_tmp))
        # Train final model on full history
        model = train_model(X, y, algo)
        feature_cols = X.columns.tolist()

    # Forecast future
    future_df = recursive_forecast(monthly, X, target, model, horizon, scenario, overrides, feature_cols)

    # Combine history + forecast for plot
    hist = monthly[[target]].copy()
    hist["type"] = "actual"
    fut = future_df.copy()
    fut["type"] = "forecast"
    combined = pd.concat([hist, fut]).reset_index().rename(columns={"index": "date"})

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        kpi_card("Training months", len(hist.dropna()))
    with col2:
        kpi_card("Avg absolute error (CV MAE)", mae, unit)
    with col3:
        kpi_card("Root mean squared error (CV RMSE)", rmse, unit)

    # Plot
    fig = px.line(
        combined,
        x="date",
        y=target,
        color="type",
        markers=True,
        title=f"{y_label}: history and {horizon}-month forecast",
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    # Forecast preview KPIs and table

    # Additional KPIs: last actual vs next forecast vs average horizon forecast
    last_actual = float(hist[target].dropna().iloc[-1]) if not hist[target].dropna().empty else float("nan")
    first_valid_future = future_df[target].dropna()
    next_forecast = float(first_valid_future.iloc[0]) if not first_valid_future.empty else float("nan")
    avg_forecast = float(future_df[target].mean()) if not np.isnan(future_df[target].mean()) else float("nan")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Last actual", last_actual, unit)
    with c2:
        kpi_card("Next forecast", next_forecast, unit, delta=(next_forecast - last_actual) if not (np.isnan(next_forecast) or np.isnan(last_actual)) else None)
    with c3:
        kpi_card("Avg forecast (horizon)", avg_forecast, unit)

    if np.isnan(next_forecast):
        st.info("No computable forecast for the immediate next month. Try reducing the forecast horizon or using manual overrides in Scenario.")

    st.subheader("Forecast table")
    out_preview = future_df.reset_index().rename(columns={"date": "month", target: target})
    st.dataframe(out_preview, use_container_width=True)


    # Feature importance if available
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(20)
        st.subheader("Top features")
        st.bar_chart(imp)

    # Data explorer
    with st.expander("Data explorer (monthly aggregates)"):
        st.dataframe(monthly.tail(36))

    # Download forecast
    out = future_df.reset_index().rename(columns={"date": "month", target: target})
    csv_buf = io.StringIO()
    out.to_csv(csv_buf, index=False)
    st.download_button("Download forecast CSV", csv_buf.getvalue(), file_name="forecast.csv", mime="text/csv")

    # Simple recommendations heuristics
    st.markdown("---")
    st.subheader("Recommendations")
    if target == "Hydroelectric_Production_GWh":
        recent = hist[target].tail(24)
        threshold = recent.quantile(0.25)
        low_months = future_df[target][future_df[target] < threshold]
        if not low_months.empty:
            st.info(
                f"Expected low hydro months: {', '.join([d.strftime('%b %Y') for d in low_months.index])}. "
                "Consider pre-allocating solar/thermal capacity or demand-response programs."
            )
        else:
            st.success("No significant hydro shortfalls detected in the forecast horizon.")
    elif target == "Total_Generation_GWh":
        growth = (future_df.iloc[-1, 0] - hist[target].tail(12).mean()) / max(1e-9, hist[target].tail(12).mean())
        if growth < -0.05:
            st.info("Total generation may dip vs last-year average. Plan contingency supply (e.g., battery, diesel peakers).")
        else:
            st.success("Total generation outlook is stable vs last-year average.")
    else:
        # Rainfall
        wet = future_df[target].idxmax().strftime('%b %Y')
        dry = future_df[target].idxmin().strftime('%b %Y')
        st.info(f"Wettest upcoming month: {wet}; Driest: {dry}. Align reservoir operations accordingly.")


if __name__ == "__main__":
    main()
