import os
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

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


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9eE+\-\.]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clean_daily(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if not {"Year", "Month", "Day"}.issubset(df.columns):
        raise ValueError("Expected columns Year, Month, Day are missing from the dataset.")

    parts = df[["Year", "Month", "Day"]].astype(str).agg(" ".join, axis=1)
    df["date"] = pd.to_datetime(parts, errors="coerce")
    df = df.dropna(subset=["date"])

    for col in NUMERIC_COLS_BASE:
        if col in df.columns:
            df[col] = _to_numeric_clean(df[col])

    if "Average_Rainfall_mm" in df.columns:
        df["Average_Rainfall_mm"] = df["Average_Rainfall_mm"].clip(lower=0)
    for c in PROD_COLS:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="first")
    return df


def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
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

    present_prod_cols = [c for c in PROD_COLS if c in m.columns]
    if present_prod_cols:
        m["Total_Generation_GWh"] = m[present_prod_cols].sum(axis=1)

    m = m.sort_index()
    m = m.interpolate(limit_direction="both")
    return m


def add_time_features(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()
    out["month"] = out.index.month
    out["year"] = out.index.year
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def make_supervised(m: pd.DataFrame, target: str, lags=(1, 3, 6, 12), roll_windows=(3, 6, 12)) -> pd.DataFrame:
    df = add_time_features(m)
    for L in lags:
        df[f"{target}_lag{L}"] = df[target].shift(L)
    for W in roll_windows:
        df[f"{target}_roll{W}_mean"] = df[target].rolling(W).mean()
        df[f"{target}_roll{W}_std"] = df[target].rolling(W).std()
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


def recursive_forecast(base_m: pd.DataFrame, feature_m: pd.DataFrame, target: str, model, horizon: int, scenario: dict):
    last_hist = pd.to_datetime(base_m.index.max())
    if pd.isna(last_hist):
        future_index = pd.date_range(pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
        return pd.DataFrame(index=future_index, columns=[target])

    start_month = (last_hist + pd.offsets.MonthBegin(1)).normalize()
    future_index = pd.date_range(start=start_month, periods=horizon, freq="MS")
    m = base_m.copy().sort_index()

    preds = {}
    for dt in future_index:
        tmp = m.copy()
        if dt not in tmp.index:
            tmp = tmp.reindex(tmp.index.union([dt]))
        for col in tmp.columns:
            if pd.isna(tmp.loc[dt, col]):
                last_val = tmp[col].ffill().iloc[-1] if not tmp[col].ffill().empty else np.nan
                tmp.loc[dt, col] = last_val
        for k, mult in (scenario or {}).items():
            if k in tmp.columns and not pd.isna(tmp.loc[dt, k]):
                tmp.loc[dt, k] = float(tmp.loc[dt, k]) * float(mult)
        sup = make_supervised(tmp, target)
        if dt not in sup.index:
            preds[dt] = np.nan
            # Persist exogenous values to enable next-step lags
            for c in tmp.columns:
                if c != target:
                    m.loc[dt, c] = tmp.loc[dt, c]
            # Carry forward target for continuity
            if target in m.columns and not m[target].dropna().empty:
                m.loc[dt, target] = float(m[target].ffill().iloc[-1])
            else:
                m.loc[dt, target] = np.nan
            continue
        row = sup.loc[[dt]]
        X_row = row.drop(columns=[target])
        y_pred = float(model.predict(X_row)[0])
        preds[dt] = y_pred
        # Persist exogenous values for future lags
        for c in tmp.columns:
            if c != target:
                m.loc[dt, c] = tmp.loc[dt, c]
        m.loc[dt, target] = y_pred

    out = pd.DataFrame({target: pd.Series(preds)}, index=future_index)
    return out
