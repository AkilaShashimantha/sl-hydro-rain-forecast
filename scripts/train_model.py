import argparse
from pathlib import Path
import pandas as pd
import joblib
from src.pipeline import (
    load_csv,
    clean_daily,
    aggregate_monthly,
    make_supervised,
    train_model,
    ts_backtest,
    recursive_forecast,
)

def main():
    parser = argparse.ArgumentParser(description="Train forecasting model, optionally forecast from a start date, and save artifact")
    parser.add_argument("--data", type=str, default="data/sri_lanka_weather_energy_daily.csv", help="Path to daily CSV")
    parser.add_argument("--target", type=str, default="Hydroelectric_Production_GWh", help="Target column to forecast")
    parser.add_argument("--algo", type=str, default="Gradient Boosting", choices=["Gradient Boosting", "Random Forest", "Linear Regression"], help="Model algorithm")
    parser.add_argument("--out", type=str, default=None, help="Output model path (.joblib). Defaults to artifacts/auto")
    parser.add_argument("--start", type=str, default=None, help="Forecast start month (YYYY-MM or YYYY-MM-01). If set, uses actuals up to previous month.")
    parser.add_argument("--months", type=int, default=6, help="Forecast horizon in months")
    args = parser.parse_args()

    raw = load_csv(args.data)
    daily = clean_daily(raw)
    monthly = aggregate_monthly(daily)

    # If a start date is provided, truncate monthly history to end at previous month
    if args.start:
        start_ts = pd.to_datetime(args.start).to_period('M').to_timestamp()
        cutoff = (start_ts - pd.offsets.MonthBegin(1)).normalize()
        monthly = monthly.loc[monthly.index <= cutoff]
        print(f"Using actuals through {cutoff.date()} and forecasting from {start_ts.date()} for {args.months} months")

    if args.target not in monthly.columns:
        raise SystemExit(f"Target '{args.target}' not found in monthly columns: {monthly.columns.tolist()}")

    sup = make_supervised(monthly, args.target)
    if sup.empty:
        raise SystemExit("Not enough data after feature engineering to train the model.")

    X, y = sup.drop(columns=[args.target]), sup[args.target]

    mae, rmse = ts_backtest(X, y, algo=args.algo, n_splits=min(5, max(2, len(X)//24)))
    print(f"CV MAE: {mae:.3f} | CV RMSE: {rmse:.3f}")

    model = train_model(X, y, args.algo)

    # Save model
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"model_{args.target}_{args.algo.replace(' ', '_').lower()}.joblib"
    joblib.dump({"model": model, "features": X.columns.tolist(), "target": args.target}, out_path)
    print(f"Saved model to: {out_path}")

    # Optional forecast
    if args.start:
        future = recursive_forecast(monthly, X, args.target, model, args.months, scenario={})
        print("\nForecast:")
        print(future)
        csv_path = out_dir / f"forecast_{args.target}_{start_ts.strftime('%Y%m')}_{args.months}m.csv"
        future.to_csv(csv_path)
        print(f"Saved forecast to: {csv_path}")

if __name__ == "__main__":
    main()
