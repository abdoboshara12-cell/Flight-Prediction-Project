from pathlib import Path

import numpy as np
import pandas as pd


# Locate the project folder, regardless of where the script is run.
project_root = Path(__file__).resolve().parent.parent

input_path = project_root / "data/processed/flights_clean.csv"
output_path = project_root / "data/processed/flights_features.csv"


def create_features(df):
    df = df.copy()

    flight_dates = pd.to_datetime(
        df["Date (MM/DD/YYYY)"],
        format="%m/%d/%Y",
        errors="raise",
    )

    departure_times = pd.to_datetime(
        df["Scheduled departure time"],
        format="%H:%M",
        errors="raise",
    )

    # Calendar features available before departure.
    df["year"] = flight_dates.dt.year
    df["month"] = flight_dates.dt.month
    df["day_of_week"] = flight_dates.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Preserve minutes instead of using only the departure hour.
    df["departure_hour"] = departure_times.dt.hour
    df["departure_minute"] = departure_times.dt.minute

    df["minutes_since_midnight"] = (
        df["departure_hour"] * 60 + df["departure_minute"]
    )

    # Cyclical features represent repeating time patterns.
    # For example, 23:59 and 00:01 should be close together.
    time_angle = 2 * np.pi * df["minutes_since_midnight"] / 1440
    df["departure_time_sin"] = np.sin(time_angle)
    df["departure_time_cos"] = np.cos(time_angle)

    # Sunday and Monday are adjacent in the weekly cycle.
    weekday_angle = 2 * np.pi * df["day_of_week"] / 7
    df["day_of_week_sin"] = np.sin(weekday_angle)
    df["day_of_week_cos"] = np.cos(weekday_angle)

    # December and January are adjacent in the yearly cycle.
    month_angle = 2 * np.pi * (df["month"] - 1) / 12
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)

    return df


def main():
    df = pd.read_csv(input_path)
    original_columns = set(df.columns)

    df = create_features(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    added_columns = [
        column for column in df.columns
        if column not in original_columns
    ]

    print(f"Flights saved: {len(df):,}")
    print(f"Added features: {', '.join(added_columns)}")
    print(f"Saved to: {output_path}")
    print(df.head())


if __name__ == "__main__":
    main()