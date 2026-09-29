from pathlib import Path

import pandas as pd


raw_path = "data/raw/flights.csv"
output_path = "data/processed/flights_clean.csv"

# Skip the report details above the CSV header.
df = pd.read_csv(raw_path, skiprows=7)

required_columns = [
    "Date (MM/DD/YYYY)",
    "Destination Airport",
    "Scheduled departure time",
    "Departure delay (Minutes)",
]

df = df[required_columns].copy()
df = df.dropna()

print(df.head())
print(f"Flights remaining: {len(df):,}")
print(df.dtypes)
print(df["Scheduled departure time"].head(10).tolist())

# Create features using information available before departure.
flight_dates = pd.to_datetime(
    df["Date (MM/DD/YYYY)"],
    format="%m/%d/%Y",
    errors="coerce",
)
df["day_of_week"] = flight_dates.dt.dayofweek

df["departure_hour"] = pd.to_datetime(
    df["Scheduled departure time"],
    format="%H:%M",
    errors="coerce",
).dt.hour

df = df.dropna(
    subset=[
        "day_of_week",
        "departure_hour",
        "Departure delay (Minutes)",
    ]
)

# Save all usable flights. Departure delay in minutes is the model's target.
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"Flights saved: {len(df):,}")
print(f"Saved to: {output_path}")