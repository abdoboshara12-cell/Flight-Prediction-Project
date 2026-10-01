
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# Find the cleaned data and create a folder for the charts.
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "processed" / "flights_clean.csv"
figures_dir = project_dir / "figures"
figures_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)
delay_column = "Departure delay (Minutes)"

print(f"Flights loaded: {len(df):,}")
print(df[delay_column].describe())


# Show the main range of delays; this does not remove rows from the dataset.
visible_delays = df.loc[
    df[delay_column].between(-30, 180),
    delay_column,
]

plt.figure(figsize=(10, 6))
plt.hist(visible_delays, bins=42, color="steelblue", edgecolor="white")
plt.title("Distribution of Departure Delays")
plt.xlabel("Departure delay (minutes)")
plt.ylabel("Number of flights")
plt.tight_layout()
plt.savefig(figures_dir / "delay_distribution.png", dpi=150)
plt.close()


# Calculate average delay for each scheduled departure hour.
hourly_delay = (
    df.groupby("departure_hour")[delay_column]
    .mean()
    .sort_index()
)

plt.figure(figsize=(10, 6))
plt.plot(hourly_delay.index, hourly_delay.values, marker="o")
plt.title("Average Departure Delay by Scheduled Hour")
plt.xlabel("Scheduled departure hour")
plt.ylabel("Average delay (minutes)")
plt.xticks(range(0, 24, 2))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "delay_by_hour.png", dpi=150)
plt.close()


# Compare the 10 destinations with the most flights.
top_destinations = df["Destination Airport"].value_counts().head(10).index

destination_delay = (
    df[df["Destination Airport"].isin(top_destinations)]
    .groupby("Destination Airport")[delay_column]
    .mean()
    .sort_values()
)

plt.figure(figsize=(11, 6))
plt.barh(destination_delay.index, destination_delay.values, color="teal")
plt.title("Average Delay for the 10 Most Common Destinations")
plt.xlabel("Average departure delay (minutes)")
plt.ylabel("Destination airport")
plt.tight_layout()
plt.savefig(figures_dir / "delay_by_destination.png", dpi=150)
plt.close()

print(f"Saved 3 figures to: {figures_dir}")