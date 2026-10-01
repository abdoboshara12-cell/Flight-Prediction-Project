# Flight Departure Delay Prediction

This project uses flight data to predict **how many minutes late a flight will depart**. The dataset contains approximately 250,000 Southwest Airlines flights departing from Los Angeles International Airport (LAX).

## Data

The data comes from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/ONTIME/Departures.aspx) and covers 2017 through 2024.

Place the downloaded CSV at `data/raw/flights.csv`. The cleaning script reads that file and saves the usable flights to `data/processed/flights_clean.csv`. It keeps all usable flights rather than taking a smaller sample.

## Data cleaning

`src/clean_data.py`:

- Skips the report information above the CSV header
- Keeps the flight date, destination, scheduled departure time, and departure delay
- Removes rows with missing or invalid required values
- Creates the day of the week and scheduled departure hour
- Saves the cleaned CSV

Run it from the main project folder:

```bash
source .venv/bin/activate
python src/clean_data.py
```

## Prediction goal

The model will predict `Departure delay (Minutes)`. Its inputs will include the destination airport, day of the week, and scheduled departure hour.

The recorded delay and actual departure time will **not** be used as inputs because they are not known before the flight departs.

## Models and evaluation

I plan to compare Linear Regression and Random Forest Regressor. I will evaluate them using mean absolute error (MAE), which measures the average size of the prediction error in minutes.

**Results will be added after training the models.**

## Limitations

The dataset contains only Southwest flights departing LAX. It does not include weather, cancellation flags, or all factors that may cause a delay