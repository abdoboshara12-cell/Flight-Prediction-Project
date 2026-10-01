from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Set the input file and output folders.
project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "processed" / "flights_clean.csv"
models_dir = project_dir / "models"
results_dir = project_dir / "results"

models_dir.mkdir(exist_ok=True)
results_dir.mkdir(exist_ok=True)

df = pd.read_csv(data_path)

features = [
    "Destination Airport",
    "departure_hour",
    "day_of_week",
]
target = "Departure delay (Minutes)"

df["flight_date"] = pd.to_datetime(
    df["Date (MM/DD/YYYY)"],
    format="%m/%d/%Y",
)

df = df.dropna(subset=features + [target, "flight_date"])
df = df.sort_values("flight_date")

# Split dates into training, validation, and testing periods.
# Flights on the same date always stay in the same group.
dates = df["flight_date"].drop_duplicates().tolist()

if len(dates) < 5:
    raise ValueError("At least five different flight dates are needed.")

validation_start = dates[int(len(dates) * 0.60)]
test_start = dates[int(len(dates) * 0.80)]

train = df[df["flight_date"] < validation_start]
validation = df[
    (df["flight_date"] >= validation_start)
    & (df["flight_date"] < test_start)
]
test = df[df["flight_date"] >= test_start]

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]

print(f"Training flights: {len(train):,}")
print(f"Validation flights: {len(validation):,}")
print(f"Testing flights: {len(test):,}")


# Each model gets its own preprocessing pipeline.
# Encode destination and weekday as categories, and scale departure hour.
def build_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                ["Destination Airport", "day_of_week"],
            ),
            (
                "numbers",
                StandardScaler(),
                ["departure_hour"],
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


models = {
    "Median Baseline": build_pipeline(
        DummyRegressor(strategy="median")
    ),
    "Linear Regression": build_pipeline(
        LinearRegression()
    ),
    "Random Forest": build_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )
    ),
}

# The baseline always predicts the training data's median delay.
# Validation data determines which model we select.
validation_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    mae = mean_absolute_error(y_validation, predictions)
    validation_scores[name] = mae

    print(f"Validation MAE: {mae:.2f} minutes")

best_name = min(validation_scores, key=validation_scores.get)

# Refit using both earlier periods, then evaluate on the latest flights.
# The winner is already selected before looking at test results.
development = pd.concat([train, validation])
results = []

for name, model in models.items():
    model.fit(development[features], development[target])
    predictions = model.predict(X_test)

    results.append({
        "model": name,
        "validation_mae_minutes": validation_scores[name],
        "test_mae_minutes": mean_absolute_error(y_test, predictions),
        "selected": name == best_name,
    })

results_df = pd.DataFrame(results)
results_df.to_csv(results_dir / "model_results.csv", index=False)

# Save the selected pipeline, including its encoder and scaler.
best_model = models[best_name]
model_path = models_dir / "flight_delay_model.joblib"
joblib.dump(best_model, model_path)

# Save actual and predicted delays for later inspection.
test_predictions = test[
    ["Date (MM/DD/YYYY)"] + features
].copy()

test_predictions["actual_delay_minutes"] = y_test
test_predictions["predicted_delay_minutes"] = best_model.predict(X_test)
test_predictions.to_csv(
    results_dir / "test_predictions.csv",
    index=False,
)

print("\nModel results:")
print(results_df.round(2).to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Model saved to: {model_path}")