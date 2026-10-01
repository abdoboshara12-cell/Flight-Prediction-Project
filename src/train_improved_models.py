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


project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data/processed/flights_features.csv"
models_dir = project_dir / "models/improved"
results_dir = project_dir / "results/improved"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

# Keep the original features and add calendar and time features.
categorical_features = [
    "Destination Airport",
    "day_of_week",
    "month",
]

numeric_features = [
    "departure_hour",
    "year",
    "is_weekend",
    "minutes_since_midnight",
    "departure_time_sin",
    "departure_time_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

features = categorical_features + numeric_features
target = "Departure delay (Minutes)"

df["flight_date"] = pd.to_datetime(
    df["Date (MM/DD/YYYY)"],
    format="%m/%d/%Y",
)

# Apply the same row filtering as the original experiment.
basic_features = [
    "Destination Airport",
    "departure_hour",
    "day_of_week",
]

df = df.dropna(subset=basic_features + [target, "flight_date"])

# Stop instead of silently dropping extra flights and changing the split.
if df[features].isna().any().any():
    raise ValueError(
        "Some engineered features are missing. "
        "Check feature_engineering.py before training."
    )

df = df.sort_values("flight_date")

# Use the same date-based 60/20/20 split as the original experiment.
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
print(f"Validation starts: {validation_start:%Y-%m-%d}")
print(f"Testing starts: {test_start:%Y-%m-%d}")


def build_pipeline(model):
    # Preprocessing is fitted only on the data used for training.
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categories",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "numbers",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# Keep model settings unchanged to isolate the effect of new features.
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

validation_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)

    mae = mean_absolute_error(y_validation, predictions)
    validation_scores[name] = mae

    print(f"Validation MAE: {mae:.2f} minutes")

# Choose the winner using validation scores, before test evaluation.
best_name = min(validation_scores, key=validation_scores.get)

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

best_model = models[best_name]
model_path = models_dir / "flight_delay_model.joblib"
joblib.dump(best_model, model_path)

test_predictions = test[
    ["Date (MM/DD/YYYY)"] + features
].copy()

test_predictions["actual_delay_minutes"] = y_test
test_predictions["predicted_delay_minutes"] = best_model.predict(X_test)
test_predictions.to_csv(
    results_dir / "test_predictions.csv",
    index=False,
)

# Record the inputs used in this experiment for GitHub readers.
pd.DataFrame({"feature": features}).to_csv(
    results_dir / "features_used.csv",
    index=False,
)

print("\nImproved experiment results:")
print(results_df.round(2).to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Model saved to: {model_path}")

# Compare against the saved results from the original experiment.
basic_results_path = project_dir / "results/model_results.csv"

if basic_results_path.exists():
    basic_results = pd.read_csv(basic_results_path)

    comparison = basic_results.merge(
        results_df,
        on="model",
        suffixes=("_basic", "_improved"),
        validate="one_to_one",
    )

    comparison["validation_mae_reduction_minutes"] = (
        comparison["validation_mae_minutes_basic"]
        - comparison["validation_mae_minutes_improved"]
    )

    comparison.to_csv(
        results_dir / "comparison.csv",
        index=False,
    )

    print("\nComparison with original experiment:")
    print(
        comparison[[
            "model",
            "validation_mae_minutes_basic",
            "validation_mae_minutes_improved",
            "validation_mae_reduction_minutes",
        ]].round(2).to_string(index=False)
    )
    print("\nPositive MAE reduction means the new features helped.")
else:
    print("\nOriginal results were not found; comparison skipped.")