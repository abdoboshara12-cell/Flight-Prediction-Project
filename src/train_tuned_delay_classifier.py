from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data/processed/flights_features.csv"
results_dir = project_dir / "results/classification_tuned"
models_dir = project_dir / "models/classification_tuned"

results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

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
delay_column = "Departure delay (Minutes)"
target = "is_delayed"

df["flight_date"] = pd.to_datetime(
    df["Date (MM/DD/YYYY)"],
    format="%m/%d/%Y",
)

df[delay_column] = pd.to_numeric(df[delay_column], errors="raise")

df = df.dropna(subset=[
    "Destination Airport",
    "departure_hour",
    "day_of_week",
    delay_column,
    "flight_date",
])

if df[features].isna().any().any():
    raise ValueError("Some engineered features are missing.")

df[target] = (df[delay_column] >= 15).astype(int)
df = df.sort_values("flight_date")

# Preserve the original chronological split.
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

if train[target].nunique() < 2:
    raise ValueError("Training data must contain both classes.")

if validation[target].nunique() < 2:
    raise ValueError("Validation data must contain both classes.")

print(f"Training flights: {len(train):,}")
print(f"Validation flights: {len(validation):,}")
print(f"Testing flights: {len(test):,}")


def build_pipeline(model):
    preprocessor = ColumnTransformer([
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
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def calculate_metrics(actual, predicted):
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(
            actual, predicted, zero_division=0
        ),
        "recall": recall_score(
            actual, predicted, zero_division=0
        ),
        "f1": f1_score(
            actual, predicted, zero_division=0
        ),
    }


def delay_probabilities(model, inputs):
    delayed_index = list(model.classes_).index(1)
    return model.predict_proba(inputs)[:, delayed_index]


models = {
    "Majority Baseline": build_pipeline(
        DummyClassifier(strategy="most_frequent")
    ),
    "Always Delayed Baseline": build_pipeline(
        DummyClassifier(strategy="constant", constant=1)
    ),
    "Logistic Regression": build_pipeline(
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )
    ),
    "Random Forest": build_pipeline(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    ),
}

# Try cutoffs from 0.00 through 1.00 in steps of 0.05.
thresholds = [i / 20 for i in range(21)]
all_results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(train[features], train[target])

    probabilities = delay_probabilities(model, validation[features])

    # Keep baseline behavior fixed; tune only the learned models.
    cutoffs = [0.5] if "Baseline" in name else thresholds

    for cutoff in cutoffs:
        predictions = (probabilities >= cutoff).astype(int)

        all_results.append({
            "model": name,
            "threshold": cutoff,
            **calculate_metrics(validation[target], predictions),
        })

threshold_results = pd.DataFrame(all_results)

# Highest F1 wins. Break exact ties using higher precision.
ranked = threshold_results.sort_values(
    ["f1", "precision", "threshold"],
    ascending=[False, False, False],
    kind="stable",
)

best_per_model = ranked.drop_duplicates("model").copy()
winner = best_per_model.iloc[0]

best_name = winner["model"]
best_threshold = float(winner["threshold"])
best_model = models[best_name]

best_per_model["selected"] = (
    best_per_model["model"] == best_name
)

threshold_results.to_csv(
    results_dir / "threshold_results.csv",
    index=False,
)
best_per_model.to_csv(
    results_dir / "validation_results.csv",
    index=False,
)

print("\nBest validation cutoff for each model:")
print(best_per_model.round(3).to_string(index=False))
print(f"\nSelected model: {best_name}")
print(f"Selected probability cutoff: {best_threshold:.2f}")

# Keep the fitted model unchanged after threshold selection.
# Refitting could change probabilities and the chosen cutoff's behavior.
test_probabilities = delay_probabilities(best_model, test[features])
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

test_metrics = calculate_metrics(test[target], test_predictions)

pd.DataFrame([{
    "model": best_name,
    "threshold": best_threshold,
    **test_metrics,
}]).to_csv(
    results_dir / "test_results.csv",
    index=False,
)

print("\nTest results:")
for metric, value in test_metrics.items():
    print(f"{metric.capitalize()}: {value:.3f}")

matrix = confusion_matrix(
    test[target],
    test_predictions,
    labels=[0, 1],
)

confusion_df = pd.DataFrame(
    matrix,
    index=["actual_not_delayed", "actual_delayed"],
    columns=["predicted_not_delayed", "predicted_delayed"],
)
confusion_df.to_csv(results_dir / "confusion_matrix.csv")

print("\nConfusion matrix:")
print(confusion_df.to_string())

prediction_output = test[[
    "Date (MM/DD/YYYY)",
    "Scheduled departure time",
    "Destination Airport",
]].copy()

prediction_output["actual_is_delayed"] = test[target]
prediction_output["predicted_is_delayed"] = test_predictions
prediction_output["predicted_delay_probability"] = test_probabilities

prediction_output.to_csv(
    results_dir / "test_predictions.csv",
    index=False,
)

# Save both the pipeline and its cutoff.
# Future predictions must use this cutoff rather than model.predict().
joblib.dump(
    {
        "model": best_model,
        "threshold": best_threshold,
        "features": features,
        "delay_definition_minutes": 15,
    },
    models_dir / "flight_delay_classifier_bundle.joblib",
)

print(f"\nResults saved to: {results_dir}")
print(f"Model and cutoff saved to: {models_dir}")
