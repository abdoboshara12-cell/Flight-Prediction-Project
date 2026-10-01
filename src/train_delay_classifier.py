from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data/processed/flights_features.csv"
models_dir = project_dir / "models/classification"
results_dir = project_dir / "results/classification"

models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

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

df[delay_column] = pd.to_numeric(
    df[delay_column],
    errors="raise",
)

# Apply the original experiment's filtering.
df = df.dropna(subset=[
    "Destination Airport",
    "departure_hour",
    "day_of_week",
    delay_column,
    "flight_date",
])

if df[features].isna().any().any():
    raise ValueError(
        "Some engineered features are missing. "
        "Check feature_engineering.py."
    )

# 1 = at least 15 minutes late; 0 = less than 15 minutes late.
df[target] = (df[delay_column] >= 15).astype(int)

# Neither the actual delay nor the target is included in features.
df = df.sort_values("flight_date")

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
    raise ValueError("Training data must contain both delay classes.")

if validation[target].nunique() < 2:
    raise ValueError("Validation data must contain both delay classes.")

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]

for name, subset in [
    ("Training", train),
    ("Validation", validation),
    ("Testing", test),
]:
    print(
        f"{name} flights: {len(subset):,} | "
        f"Delayed: {subset[target].mean():.1%}"
    )

print(f"Validation starts: {validation_start:%Y-%m-%d}")
print(f"Testing starts: {test_start:%Y-%m-%d}")


def build_pipeline(model):
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


def calculate_metrics(actual, predicted):
    # Precision, recall, and F1 refer to the delayed class (1).
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

validation_results = []

# Use each classifier's default decision rule.
# No thresholds or model settings are chosen using test results.
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    metrics = calculate_metrics(y_validation, predictions)

    validation_results.append({
        "model": name,
        **metrics,
    })

    print(
        f"Validation F1: {metrics['f1']:.3f} | "
        f"Precision: {metrics['precision']:.3f} | "
        f"Recall: {metrics['recall']:.3f}"
    )

validation_df = pd.DataFrame(validation_results)

# Select using validation F1 only.
best_index = validation_df["f1"].idxmax()
best_name = validation_df.loc[best_index, "model"]
validation_df["selected"] = validation_df["model"] == best_name

validation_df.to_csv(
    results_dir / "validation_results.csv",
    index=False,
)

print("\nValidation results:")
print(validation_df.round(3).to_string(index=False))
print(f"\nSelected model: {best_name}")

# Refit the selected model on training + validation.
development = pd.concat([train, validation])
best_model = models[best_name]
best_model.fit(development[features], development[target])

# Evaluate the selected model on the latest flights.
test_predictions = best_model.predict(X_test)
test_metrics = calculate_metrics(y_test, test_predictions)

pd.DataFrame([{
    "model": best_name,
    **test_metrics,
}]).to_csv(
    results_dir / "test_results.csv",
    index=False,
)

print("\nSelected model's test results:")
for metric, value in test_metrics.items():
    print(f"{metric.capitalize()}: {value:.3f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["Less than 15 minutes late", "Delayed 15+ minutes"],
        zero_division=0,
    )
)

matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
)

confusion_df = pd.DataFrame(
    matrix,
    index=["actual_not_delayed", "actual_delayed"],
    columns=["predicted_not_delayed", "predicted_delayed"],
)

confusion_df.to_csv(results_dir / "confusion_matrix.csv")

print("Confusion matrix:")
print(confusion_df.to_string())

prediction_output = test[
    ["Date (MM/DD/YYYY)", "Scheduled departure time"] + features
].copy()

prediction_output["actual_delay_minutes"] = test[delay_column]
prediction_output["actual_is_delayed"] = y_test
prediction_output["predicted_is_delayed"] = test_predictions

# Find the probability column corresponding to delayed class 1.
delayed_index = list(best_model.classes_).index(1)
prediction_output["predicted_delay_probability"] = (
    best_model.predict_proba(X_test)[:, delayed_index]
)

prediction_output.to_csv(
    results_dir / "test_predictions.csv",
    index=False,
)

model_path = models_dir / "flight_delay_classifier.joblib"
joblib.dump(best_model, model_path)

pd.DataFrame({"feature": features}).to_csv(
    results_dir / "features_used.csv",
    index=False,
)

print(f"\nModel saved to: {model_path}")
print(f"Results saved to: {results_dir}")