from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_csv(relative_path, **kwargs):
    path = PROJECT_DIR / relative_path

    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the corresponding training script first."
        )

    return pd.read_csv(path, **kwargs)


def table(df):
    return df.to_markdown(index=False, floatfmt=".3f")


def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


def comparison_chart(
    df, columns, labels, title, ylabel, filename, score_chart=False
):
    plot_data = df.set_index("model")[columns].copy()
    plot_data.columns = labels

    fig, ax = plt.subplots(figsize=(10, 5))

    plot_data.plot.bar(
        ax=ax,
        color=["#94a3b8", "#2563eb"],
        rot=15,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.set_ylim(bottom=0)
    ax.legend(title="")

    if score_chart:
        ax.set_ylim(0, 1)

    save_plot(fig, filename)


def main():
    print("Reading saved experiment results...")

    basic = load_csv("results/model_results.csv")
    expanded = load_csv("results/improved/model_results.csv")

    default_validation = load_csv(
        "results/classification/validation_results.csv"
    )
    default_test = load_csv(
        "results/classification/test_results.csv"
    )

    tuned_validation = load_csv(
        "results/classification_tuned/validation_results.csv"
    )
    tuned_test = load_csv(
        "results/classification_tuned/test_results.csv"
    )
    thresholds = load_csv(
        "results/classification_tuned/threshold_results.csv"
    )
    predictions = load_csv(
        "results/classification_tuned/test_predictions.csv"
    )

    confusion = load_csv(
        "results/classification_tuned/confusion_matrix.csv",
        index_col=0,
    )

    confusion = confusion.loc[
        ["actual_not_delayed", "actual_delayed"],
        ["predicted_not_delayed", "predicted_delayed"],
    ]

    matrix = confusion.to_numpy(dtype=int)

    # Verify the matrix against the saved prediction labels.
    actual = predictions["actual_is_delayed"]
    predicted = predictions["predicted_is_delayed"]

    if not actual.isin([0, 1]).all():
        raise ValueError("Actual labels must be 0 or 1.")

    if not predicted.isin([0, 1]).all():
        raise ValueError("Predicted labels must be 0 or 1.")

    reconstructed = [
        [
            int(((actual == 0) & (predicted == 0)).sum()),
            int(((actual == 0) & (predicted == 1)).sum()),
        ],
        [
            int(((actual == 1) & (predicted == 0)).sum()),
            int(((actual == 1) & (predicted == 1)).sum()),
        ],
    ]

    if matrix.tolist() != reconstructed:
        raise ValueError(
            "Confusion matrix and saved predictions disagree. "
            "Rerun train_tuned_delay_classifier.py."
        )

    if len(predictions) == 0:
        raise ValueError("The saved test predictions are empty.")

    print("Creating graphs...")

    # 1. Regression comparison.
    regression_comparison = basic[[
        "model", "validation_mae_minutes"
    ]].merge(
        expanded[["model", "validation_mae_minutes"]],
        on="model",
        suffixes=("_basic", "_expanded"),
        validate="one_to_one",
    )

    regression_comparison["mae_reduction_minutes"] = (
        regression_comparison["validation_mae_minutes_basic"]
        - regression_comparison["validation_mae_minutes_expanded"]
    )

    comparison_chart(
        regression_comparison,
        [
            "validation_mae_minutes_basic",
            "validation_mae_minutes_expanded",
        ],
        ["Basic features", "Expanded features"],
        "Regression: validation MAE",
        "MAE in minutes — lower is better",
        "regression_comparison.png",
    )

    # 2. Classification comparison.
    classification_comparison = default_validation[[
        "model", "f1"
    ]].merge(
        tuned_validation[["model", "f1", "threshold"]],
        on="model",
        suffixes=("_default", "_tuned"),
        validate="one_to_one",
    )

    comparison_chart(
        classification_comparison,
        ["f1_default", "f1_tuned"],
        ["Default decision", "Selected cutoff"],
        "Classification: validation F1",
        "Delayed-class F1 — higher is better",
        "classification_comparison.png",
        score_chart=True,
    )

    # 3. Actual threshold sweep.
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, rows in thresholds.groupby("model"):
        if "Baseline" in name:
            continue

        rows = rows.sort_values("threshold")

        ax.plot(
            rows["threshold"],
            rows["f1"],
            marker="o",
            markersize=4,
            label=name,
        )

    baseline = thresholds[
        thresholds["model"] == "Always Delayed Baseline"
    ]

    if not baseline.empty:
        ax.axhline(
            float(baseline.iloc[0]["f1"]),
            color="#64748b",
            linestyle="--",
            label="Always delayed baseline",
        )

    ax.set_title("Probability cutoff versus validation F1")
    ax.set_xlabel("Probability cutoff")
    ax.set_ylabel("Delayed-class F1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()

    save_plot(fig, "threshold_sweep.png")

    # 4. Selected models' test metrics.
    metric_names = ["accuracy", "precision", "recall", "f1"]

    test_plot_data = pd.DataFrame({
        "Default experiment winner":
            default_test.iloc[0][metric_names].astype(float),
        "Tuned experiment winner":
            tuned_test.iloc[0][metric_names].astype(float),
    })

    fig, ax = plt.subplots(figsize=(9, 5))

    test_plot_data.plot.bar(
        ax=ax,
        color=["#94a3b8", "#2563eb"],
        rot=0,
    )

    ax.set_title("Selected models: exploratory test metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.15)
    ax.legend(title="")

    save_plot(fig, "test_metrics.png")

    # 5. Confusion matrix.
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0)

    outcome_names = [
        ["True negatives", "False positives"],
        ["False negatives", "True positives"],
    ]

    for row in range(2):
        for column in range(2):
            value = matrix[row, column]

            text_color = (
                "white"
                if value > matrix.max() * 0.6
                else "#0f172a"
            )

            ax.text(
                column,
                row,
                f"{outcome_names[row][column]}\n{value:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    ax.set_xticks([0, 1], ["Not delayed", "Delayed"])
    ax.set_yticks([0, 1], ["Not delayed", "Delayed"])
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Tuned classifier: test confusion matrix")

    fig.colorbar(image, ax=ax, label="Flights", shrink=0.8)
    save_plot(fig, "confusion_matrix.png")

    # Values used in the written report come from actual files.
    final = tuned_test.iloc[0]
    previous = default_test.iloc[0]

    f1_change = float(final["f1"] - previous["f1"])

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    total_test = int(matrix.sum())

    all_not_delayed_accuracy = (
        true_negative + false_positive
    ) / total_test

    test_dates = pd.to_datetime(
        predictions["Date (MM/DD/YYYY)"],
        format="%m/%d/%Y",
    )

    confusion_table = confusion.copy()
    confusion_table.index.name = "actual_class"
    confusion_table = confusion_table.reset_index()

    feature_path = (
        PROJECT_DIR / "results/classification/features_used.csv"
    )

    if feature_path.exists():
        feature_text = table(pd.read_csv(feature_path))
    else:
        feature_text = "See the feature lists in the training scripts."

    # Assemble the report in sections instead of one long string.
    sections = []

    sections.append(f"""# Flight Delay Prediction: Experiment Report

This report follows the project from basic regression to
threshold-tuned classification.

**All result tables and graphs are generated from saved CSV files.**
The explanations describe the implemented experiments.

## Final recorded result

The tuned experiment selected **{final['model']}**, using a
probability cutoff of **{final['threshold']:.2f}**.

| Test metric | Recorded value |
|---|---:|
| Precision | {final['precision']:.3f} |
| Recall | {final['recall']:.3f} |
| F1 | {final['f1']:.3f} |
| Accuracy | {final['accuracy']:.3f} |

Test F1 changed by **{f1_change:+.3f}** compared with the default
classification experiment's selected model.

The saved tuned predictions cover **{total_test:,} flights**, from
**{test_dates.min():%Y-%m-%d}** through
**{test_dates.max():%Y-%m-%d}**.

> Test scores are exploratory because the test period was inspected
> across experiments. Model and cutoff selection used validation data.
""")

    sections.append("""## 1. Data and cleaning

The source is a
[Bureau of Transportation Statistics departure report](https://www.transtats.bts.gov/ONTIME/Departures.aspx)
covering Southwest flights departing LAX.

[The cleaning script](../src/clean_data.py):

- Skips the seven report lines above the CSV header.
- Retains flight date, destination, scheduled time, and delay.
- Removes missing required values.
- Parses dates and scheduled departure times.
- Creates weekday and departure-hour features.
- Removes rows with unsuccessful date/time parsing.

Cleaned data is saved to `data/processed/flights_clean.csv`.

Actual departure delay is an outcome, never a model input.

## 2. Evaluation design

The training scripts sort flights chronologically and split
distinct flight dates:

- First 60% of dates: training.
- Next 20%: validation.
- Final 20%: testing.

All flights on the same date stay together. The percentages refer
to dates rather than flight counts.

Encoders and scalers are fitted within training pipelines.
""")

    sections.append(f"""## 3. Experiment 1: Basic regression

[Implementation](../src/train_basic_models.py)

### Goal

Predict departure delay in minutes using destination, weekday,
and scheduled departure hour.

### Approaches

- Median baseline: predict the training median delay.
- Linear Regression.
- Random Forest Regressor.

Models were selected using validation mean absolute error (MAE).
MAE measures average absolute prediction error in minutes.
Lower is better.

### Saved results

{table(basic)}

The `selected` column records the validation winner. Test scores
were not used to retroactively select a different model.
""")

    sections.append(f"""## 4. Experiment 2: Expanded features

[Feature engineering](../src/feature_engineering.py) |
[Training](../src/train_improved_models.py)

This experiment added:

- Year and month.
- Weekend indicator.
- Minute-level scheduled timing.
- Daily, weekly, and monthly sine/cosine features.

Cyclical features represent repeating patterns, making the end
and start of a cycle adjacent.

The expanded models use minutes since midnight. Departure minute
is saved in the feature CSV but is not a separate model input.

These features re-express existing date/time information.
They do not introduce weather or operational disruption data.

Output: `data/processed/flights_features.csv`.

### Validation comparison

{table(regression_comparison)}

Positive MAE reduction means the expanded features helped.

![Regression validation comparison](figures/regression_comparison.png)

### Expanded-feature results

{table(expanded)}

The regression models were refitted on training plus validation
after validation-based selection.
""")

    sections.append(f"""## 5. Experiment 3: Delay classification

[Implementation](../src/train_delay_classifier.py)

### New question

Will a flight depart **at least 15 minutes late**?

Class 1 means a departure delay of at least 15 minutes.
Class 0 includes early, on-time, and less-than-15-minute-late flights.

The actual delay and classification label are excluded from inputs.

### Models

- Majority-class baseline.
- Always-delayed baseline.
- Logistic Regression with balanced class weights.
- Random Forest Classifier with balanced class weights.

### Metrics

| Metric | Meaning |
|---|---|
| Precision | Fraction of delay warnings that were correct |
| Recall | Fraction of actual delayed flights caught |
| F1 | Harmonic mean of precision and recall |
| Accuracy | Fraction of all predictions that were correct |

Precision, recall, and F1 refer to the delayed class.

### Default validation results

{table(default_validation)}

### Selected model's test results

{table(default_test)}

This script refitted the selected model on training plus validation
before test evaluation.
""")

    sections.append(f"""## 6. Experiment 4: Threshold tuning

[Implementation](../src/train_tuned_delay_classifier.py)

A classifier score becomes a warning when it reaches a selected
probability cutoff.

The experiment searched cutoffs from 0.00 to 1.00 in steps of
0.05, using validation F1.

Baseline behavior stayed fixed. Exact F1 ties were broken using
higher precision, then higher cutoff.

Lower cutoffs generally catch more delays but create more false
alarms. With balanced class weights, classifier scores are not
assumed to be calibrated real-world probabilities.

### Best validation cutoff for each model

{table(tuned_validation)}

### Default versus tuned validation F1

{table(classification_comparison)}

![Classification comparison](figures/classification_comparison.png)

### Actual recorded threshold sweep

This graph reads every tested cutoff from
`results/classification_tuned/threshold_results.csv`.

![Threshold sweep](figures/threshold_sweep.png)

The selected model was not refitted after choosing its cutoff.
The fitted pipeline used for validation was also used for testing.
""")

    sections.append(f"""## 7. Final exploratory test evaluation

{table(tuned_test)}

![Test metric comparison](figures/test_metrics.png)

The graph compares the selected models from the default and tuned
experiments. Their procedures differ: the default winner was
refitted, while the tuned winner was not.

### Confusion matrix

{table(confusion_table)}

![Confusion matrix](figures/confusion_matrix.png)

The final model:

- Caught **{true_positive:,} delayed flights**.
- Missed **{false_negative:,} delayed flights**.
- Produced **{false_positive:,} false delay warnings**.
- Correctly identified **{true_negative:,} flights below the delay cutoff**.

Predicting every test flight not delayed would achieve
**{all_not_delayed_accuracy:.1%} accuracy**, while catching no delays.

Accuracy must therefore be considered alongside precision,
recall, and F1.

## 8. Recorded classification features

This list comes from the saved classification feature file.
The supplied default and tuned scripts use the same inputs.

{feature_text}
""")

    sections.append("""## 9. Implementation and artifacts

| File | Responsibility |
|---|---|
| [clean_data.py](../src/clean_data.py) | Data cleaning |
| [train_basic_models.py](../src/train_basic_models.py) | Basic regression |
| [feature_engineering.py](../src/feature_engineering.py) | Additional features |
| [train_improved_models.py](../src/train_improved_models.py) | Expanded regression |
| [train_delay_classifier.py](../src/train_delay_classifier.py) | Default classification |
| [train_tuned_delay_classifier.py](../src/train_tuned_delay_classifier.py) | Cutoff selection |
| [generate_experiment_report.py](../src/generate_experiment_report.py) | Report generation |

| Experiment | Results folder |
|---|---|
| Basic regression | `results/` |
| Expanded regression | `results/improved/` |
| Default classification | `results/classification/` |
| Tuned classification | `results/classification_tuned/` |

The final bundle is saved at:

`models/classification_tuned/flight_delay_classifier_bundle.joblib`

It stores the fitted pipeline, cutoff, feature names, and delay
definition. Future predictions must use the saved cutoff.

## 10. Lessons and limitations

- Baselines show whether trained models add value.
- More features do not automatically improve predictions.
- Regression MAE and classification F1 answer different questions.
- The cutoff changes the balance between false alarms and missed delays.
- Accuracy alone can hide failure to detect delayed flights.
- Weather, congestion, and incoming-aircraft delays were not included.
- Cancellations and diversions are not separate prediction targets.
- Pandemic-era and later travel patterns may differ.
- Test scores are exploratory because the period was inspected repeatedly.

The project demonstrates cleaning, chronological evaluation,
baseline comparisons, feature engineering, regression,
classification, cutoff selection, and model persistence.

## Updating this report

Run `python src/generate_experiment_report.py` from the project root.

The generator reads saved outputs and does not train models.
When experiments change, regenerate their results first.

The confusion matrix is checked against saved predictions.
This does not verify that all experiment outputs use the same
dataset version; keep results from consistent runs.
""")

    report_path = DOCS_DIR / "EXPERIMENT_REPORT.md"

    report_path.write_text(
        "\n\n".join(section.strip() for section in sections) + "\n",
        encoding="utf-8",
    )

    print(f"Created report: {report_path}")
    print(f"Created five graphs in: {FIGURES_DIR}")


if __name__ == "__main__":
    main()