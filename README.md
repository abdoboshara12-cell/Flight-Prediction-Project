# Flight Delay Prediction

A machine learning project using 251,780 Southwest Airlines flights departing Los Angeles International Airport (LAX) to explore departure delays.

I tested regression models to predict delay minutes, then developed classifiers to identify flights departing at least 15 minutes late. The project documents each experiment, including baseline comparisons, feature engineering, and threshold tuning.

## Data source

Flight records were obtained from the [Bureau of Transportation Statistics](https://www.transtats.bts.gov/ONTIME/Departures.aspx), covering Southwest departures from LAX during 2017–2024.

## Skills developed

### Data preparation and feature engineering
- Cleaned and transformed flight records using Python and pandas.
- Created calendar, scheduled departure time, and cyclical features.
- Kept outcome information out of model inputs to prevent data leakage.

### Machine learning
- Built preprocessing and training pipelines with scikit-learn.
- Trained Linear Regression, Logistic Regression, and Random Forest models.
- Tuned classification thresholds to balance precision and recall.

### Evaluation and experimentation
- Used chronological training, validation, and test splits.
- Compared model performance against simple baselines.
- Evaluated MAE, precision, recall, F1, and confusion matrices.

### Visualization and reproducibility
- Generated graphs and Markdown reports from saved experiment results.
- Saved model pipelines and decision thresholds with joblib.
- Preserved experiment progression using Git and GitHub.

## Full project walkthrough

Open the **Contributing** tab for the complete work, including implementation details, experiment comparisons, graphs, results, and lessons learned.
