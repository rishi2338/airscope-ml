# Trained clustering model integration

AirScope's optional clustering artifacts must be produced by one consistent training pipeline. A model file by itself is not enough: the feature order, scaling, and missing-value handling used during training must be reproduced exactly during inference.

## Expected artifact paths

Place trusted artifacts in `air_pollution_dashboard/models/`:

```text
models/
├── air_pollution_model.pkl
├── air_pollution_scaler.pkl
├── feature_columns.pkl
└── cluster_profiles.pkl
```

The first three names are the expected model, scaler, and feature-list paths. The cluster-profile artifact is optional and should only be added if the app's current code uses it.

## Training contract

The training script should:

1. Select a documented set of pollutant columns.
2. Convert those columns to numeric values and handle missing values with a documented strategy.
3. Fit the preprocessing steps on the training data only.
4. Fit a `StandardScaler` on the same ordered feature matrix used by K-Means.
5. Train K-Means and evaluate plausible cluster counts. Record the evaluation method and limitations.
6. Persist the trained model, fitted scaler, and exact ordered feature names.
7. Record training data provenance, library versions, and a model version or date.

The feature list must preserve the training-time order. Do not sort or reconstruct the list independently at inference time.

## Inference contract

Before applying a trained model to a new CSV:

- Map source columns to the expected canonical names.
- Reorder the matrix to match the saved feature list.
- Apply the same missing-value transformation used during training.
- Apply the saved scaler; do not fit a new scaler on uploaded data.
- Call the trained model's prediction method on the transformed matrix.
- Preserve row identifiers separately so cluster assignments can be joined back to the correct observations.
- Handle missing required features with a clear validation message instead of silently substituting arbitrary values.

If missing-value imputation was part of training, its fitted object should also be saved and loaded. A robust long-term option is to persist a single scikit-learn Pipeline containing imputation, scaling, and clustering so these steps cannot drift apart. Any app changes needed for a pipeline should be implemented and tested as a coordinated code change.

## Security

Python pickle/joblib files are executable serialization formats. Loading an untrusted `.pkl` or `.joblib` file can execute arbitrary code. Only load artifacts produced by a trusted source, and do not accept arbitrary uploaded model files from users.

## Evaluation and interpretation

- Keep a holdout or otherwise appropriate validation strategy.
- Record silhouette score only when it is meaningful for the sample size and distance geometry.
- Inspect cluster sizes and pollutant profiles; tiny clusters may reflect outliers or data-quality issues.
- Check stability across seeds and reasonable preprocessing choices.
- Use descriptive labels such as “Cluster 0” until the profile has been reviewed.
- Do not call clusters official AQI categories, health-risk levels, or causal source attributions without a separate validated methodology.

## Before enabling a new model

- [ ] Artifact paths and versions are recorded.
- [ ] Feature names and order match the saved training metadata.
- [ ] Inference preprocessing matches training preprocessing.
- [ ] Missing required columns produce understandable errors.
- [ ] Test data is not used to fit the scaler or imputer.
- [ ] Cluster assignments align with the original rows.
- [ ] Outputs are checked against a known small fixture.
- [ ] Documentation distinguishes exploratory clusters from official air-quality classifications.
