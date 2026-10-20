# AirScope dataset preparation guide

This guide covers practical checks for the CSV datasets used to explore air-pollution patterns. It is intentionally source-agnostic: confirm the documentation that came with your own dataset before relying on column names or units.

## 1. Choose one observation grain

Common dataset shapes include:

- **City daily** — one record per city and day.
- **City hourly** — one record per city and hour.
- **Station daily** — one record per monitoring station and day.
- **Station hourly** — one record per monitoring station and hour.
- **Station metadata** — station identifiers, names, and possibly coordinates.

Start with one table and one time resolution. Hourly and daily records represent different sampling grains; concatenating them without an explicit aggregation plan can double-count periods and bias patterns.

## 2. Inspect the schema

Before running analysis, inspect:

- Column names and their spelling/capitalization.
- Date or timestamp fields and parsing errors.
- City, state, station, and station-ID fields.
- Latitude and longitude, if geographic analysis is required.
- Pollutant columns and their measurement units.
- Duplicate records and invalid numeric values.

Typical pollutant names include `PM2.5`, `PM10`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, and `O3`. Actual datasets may use different spellings, abbreviations, or units. Map columns explicitly rather than assuming every dataset follows one schema.

## 3. Handle missing values carefully

Missingness is often informative: some stations do not measure every pollutant, and instruments may be offline for part of a period.

- Report missingness before cleaning.
- Do not replace every missing value with zero; zero is a measurement, not a generic missing-value marker.
- Fit imputers only on the training data, then reuse the fitted preprocessing for validation and inference.
- Consider dropping a feature only after assessing its usefulness and coverage.
- Keep a record of rows removed or transformed.

## 4. Keep units consistent

Measurements may use different units, including micrograms per cubic metre (`µg/m³`) and milligrams per cubic metre (`mg/m³`). Verify the source's units before comparing or scaling features. Do not assume that similarly named columns use the same unit across sources.

## 5. Prepare a clustering table

A typical clustering input contains one row per observation and numeric pollutant columns. Keep identifying fields such as city, station ID, and timestamp outside the feature matrix unless they are intentionally engineered into features.

A safe workflow is:

1. Select the pollutant features that exist in the chosen dataset.
2. Validate numeric types, units, and plausible ranges.
3. Decide how missing data will be handled.
4. Split data for evaluation when appropriate.
5. Fit preprocessing on training observations only.
6. Scale the selected features and fit K-Means.
7. Evaluate candidate cluster counts, including silhouette score when the sample and distance assumptions make it appropriate.
8. Save the feature order and preprocessing pipeline with the model.

K-Means is sensitive to feature scaling and outliers. Silhouette score is one diagnostic, not proof that a chosen number of clusters is scientifically correct.

## 6. Link station metadata

If readings and station metadata live in separate CSVs, join them using the documented station identifier. Check for unmatched IDs and duplicate metadata rows. Never join tables using row position or station name alone when a stable identifier is available.

## 7. Keep data provenance visible

For each analysis, record:

- Source and download date.
- Original file name and observation grain.
- Units and column mapping.
- Cleaning and imputation rules.
- Rows/features retained or removed.
- Model version and training date, if a model is used.

Do not present offline demo values, synthetic data, estimates, or model predictions as official live measurements. If sharing the project, check the dataset licence and avoid committing large or restricted source files.
