# AirScope — Air Quality Intelligence

A responsive, dark-first Streamlit platform for 26 Indian cities. The original
StandardScaler/K-Means workflow is retained alongside CAMS data, maps and a
transparent Ridge/CAMS forecast. This is an exploratory tool, not an official
air-quality advisory.

## Run on Windows (VS Code)

1. Extract this ZIP file.
2. Open the extracted `air_pollution_dashboard` folder in VS Code.
3. Open Terminal in VS Code.
4. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

5. Start the app:

   ```powershell
   python -m streamlit run app.py
   ```

If `python` points to the wrong installation, select your intended Python interpreter in VS Code, or use its full path with `-m pip` and `-m streamlit`.

## Pages included

- **Overview:** city/all-cities context, rolling particulate AQI, six pollutant
  cards with 24-hour sparklines, 14/30/90-day history, 14-day outlook, map,
  watchlist, pollutant/city comparisons, current-reading export and model status.
- **Map:** interactive AQI/pollutant layers, forecast time travel, satellite maps,
  keyless Google embed, optional full Google marker map, opt-in animated outlook.
- **Forecast:** city or custom coordinates, 3–14-day horizon, pollutant selection,
  hourly rolling particulate AQI, CSV export, national ranking and MAE/skill table.
- **Patterns:** one lazily rendered view at a time—3D, 2D, PCA, cluster mix or map.
- **Analyze:** editable presets, model assignment, particulate AQI, radar and export.
- **Batch upload:** validated CSV, optional coordinate mapping, cluster charts and
  export. Limits: 10 MB / 50,000 rows; charts sample at most 2,000 rows.
- **Methodology:** sources, units, proxies, model limitations and configuration.

## Development and deployment

Use Python 3.11 or 3.12. Start Streamlit **from this directory**, so it reads
`.streamlit/config.toml` (including the dark theme and upload limit).
For Streamlit Community Cloud, select `air_pollution_dashboard/app.py` as the
entry point. Matching `.streamlit/config.toml` files at the repository root and
in the app directory support both launch locations; keep them in sync. Never
commit secrets.

```sh
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m mypy quality.py ui.py
python -m pytest -q
```

CI runs syntax, lint, boundary-module type checks and offline regression tests
on Python 3.11/3.12. Tests cover EPA band boundaries, nonfinite/invalid inputs,
all seven pages, key controls, forecast/blending invariants, API conversions and
trusted artifact roundtrips. They do not rely on external services.

No custom pre-commit hooks are present. `assets/styles.css` holds the responsive
design system; `ui.py` holds presentation helpers; `quality.py` holds pure AQI,
validation and spreadsheet-safe CSV functions. The existing pipeline remains in
`app.py`.

## Data, models and scientific limitations

- [Open-Meteo / Copernicus CAMS](https://open-meteo.com/en/docs/air-quality-api)
  supplies hourly **model estimates and archived forecasts**, not ground-station
  observations. API outages, incomplete location responses or unusable pollutant
  history trigger a clearly labelled synthetic fallback. Cache TTL is 30 minutes;
  the refresh button clears shared caches for all visitors.
- Features: `PM2.5, PM10, NO2, NOx, NH3, CO, SO2, O3`. Concentrations use µg/m³
  except CO (mg/m³). NOx = NO2 + NO (NO2 alone if NO is missing); NH3 is estimated
  as `6 + 0.12 * PM2.5` where unavailable. Clustering/forecasting uses these proxies.
- Daily modeling excludes incomplete days. Ridge models predict log-space
  changes using 1/2/3/7-day lags, rolling means, seasonality and coordinates.
  Recursive predictions are non-negative; the outlook is a 50/50 ML–CAMS blend
  on overlapping dates and ML-only afterward.
- The final 14 days are held out for 1/3/7-day MAE and skill against persistence.
  **These scores evaluate ML alone against CAMS history (or synthetic data in
  offline mode), not station data or the ensemble.** Negative skill is shown
  honestly. Shading is a heuristic ML-only range, not calibrated coverage.
- The particulate-only index follows the [EPA May 2024 AQI technical document](https://www.airnow.gov/sites/default/files/2024-05/aqi-technical-assistance-document-may2024.pdf):
  truncate PM2.5 to one decimal and PM10 to integers, interpolate within discrete
  bands, round half up and take the larger sub-index. Display is capped at 500.
  This intentionally corrects the previous continuous-knot approximation, so
  some values differ slightly. It is **not** India's CPCB AQI, AirNow NowCast or a
  complete gas-inclusive AQI.
- Current values use rolling 24-hour means; daily forecasts use calendar days.
  All-cities values are unweighted city averages, not population-weighted
  national statistics. Gas pollutants appear in cards/charts but not the index.
- Custom coordinates use local API timezones. Forecast models are trained on
  Indian cities and may generalize poorly to other regions.
- Invalid CSV pollutant rows are rejected with row numbers rather than silently
  removed. Invalid coordinates remain in the CSV but are omitted from the map.
  Text that could execute as a spreadsheet formula is escaped in batch exports.
- The free Open-Meteo endpoint is for non-commercial use. Review
  [its terms](https://open-meteo.com/en/terms) before commercial deployment.

## Google Maps

No key is required for single-city Google embeds or the interactive fallback.
For the full multi-city Google map, enable the Maps JavaScript API and provide a
**browser key** using the sidebar, `GOOGLE_MAPS_API_KEY` environment variable or
Streamlit secrets. Restrict HTTP referrers and API scope in Google Cloud; the
key is necessarily visible to the browser. Google billing/quotas still apply.

## Demo model

The app can run without saved model files by creating a temporary synthetic K-Means model in memory. This synthetic data is only for testing the interface and is not official air-quality data.

Optional: run `python train_demo_model.py` to save demo model artifacts into `models/`.
Generated artifacts carry a synthetic-model marker and continue to display as
demo models after loading. The helper does not write artifacts when imported.

## Replace with your friend's model

Place the matching artifacts in `models/`:

- `air_pollution_model.pkl` — fitted K-Means model
- `air_pollution_scaler.pkl` — fitted scaler from the same training run
- `feature_columns.pkl` — list of features in exact training order
- `cluster_profiles.pkl` — optional dictionary of cluster IDs to `{"label": "..."}`

Restart Streamlit after replacing the files. The app expects all three artifacts to be compatible. If your friend uses different feature names or a different feature list, update the app accordingly.
Missing or incompatible bundles produce a visible warning and demo fallback.
Use matching scikit-learn versions when exchanging pickles. No real trained
artifacts or datasets are included in this repository.

Only load pickle files from a trusted source because pickle files can execute code when loaded.

## Dataset note

For the India Air Quality dataset, the main CSV is commonly named `city_day.csv`. Before training, clean missing pollutant values and ensure the feature names and order used in training are saved. Do not use `AQI` or `AQI_Bucket` as model features if the goal is to discover pollutant patterns independently of AQI.
