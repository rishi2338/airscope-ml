<div align="center">

# AirScope

### Discover the patterns behind air pollution.

An interactive environmental analytics dashboard for exploring pollutant measurements, comparing locations, studying trends, and investigating machine-learning clusters.

[**Open the live dashboard**](https://airscope.streamlit.app) · [**Explore the code**](https://github.com/rishi2338/airscope-ml/tree/airscope-2) · [**Report a bug**](https://github.com/rishi2338/airscope-ml/issues)

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/status-active%20development-2E8B57)

</div>

---

## The idea

Air pollution data is complex, spread across pollutants, locations, and time. **AirScope** brings those dimensions into one workspace so people can inspect measurements, explore relationships, and discover groups of observations with similar pollutant profiles.

The project is built with Python and Streamlit, with interactive visual analysis and an extensible path toward a separately trained clustering model.

> **Interpretation matters:** a cluster is a group of observations that resemble one another under the selected features. It is not an official AQI category, a causal explanation, or medical advice. Always check the data source, timestamp, units, and coverage before drawing conclusions.

## What you can explore

- **Pollutant profiles** — inspect available measurements such as PM2.5, PM10, NO₂, NOx, NH₃, CO, SO₂, and O₃ when present.
- **Location comparisons** — compare supported city or station observations.
- **Time-series exploration** — examine historical patterns and trends where valid timestamped data is available.
- **Pattern discovery** — use K-Means and dimensionality-reduction techniques to explore similarity between observations.
- **Data quality** — inspect missing measurements and validate inputs before interpreting results.
- **Exportable analysis** — download supported tabular results for further work.
- **Responsive visual design** — a dark, high-contrast interface with motion designed to respect reduced-motion preferences.

Available views and outputs depend on the current data source, dataset schema, and configured model artifacts. A feature should not be interpreted as live or official unless the dashboard explicitly identifies its source and status.

## Quick start

### 1. Get the project

```bash
git clone https://github.com/rishi2338/airscope-ml.git
cd airscope-ml
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dashboard dependencies

```bash
cd air_pollution_dashboard
python -m pip install -r requirements.txt
```

If your checkout does not contain a requirements file in that directory, install the core packages and any additional packages imported by the app:

```bash
python -m pip install streamlit pandas numpy scikit-learn plotly requests
```

### 4. Run AirScope

From the `air_pollution_dashboard` directory:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, typically [http://localhost:8501](http://localhost:8501).

## Data you can use

AirScope works with structured air-quality observations. Example datasets may contain city-level daily or hourly readings, station-level readings, and station metadata.

Before analysis:

1. Confirm the dataset's source, licence, time zone, and units.
2. Use a consistent time resolution. Do not combine hourly and daily records as if they were equivalent observations.
3. Check missing values and pollutant column names.
4. Keep station identifiers and coordinates aligned with the correct observations.
5. Separate measured data from estimates, generated demo data, and model outputs.
6. Avoid committing large, restricted, or personally sensitive files to Git.

See the [dataset guide](docs/DATASET_GUIDE.md) for practical preparation notes.

## Machine-learning model integration

AirScope is designed to support a trained K-Means workflow. The app currently looks for these artifacts under `air_pollution_dashboard/models/`:

```text
models/
├── air_pollution_model.pkl
├── air_pollution_scaler.pkl
├── feature_columns.pkl
└── cluster_profiles.pkl   # optional profile artifact
```

The model, scaler, and exact feature order must come from the same training pipeline. Inference preprocessing must match training preprocessing; missing-value handling cannot safely be guessed at prediction time. Only load trusted pickle files, because unpickling an untrusted file can execute arbitrary code.

Read [ML model integration](docs/ML_MODEL_INTEGRATION.md) before adding or replacing artifacts.

## Project layout

```text
airscope-ml/
├── air_pollution_dashboard/
│   ├── app.py                 # Streamlit application
│   ├── ui.py                  # Shared interface components
│   ├── quality.py             # Measurement validation and quality helpers
│   ├── assets/
│   │   └── styles.css         # Theme, responsive styling, motion
│   └── models/                # Optional trusted trained artifacts
├── docs/
│   ├── DATASET_GUIDE.md
│   └── ML_MODEL_INTEGRATION.md
└── README.md
```

The layout above describes the intended key files; your checkout may contain additional files.

## Data integrity and limitations

- Results are only as reliable as the input data and its coverage.
- Missing values, inconsistent units, duplicate records, and changing monitoring methods can distort comparisons.
- Clustering is exploratory; it does not establish causation or automatically assign official pollution categories.
- Forecasts are model outputs, not guarantees of future conditions.
- Always distinguish official observations from offline fallbacks, synthetic examples, estimates, and predictions.
- Do not use this project as a replacement for official monitoring services or public-health advice.

## Contributing

Ideas, reproducible bug reports, dataset-schema improvements, and tested fixes are welcome. When reporting an issue, include the steps to reproduce it, the expected behavior, the actual behavior, and a small synthetic sample of the relevant schema. Do not upload credentials, secrets, or restricted datasets.

## Roadmap

- [ ] Validate end-to-end model loading with the final trained artifacts.
- [ ] Add repeatable tests for pollutant-column mapping and missing-data handling.
- [ ] Improve model evaluation and cluster interpretation.
- [ ] Add clearer source and freshness indicators for each data mode.
- [ ] Expand accessible visual and downloadable reporting.

## License

A licence has not yet been specified in this repository. Until one is added, do not assume the project is open for unrestricted reuse or redistribution.

---

<div align="center">

**AirScope** · Environmental data, made more explorable.

</div>
