# AirScope — Air Pollution Pattern Discovery

AirScope is a Streamlit-based machine-learning dashboard for exploring air-pollution data and discovering patterns in pollutant measurements. It is designed to make environmental data easier to inspect through an interactive interface.

**Live demo:** https://airscope.streamlit.app

## What it does

- Loads air-quality data from a CSV file.
- Identifies available pollutant columns, such as PM2.5, PM10, NO₂ and NOx, when present in the dataset.
- Prepares data for analysis by handling missing values and scaling selected features.
- Uses machine-learning techniques, including K-Means clustering, to explore pollution patterns.
- Presents analysis through an interactive Streamlit dashboard.

> Results depend on the dataset and columns supplied. Clusters indicate similarities in the selected measurements; they are not, by themselves, official air-quality classifications or health advice.

## Technology stack

- **Python** — application and analysis logic
- **Streamlit** — interactive web dashboard
- **Pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **scikit-learn** — preprocessing and machine-learning algorithms

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/rishi2338/airscope-ml.git
cd airscope-ml
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains a `requirements.txt` file:

```bash
python -m pip install -r requirements.txt
```

Otherwise, install the core packages:

```bash
python -m pip install streamlit pandas numpy scikit-learn
```

### 4. Add your dataset

Place the dataset CSV where the application expects it and confirm the filename and column names match the code. For example, a project using `city_day.csv` should have that file available at the configured path.

### 5. Start the dashboard

Run the Streamlit entry-point file in the repository. If it is named `app.py`, use:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Dataset notes

AirScope expects structured air-quality data in CSV format. Available features depend on the source dataset. Check column names, missing values, units and date formats before interpreting results. Do not commit private, licensed or very large datasets unless you have permission to share them.

## Project structure

The exact file layout may change as the project develops. The main components are:

- **Dashboard** — Streamlit interface and user interactions
- **Data processing** — loading, cleaning and preparing pollution measurements
- **Machine learning** — feature preprocessing and clustering
- **Visual analysis** — presenting discovered patterns for exploration

## Limitations

- The quality of the results depends on the completeness and reliability of the input data.
- Clustering is exploratory and does not prove causation.
- Predictions or groupings should not replace official monitoring data or public-health guidance.
- A trained model should only be used when its expected features and preprocessing steps match the data supplied to it.

## Future improvements

Potential extensions include richer time-series analysis, geographic comparisons, model evaluation, downloadable reports and integration of a validated trained model.

## License

No license is specified here. Unless a license is added to the repository, reuse and redistribution may be restricted by default.

---

Built as a machine-learning project to support exploration of air-pollution patterns.
