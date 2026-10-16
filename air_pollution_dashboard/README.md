# AirScope — Air Pollution Pattern Discovery Dashboard

A Streamlit dashboard for exploring air-pollution clusters with a K-Means model.

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

- Overview dashboard with KPI cards and interactive charts
- Analyze a sample using pollutant inputs
- Upload a CSV and assign cluster labels in bulk
- Model information and instructions for replacing demo artifacts

## Demo model

The app can run without saved model files by creating a temporary synthetic K-Means model in memory. This synthetic data is only for testing the interface and is not official air-quality data.

Optional: run `python train_demo_model.py` to save demo model artifacts into `models/`.

## Replace with your friend's model

Place the matching artifacts in `models/`:

- `air_pollution_model.pkl` — fitted K-Means model
- `air_pollution_scaler.pkl` — fitted scaler from the same training run
- `feature_columns.pkl` — list of features in exact training order

Restart Streamlit after replacing the files. The app expects all three artifacts to be compatible. If your friend uses different feature names or a different feature list, update the app accordingly.

Only load pickle files from a trusted source because pickle files can execute code when loaded.

## Dataset note

For the India Air Quality dataset, the main CSV is commonly named `city_day.csv`. Before training, clean missing pollutant values and ensure the feature names and order used in training are saved. Do not use `AQI` or `AQI_Bucket` as model features if the goal is to discover pollutant patterns independently of AQI.
