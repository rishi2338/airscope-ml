# AirScope UI redesign preview

This branch refreshes the shared visual system used by the existing Streamlit app. It keeps the current app entry point and the existing data, clustering, forecasting, maps, analysis, validation, and export logic intact.

## What changed

- Deep charcoal and atmospheric teal/emerald palette with a subtle radial background.
- Stronger typography hierarchy and more deliberate spacing across all pages.
- Refined navigation states, focus rings, and keyboard-visible controls.
- More polished KPI, pollutant, AQI, forecast, city, and methodology cards.
- Lightweight hover and entrance transitions with reduced-motion support.
- Responsive spacing and grid adjustments for smaller screens.
- No new Python dependencies and no replacement of real data/model logic.

## Preview locally

From the repository root:

```powershell
python -m pip install -r air_pollution_dashboard/requirements.txt
python -m streamlit run air_pollution_dashboard/app.py
```

The main app remains the original entry point. The shared stylesheet is loaded from `air_pollution_dashboard/assets/styles.css`.

## Data integrity

AirScope can fall back to synthetic demo data when live CAMS/Open-Meteo data is unavailable. The app already labels this condition. Its particulate-only AQI uses US EPA breakpoints and is not India's CPCB AQI. Clustering labels describe relative pollutant profiles, not official health classifications.

## Review scope

This is a visual redesign branch. It deliberately does not replace `app.py` or change model/data algorithms. Test all pages with live data and with the offline fallback before merging or switching the deployed app.
