# AirScope Redesign Preview

This folder entrypoint was added as a separate file so the currently deployed app is not overwritten.

## Run locally

From the repository root:

```powershell
python -m pip install -r air_pollution_dashboard/requirements.txt
python -m streamlit run air_pollution_dashboard/app_redesign.py
```

## What's included

- Dark environmental analytics interface with teal/cyan accents
- Overview cards, pollutant averages, and correlation heatmap
- K-Means clustering with PCA visualization
- CSV upload, basic numeric cleaning, charting, and CSV export
- Model transparency and synthetic-data disclosure
- Reduced-motion support

## Important

The redesign entrypoint defaults to synthetic demo records. These are not live readings or official monitoring-station measurements. This is a separate preview entrypoint; the existing `app.py` and deployed Streamlit configuration have not been replaced. Review the preview before switching the live app entrypoint.
