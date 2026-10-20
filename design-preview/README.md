# AirScope 3D Motion Dashboard Preview

A standalone, dependency-free visual concept for the AirScope dashboard. It demonstrates the dark environmental-intelligence design, animated CSS 3D-style globe and data clusters, smooth page entrance, staggered KPI cards, chart-line drawing, responsive layout, and `prefers-reduced-motion` support.

## Preview locally

Open `index.html` in a modern browser. No npm, build step, or server is required.

## Data integrity

All displayed pollutant readings, AQI-like index, city comparison values, and trends are illustrative placeholders. They are not live monitoring data and must not be presented as official CPCB AQI. CSV selection only reads the file locally to count rows and columns; it does not train or load the AirScope ML model.

## Integration note

This is an isolated frontend concept in `design-preview/`. It does not replace or modify the existing Streamlit app entry point or its analysis logic. Integrate the visual treatment into the production Streamlit UI only after checking dependencies, data flow, accessibility, and mobile layout.
