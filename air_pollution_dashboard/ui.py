"""Small presentation helpers; no data fetching or model side effects."""
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st


def configure_style():
    css = Path(__file__).with_name("assets").joinpath("styles.css").read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def sparkline(values, color="#38bdf8"):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return '<div class="sparkline"></div>'
    x = np.linspace(0, 140, len(values))
    span = np.ptp(values)
    y = 25 - 22 * (values - values.min()) / (span if span else 1)
    points = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(x, y))
    return (
        '<svg class="sparkline" viewBox="0 0 140 29" preserveAspectRatio="none" '
        'aria-hidden="true">'
        f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8" '
        'vector-effect="non-scaling-stroke"/></svg>'
    )


def pollutant_cards(current, hourly):
    names = {
        "PM2.5": ("Fine particles", "#38bdf8"),
        "PM10": ("Coarse particles", "#60a5fa"),
        "NO2": ("Nitrogen dioxide", "#2dd4bf"),
        "SO2": ("Sulphur dioxide", "#a5b4fc"),
        "CO": ("Carbon monoxide", "#fbbf24"),
        "O3": ("Ozone", "#67e8f9"),
    }
    cards = []
    for feature, (name, color) in names.items():
        value = current.get(feature, np.nan)
        display = f"{value:.2f}" if feature == "CO" else f"{value:.1f}"
        if not np.isfinite(value):
            display = "—"
        unit = "mg/m³" if feature == "CO" else "µg/m³"
        history = hourly[feature] if feature in hourly else []
        label = feature.replace("2.5", "₂.₅").replace("10", "₁₀").replace("2", "₂").replace("3", "₃")
        delay = (len(cards) % 6) * 55
        cards.append(
            f'<div class="pollutant reveal-card" style="--reveal-delay:{delay}ms">'
            f'<div class="pollutant-name">{label}<small>{name}</small></div>'
            f'<div class="pollutant-value">{display}</div><div class="pollutant-unit">{unit}</div>'
            f'{sparkline(history, color)}<div class="sparkline-caption">Past 24 hours · model estimate</div>'
            '</div>'
        )
    # st.html's HTML-only sanitizer removes SVG; this markup uses only trusted labels and numeric values.
    st.markdown('<div class="pollutant-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def forecast_cards(future, category):
    cards = []
    for _, row in future.head(7).iterrows():
        label, color, _ = category(row["AQI"])
        delay = len(cards) * 45
        cards.append(
            f'<div class="fc-card reveal-card" style="--reveal-delay:{delay}ms">'
            f'<div class="fc-day">{row["date"]:%a · %d %b}</div>'
            f'<div class="fc-aqi" style="color:{color}">{row["AQI"]:.0f}</div>'
            f'<span class="aqi-badge" style="background:{color}">{escape(label)}</span>'
            f'<div class="fc-meta">PM₂.₅ {row["PM2.5"]:.0f} µg/m³</div></div>'
        )
    st.html('<div class="forecast-grid">' + "".join(cards) + "</div>")
