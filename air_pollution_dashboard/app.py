import json
import os
import pickle
from datetime import timedelta
from html import escape
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from quality import compute_aqi, csv_export, validate_measurements
from ui import configure_style, forecast_cards, pollutant_cards
from dataset_lab import page_dataset_lab

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "air_pollution_model.pkl"
SCALER_PATH = MODEL_DIR / "air_pollution_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
PROFILES_PATH = MODEL_DIR / "cluster_profiles.pkl"
FORECAST_MODEL_PATH = MODEL_DIR / "air_pollution_forecast.pkl"

st.set_page_config(
    page_title="AirScope 2 | Environmental Intelligence",
    page_icon=":material/air:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Design system — premium AirScope theme
# ============================================================================
configure_style()

# Load the shared stylesheet at the app entry point so the premium theme is
# applied consistently even if the UI helper's stylesheet path changes.
PREMIUM_THEME_PATH = APP_DIR / "assets" / "styles.css"
if PREMIUM_THEME_PATH.is_file():
    st.html(f"<style>{PREMIUM_THEME_PATH.read_text(encoding='utf-8')}</style>")

# ============================================================================
# Constants
# ============================================================================
DEFAULT_FEATURES = ["PM2.5", "PM10", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
UNITS = {"PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³", "NOx": "µg/m³", "NH3": "µg/m³",
         "CO": "mg/m³", "SO2": "µg/m³", "O3": "µg/m³"}
CLUSTER_COLORS = ["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#fb7185"]
PAST_DAYS = 92
CAMS_DAYS = 5
MAX_HORIZON = 14

# (city, state, lat, lon, typical PM2.5 used only for the offline fallback)
CITIES = [
    ("Delhi", "Delhi", 28.6139, 77.2090, 110), ("Mumbai", "Maharashtra", 19.0760, 72.8777, 45),
    ("Kolkata", "West Bengal", 22.5726, 88.3639, 65), ("Chennai", "Tamil Nadu", 13.0827, 80.2707, 32),
    ("Bengaluru", "Karnataka", 12.9716, 77.5946, 30), ("Hyderabad", "Telangana", 17.3850, 78.4867, 40),
    ("Ahmedabad", "Gujarat", 23.0225, 72.5714, 60), ("Pune", "Maharashtra", 18.5204, 73.8567, 42),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873, 70), ("Lucknow", "Uttar Pradesh", 26.8467, 80.9462, 95),
    ("Kanpur", "Uttar Pradesh", 26.4499, 80.3319, 100), ("Patna", "Bihar", 25.5941, 85.1376, 105),
    ("Chandigarh", "Chandigarh", 30.7333, 76.7794, 60), ("Gurugram", "Haryana", 28.4595, 77.0266, 105),
    ("Amritsar", "Punjab", 31.6340, 74.8723, 75), ("Bhopal", "Madhya Pradesh", 23.2599, 77.4126, 55),
    ("Visakhapatnam", "Andhra Pradesh", 17.6868, 83.2185, 38), ("Thiruvananthapuram", "Kerala", 8.5241, 76.9366, 22),
    ("Guwahati", "Assam", 26.1445, 91.7362, 55), ("Kochi", "Kerala", 9.9312, 76.2673, 25),
    ("Nagpur", "Maharashtra", 21.1458, 79.0882, 50), ("Indore", "Madhya Pradesh", 22.7196, 75.8577, 50),
    ("Varanasi", "Uttar Pradesh", 25.3176, 82.9739, 90), ("Coimbatore", "Tamil Nadu", 11.0168, 76.9558, 28),
    ("Bhubaneswar", "Odisha", 20.2961, 85.8245, 48), ("Shillong", "Meghalaya", 25.5788, 91.8933, 20),
]
CITY_INDEX = {c[0]: c for c in CITIES}

AQI_CATEGORIES = [
    (50, "Good", "#22c55e", "Air quality is satisfactory. Enjoy outdoor activities."),
    (100, "Moderate", "#eab308", "Acceptable. Unusually sensitive people should limit long outdoor exertion."),
    (150, "Unhealthy for Sensitive Groups", "#f97316", "Children, older adults and people with lung or heart disease should reduce prolonged outdoor exertion."),
    (200, "Unhealthy", "#ef4444", "Everyone may feel effects. Wear a mask outdoors and reduce strenuous activity."),
    (300, "Very Unhealthy", "#a855f7", "Health alert. Avoid outdoor exertion; keep windows closed and use air purifiers."),
    (10_000, "Hazardous", "#9f1239", "Emergency conditions. Stay indoors and avoid all outdoor activity."),
]
AQI_COLOR_MAP = {name: color for _, name, color, _ in AQI_CATEGORIES}
AQI_ORDER = [name for _, name, _, _ in AQI_CATEGORIES]


def aqi_category(aqi):
    if aqi is None or pd.isna(aqi):
        return ("Unknown", "#64748b", "No data available.")
    for limit, name, color, advice in AQI_CATEGORIES:
        if aqi <= limit:
            return name, color, advice
    return AQI_CATEGORIES[-1][1:]


def badge(aqi):
    name, color, _ = aqi_category(aqi)
    return f'<span class="aqi-badge" style="background:{color}">{name}</span>'


def aqi_legend_html():
    items = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c, _ in AQI_CATEGORIES)
    return f'<div class="legend">{items}</div>'


def style_fig(fig, height=420, legend=True, margin=None):
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#a9bacb", size=11),
        margin=margin or dict(l=10, r=10, t=48, b=10), showlegend=legend,
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", yanchor="bottom", y=1.02, x=0),
        hoverlabel=dict(bgcolor="#161e28", bordercolor="#354454", font_color="#e9f0f7"),
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family="Space Grotesk, sans-serif", size=16, color="#f1f5f9"))
    else:
        fig.update_layout(title=None)
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.15)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.15)")
    return fig


PLOT_CONFIG = {"displaylogo": False, "scrollZoom": True}

# ============================================================================
# Clustering model (loads saved artifacts, otherwise a synthetic demo model)
# ============================================================================

def load_artifacts():
    needed = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    if not all(p.exists() for p in needed):
        if any(p.exists() for p in needed):
            raise ValueError("Incomplete artifact bundle")
        return None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        features = pickle.load(f)
    profiles = None
    if PROFILES_PATH.exists():
        with open(PROFILES_PATH, "rb") as f:
            profiles = pickle.load(f)
    features = list(features)
    if not features or any(not isinstance(f, str) for f in features) or len(set(features)) != len(features):
        raise ValueError("Invalid feature list")
    if not hasattr(model, "cluster_centers_") or not hasattr(scaler, "transform"):
        raise ValueError("Expected fitted K-Means and scaler")
    if model.cluster_centers_.shape[1] != len(features) or scaler.n_features_in_ != len(features):
        raise ValueError("Artifact feature dimensions do not match")
    probe = scaler.inverse_transform(model.cluster_centers_[:1])
    if not np.isfinite(probe).all():
        raise ValueError("Invalid centroids")
    model.predict(scaler.transform(pd.DataFrame(probe, columns=features)))
    return model, scaler, features, profiles


def make_demo_data(n=650, seed=19):
    """Synthetic data for interface testing only; not real observed pollution."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [18, 42, 12, 20, 10, 0.45, 7, 22],
        [65, 115, 32, 48, 20, 1.15, 15, 35],
        [145, 245, 58, 82, 34, 2.3, 30, 48],
    ], dtype=float)
    sizes = [int(n * .45), int(n * .38), n - int(n * .45) - int(n * .38)]
    blocks = []
    for center, size in zip(centers, sizes):
        std = np.array([9, 18, 7, 10, 5, .18, 4, 8])
        blocks.append(rng.normal(center, std, size=(size, len(center))))
    data = np.clip(np.vstack(blocks), [0, 0, 0, 0, 0, .05, 0, 0], None)
    return pd.DataFrame(data, columns=DEFAULT_FEATURES)


def infer_profile_labels(data, labels):
    """Tentative descriptive labels based on relative PM2.5 cluster means."""
    means = data.assign(cluster=labels).groupby("cluster")["PM2.5"].mean().sort_values()
    mapping = {}
    for rank, (cid, _) in enumerate(means.items()):
        if len(means) == 1:
            name = "Single pollution profile"
        elif rank == 0:
            name = "Lower pollutant profile"
        elif rank == len(means) - 1:
            name = "Higher pollutant profile"
        else:
            name = "Moderate pollutant profile"
        mapping[int(cid)] = name
    return mapping


@st.cache_resource(show_spinner=False)
def get_cluster_engine():
    error = ""
    try:
        artifacts = load_artifacts()
    except Exception:
        artifacts = None
        error = "Saved model artifacts could not be loaded or are incompatible. Using a demo model. Check their features and scikit-learn version."
    if artifacts:
        model, scaler, features, profiles = artifacts
        return model, scaler, features, profiles, bool(getattr(model, "_airscope_demo", False)), error
    demo = make_demo_data()
    scaler = StandardScaler().fit(demo[DEFAULT_FEATURES])
    model = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(demo[DEFAULT_FEATURES]))
    return model, scaler, DEFAULT_FEATURES, None, True, error


model, scaler, features, profiles, demo_mode, artifact_error = get_cluster_engine()


def assign_clusters(frame):
    scaled = scaler.transform(validate_measurements(frame, features))
    return model.predict(scaled), scaled


def can_cluster(frame):
    return all(f in frame.columns for f in features)


@st.cache_data(show_spinner=False)
def cluster_label_map():
    ref = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
    mapping = infer_profile_labels(ref, np.arange(len(ref))) if "PM2.5" in ref else {}
    if isinstance(profiles, dict):
        for cid, prof in profiles.items():
            if isinstance(prof, dict) and prof.get("label"):
                mapping[int(cid)] = prof["label"]
    return mapping


def cluster_name(cid):
    return cluster_label_map().get(int(cid), f"Profile {int(cid)}")


def prepare_input(values):
    return validate_measurements(pd.DataFrame([values]), features)


# ============================================================================
# Live air-quality data (Open-Meteo / Copernicus CAMS, no API key required)
# ============================================================================
OPEN_METEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OM_VARS = ["pm2_5", "pm10", "nitrogen_dioxide", "nitrogen_monoxide", "ammonia",
           "carbon_monoxide", "sulphur_dioxide", "ozone"]


def _om_to_features(hourly):
    h = pd.DataFrame(hourly)
    h["time"] = pd.to_datetime(h["time"], errors="raise")
    if h.empty or h["time"].isna().any() or h["time"].duplicated().any():
        raise ValueError("Missing or duplicate hourly timestamps")
    h = h.sort_values("time").reset_index(drop=True)
    if not h["time"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Hourly data must have a continuous hourly time axis")
    for var in OM_VARS:
        if var not in h:
            h[var] = np.nan
        h[var] = pd.to_numeric(h[var], errors="coerce").replace([np.inf, -np.inf], np.nan)
        h.loc[h[var] < 0, var] = np.nan
    out = pd.DataFrame({"time": h["time"]})
    out["PM2.5"] = h["pm2_5"]
    out["PM10"] = h["pm10"]
    out["NO2"] = h["nitrogen_dioxide"]
    out["NOx"] = h["nitrogen_dioxide"] + h["nitrogen_monoxide"].fillna(0)
    # CAMS has no ammonia outside Europe; estimate from PM2.5 so the 8-feature model still works.
    nh3_est = 6 + 0.12 * h["pm2_5"]
    out["NH3"] = h["ammonia"].where(h["ammonia"].notna(), nh3_est)
    out["CO"] = h["carbon_monoxide"] / 1000.0  # µg/m³ -> mg/m³
    out["SO2"] = h["sulphur_dioxide"]
    out["O3"] = h["ozone"]
    out["nh3_estimated"] = h["ammonia"].isna()
    return out


@st.cache_data(ttl=1800, show_spinner=False, max_entries=128)
def fetch_open_meteo(lats, lons, past_days=PAST_DAYS, forecast_days=CAMS_DAYS):
    if not lats or len(lats) != len(lons):
        raise ValueError("Latitude and longitude lists must match")
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "hourly": ",".join(OM_VARS),
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    payload = payload if isinstance(payload, list) else [payload]
    if len(payload) != len(lats):
        raise ValueError("Incomplete location response from CAMS")
    frames, offsets = [], []
    for loc in payload:
        frame = _om_to_features(loc["hourly"])
        frame[DEFAULT_FEATURES] = frame[DEFAULT_FEATURES].interpolate(limit=6, limit_direction="both")
        if len(frame) < 24 * 30 or not np.isfinite(frame[DEFAULT_FEATURES]).all().all():
            raise ValueError("Insufficient pollutant history from CAMS")
        frames.append(frame)
        offsets.append(int(loc.get("utc_offset_seconds", 0)))
    return frames, offsets


def synthetic_hourly(lat, lon, base_pm25, seed, now_local):
    """Offline fallback: plausible synthetic series. Not real measurements."""
    rng = np.random.default_rng(seed)
    start = (now_local - timedelta(days=PAST_DAYS)).normalize()
    times = pd.date_range(start, periods=(PAST_DAYS + CAMS_DAYS) * 24, freq="h")
    t = np.arange(len(times))
    doy = times.dayofyear.values
    seasonal = 1 + 0.45 * np.cos(2 * np.pi * (doy - 15) / 365)
    diurnal = 1 + 0.25 * np.cos(2 * np.pi * (times.hour.values - 21) / 24)
    weather = np.exp(np.convolve(rng.normal(0, .35, len(t)), np.ones(36) / 36, mode="same") * 3)
    pm25 = base_pm25 * seasonal * diurnal * weather
    ratio = rng.uniform(1.6, 2.1)
    out = pd.DataFrame({"time": times})
    out["PM2.5"] = pm25
    out["PM10"] = pm25 * ratio * rng.lognormal(0, .08, len(t))
    out["NO2"] = (12 + 0.22 * pm25) * diurnal * rng.lognormal(0, .12, len(t))
    out["NOx"] = out["NO2"] * rng.uniform(1.3, 1.7)
    out["NH3"] = 6 + 0.12 * pm25
    out["CO"] = (0.25 + 0.008 * pm25) * rng.lognormal(0, .1, len(t))
    out["SO2"] = (5 + 0.08 * pm25) * rng.lognormal(0, .15, len(t))
    out["O3"] = (30 + 25 * np.sin(2 * np.pi * (times.hour.values - 9) / 24).clip(0)) * rng.lognormal(0, .1, len(t))
    out["nh3_estimated"] = True
    return out


@st.cache_data(ttl=1800, show_spinner="Fetching live air-quality data for Indian cities...")
def load_city_hourly():
    lats = tuple(c[2] for c in CITIES)
    lons = tuple(c[3] for c in CITIES)
    try:
        frames, offsets = fetch_open_meteo(lats, lons)
        source = "live"
        now_local = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(seconds=offsets[0])).floor("h")
    except Exception:
        source = "synthetic"
        now_local = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(hours=5.5)).floor("h")
        frames = [synthetic_hourly(c[2], c[3], c[4], i, now_local) for i, c in enumerate(CITIES)]
    parts = []
    for city, frame in zip(CITIES, frames):
        f = frame.copy()
        f.insert(0, "city", city[0])
        f["lat"], f["lon"] = city[2], city[3]
        parts.append(f)
    hourly = pd.concat(parts, ignore_index=True)
    hourly[DEFAULT_FEATURES] = hourly.groupby("city")[DEFAULT_FEATURES].transform(lambda s: s.interpolate(limit_direction="both"))
    return hourly, source, now_local


def to_daily(hourly):
    daily = (hourly.assign(date=hourly["time"].dt.normalize())
             .groupby(["city", "date"], as_index=False)
             .agg({**{f: "mean" for f in DEFAULT_FEATURES}, "lat": "first", "lon": "first"}))
    counts = hourly.assign(date=hourly["time"].dt.normalize()).groupby(["city", "date"]).size()
    complete = counts[counts == 24].index
    daily = daily.set_index(["city", "date"]).loc[lambda d: d.index.isin(complete)].reset_index()
    daily["AQI"] = [compute_aqi(a, b) for a, b in zip(daily["PM2.5"], daily["PM10"])]
    return daily


def city_snapshot(hourly, now_local):
    rows = []
    for city, g in hourly.groupby("city", sort=False):
        obs = g[g["time"] <= now_local]
        last24 = obs[obs["time"] > now_local - pd.Timedelta(hours=24)]
        prev24 = obs[(obs["time"] <= now_local - pd.Timedelta(hours=24)) & (obs["time"] > now_local - pd.Timedelta(hours=48))]
        rec = {"city": city, "state": CITY_INDEX[city][1], "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0]}
        rec.update(last24[DEFAULT_FEATURES].mean().to_dict())
        rec["AQI"] = compute_aqi(rec["PM2.5"], rec["PM10"])
        prev_aqi = compute_aqi(prev24["PM2.5"].mean(), prev24["PM10"].mean())
        rec["AQI_change"] = rec["AQI"] - prev_aqi if not pd.isna(prev_aqi) else 0.0
        rec["PM2.5_now"] = obs["PM2.5"].iloc[-1] if len(obs) else np.nan
        rec["nh3_estimated"] = bool(g["nh3_estimated"].any())
        rows.append(rec)
    snap = pd.DataFrame(rows)
    snap["Category"] = [aqi_category(a)[0] for a in snap["AQI"]]
    snap["Color"] = [aqi_category(a)[1] for a in snap["AQI"]]
    if can_cluster(snap):
        snap["Cluster"] = assign_clusters(snap)[0]
        snap["Profile"] = [cluster_name(c) for c in snap["Cluster"]]
    return snap


# ============================================================================
# Forecasting model (Ridge regression on lagged daily pollutant changes)
# ============================================================================
LAGS = (1, 2, 3, 7)


def _design_row(logv, t, date, lat, lon):
    """Features for predicting day index t using log1p history logv[:t]."""
    parts = [logv[t - lag] for lag in LAGS]
    parts.append(logv[t - 7:t].mean(axis=0))
    parts.append(logv[t - 3:t].mean(axis=0))
    doy = date.dayofyear
    parts.append(np.array([date.dayofweek, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), lat, lon]))
    return np.concatenate(parts)


def _make_regressor():
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0))


def _series_states(groups):
    states = []
    for name, g in groups:
        g = g.sort_values("date")
        states.append({"name": name, "logv": np.log1p(g[DEFAULT_FEATURES].clip(lower=0).to_numpy()),
                       "last": g["date"].iloc[-1], "lat": g["lat"].iloc[0], "lon": g["lon"].iloc[0]})
    return states


def _recursive(models, states, horizon):
    """Roll every series forward `horizon` days; each model predicts the day-over-day log change."""
    out = [[] for _ in states]
    for h in range(1, horizon + 1):
        X = np.asarray([_design_row(s["logv"], len(s["logv"]), s["last"] + pd.Timedelta(days=h), s["lat"], s["lon"])
                        for s in states])
        last = np.asarray([s["logv"][-1] for s in states])
        preds = np.clip(last + np.column_stack([models[p].predict(X) for p in DEFAULT_FEATURES]), 0, np.log1p(1e6))
        for i, (s, p) in enumerate(zip(states, preds)):
            s["logv"] = np.vstack([s["logv"], p])
            out[i].append(p)
    return [np.asarray(o) for o in out]


@st.cache_resource(show_spinner="Training the forecasting model...")
def train_forecaster(history: pd.DataFrame, holdout_days: int = 14, eval_horizons=(1, 3, 7)):
    """history: daily observed rows (city, date, lat, lon, pollutants). Returns models + backtest."""
    validate_measurements(history, DEFAULT_FEATURES)
    revision = int(pd.util.hash_pandas_object(history, index=True).sum())
    # Reuse the saved forecaster only when its training history is identical.
    # This prevents stale forecasts when the live data changes.
    try:
        if FORECAST_MODEL_PATH.is_file():
            with FORECAST_MODEL_PATH.open("rb") as artifact_file:
                saved = pickle.load(artifact_file)
            if (isinstance(saved, dict) and saved.get("revision") == revision
                    and set(saved.get("models", {})) == set(DEFAULT_FEATURES)
                    and set(saved.get("resid_std", {})) == set(DEFAULT_FEATURES)):
                return saved
    except Exception:
        # A missing, stale, or incompatible artifact is retrained below.
        pass
    if history.groupby("city").size().min() <= holdout_days + max(LAGS) + 1:
        raise ValueError("At least 23 complete history days per city are required.")
    X, Y, P, is_test = [], [], [], []
    for _, g in history.groupby("city"):
        g = g.sort_values("date")
        logv = np.log1p(g[DEFAULT_FEATURES].clip(lower=0).to_numpy())
        dates = list(g["date"])
        lat, lon = g["lat"].iloc[0], g["lon"].iloc[0]
        for t in range(max(LAGS), len(g)):
            X.append(_design_row(logv, t, dates[t], lat, lon))
            Y.append(logv[t])
            P.append(logv[t - 1])
            is_test.append(t >= len(g) - holdout_days)
    X, Y, P, is_test = map(np.asarray, (X, Y, P, is_test))

    eval_models, resid_std = {}, {}
    for j, pol in enumerate(DEFAULT_FEATURES):
        m = _make_regressor().fit(X[~is_test], Y[~is_test, j] - P[~is_test, j])
        eval_models[pol] = m
        resid_std[pol] = float(np.std(Y[is_test, j] - (P[is_test, j] + m.predict(X[is_test]))))

    # Multi-day backtest: forecast from several origins inside the hold-out window.
    max_h = max(eval_horizons)
    errs = {h: {"model": [], "persist": [], "model_aqi": [], "persist_aqi": []} for h in eval_horizons}
    for _, g in history.groupby("city"):
        g = g.sort_values("date").reset_index(drop=True)
        for origin in range(len(g) - holdout_days, len(g) - max_h + 1):
            past, future = g.iloc[:origin], g.iloc[origin:origin + max_h]
            pred = np.expm1(_recursive(eval_models, _series_states([("x", past)]), max_h)[0])
            last = past[DEFAULT_FEATURES].iloc[-1].to_numpy()
            for h in eval_horizons:
                true = future[DEFAULT_FEATURES].iloc[h - 1].to_numpy()
                errs[h]["model"].append(np.abs(pred[h - 1] - true))
                errs[h]["persist"].append(np.abs(last - true))
                true_aqi = compute_aqi(true[0], true[1])
                errs[h]["model_aqi"].append(abs(compute_aqi(pred[h - 1][0], pred[h - 1][1]) - true_aqi))
                errs[h]["persist_aqi"].append(abs(compute_aqi(last[0], last[1]) - true_aqi))
    rows = []
    for h in eval_horizons:
        mm, pm = np.mean(errs[h]["model"], axis=0), np.mean(errs[h]["persist"], axis=0)
        ma, pa = np.mean(errs[h]["model_aqi"]), np.mean(errs[h]["persist_aqi"])
        rows.append({"Horizon": f"{h} day" + ("s" if h > 1 else ""), "AQI MAE (model)": ma, "AQI MAE (persistence)": pa,
                     "PM2.5 MAE (model)": mm[0], "PM2.5 MAE (persistence)": pm[0],
                     "AQI skill vs persistence": 1 - ma / pa if pa else np.nan})

    models = {pol: _make_regressor().fit(X, Y[:, j] - P[:, j]) for j, pol in enumerate(DEFAULT_FEATURES)}
    forecaster = {"models": models, "resid_std": resid_std, "metrics": pd.DataFrame(rows),
                  "n_train": int(len(X)), "holdout_days": holdout_days, "revision": revision}
    # Persist the trained estimator bundle so future app starts can reuse it.
    # If the hosting filesystem is read-only, the in-memory forecast still works.
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = FORECAST_MODEL_PATH.with_suffix(".pkl.tmp")
        with temp_path.open("wb") as artifact_file:
            pickle.dump(forecaster, artifact_file, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path.replace(FORECAST_MODEL_PATH)
    except OSError:
        pass
    return forecaster


def forecast_series(forecaster, groups, horizon):
    """groups: list of (name, history_df). Recursive multi-day forecast for all series at once."""
    states = _series_states(groups)
    meta = [(s["name"], s["last"], s["lat"], s["lon"]) for s in states]
    paths = _recursive(forecaster["models"], states, horizon)
    rows = []
    for (name, last, lat, lon), path in zip(meta, paths):
        for h in range(1, horizon + 1):
            p = path[h - 1]
            rec = {"city": name, "date": last + pd.Timedelta(days=h), "horizon": h, "lat": lat, "lon": lon}
            for j, pol in enumerate(DEFAULT_FEATURES):
                sd = forecaster["resid_std"][pol] * np.sqrt(h)
                rec[pol] = float(np.expm1(p[j]))
                rec[f"{pol}_lo"] = float(max(np.expm1(p[j] - 1.28 * sd), 0))
                rec[f"{pol}_hi"] = float(np.expm1(min(p[j] + 1.28 * sd, np.log1p(1e6))))
            rows.append(rec)
    fc = pd.DataFrame(rows)
    fc["AQI"] = [compute_aqi(a, b) for a, b in zip(fc["PM2.5"], fc["PM10"])]
    fc["AQI_lo"] = [compute_aqi(a, b) for a, b in zip(fc["PM2.5_lo"], fc["PM10_lo"])]
    fc["AQI_hi"] = [compute_aqi(a, b) for a, b in zip(fc["PM2.5_hi"], fc["PM10_hi"])]
    fc["Category"] = [aqi_category(a)[0] for a in fc["AQI"]]
    if can_cluster(fc):
        fc["Cluster"] = assign_clusters(fc)[0]
        fc["Profile"] = [cluster_name(c) for c in fc["Cluster"]]
    return fc


def blend_outlook(ml, cams):
    """Average the ML forecast with the CAMS forecast where both exist; ML alone beyond the CAMS range."""
    merged = ml.merge(cams[["city", "date"] + DEFAULT_FEATURES], on=["city", "date"], how="left", suffixes=("", "_cams"))
    for pol in DEFAULT_FEATURES:
        cams_val = merged[f"{pol}_cams"]
        merged[pol] = np.where(cams_val.notna(), (merged[pol] + cams_val) / 2, merged[pol])
    merged["source"] = np.where(merged["PM2.5_cams"].notna(), "ML + CAMS ensemble", "ML extended")
    merged = merged.drop(columns=[f"{p}_cams" for p in DEFAULT_FEATURES] +
                         [c for c in merged if c.endswith(("_lo", "_hi"))])
    merged["AQI"] = [compute_aqi(a, b) for a, b in zip(merged["PM2.5"], merged["PM10"])]
    merged["Category"] = [aqi_category(a)[0] for a in merged["AQI"]]
    if can_cluster(merged):
        merged["Cluster"] = assign_clusters(merged)[0]
        merged["Profile"] = [cluster_name(c) for c in merged["Cluster"]]
    return merged


@st.cache_resource(ttl=1800, show_spinner="Preparing air-quality intelligence...")
def get_data():
    hourly, source, now_local = load_city_hourly()
    daily = to_daily(hourly)
    today = now_local.normalize()
    history = daily[daily["date"] < today].reset_index(drop=True)
    cams = daily[daily["date"] >= today].reset_index(drop=True)
    forecaster = train_forecaster(history)
    ml = forecast_series(forecaster, list(history.groupby("city", sort=False)), MAX_HORIZON)
    snap = city_snapshot(hourly, now_local)
    return {"hourly": hourly, "daily": daily, "history": history, "cams": cams, "ml": ml,
            "outlook": blend_outlook(ml, cams), "snap": snap, "forecaster": forecaster,
            "source": source, "now": now_local, "today": today}


def google_maps_key():
    key = st.session_state.get("gmaps_key") or os.environ.get("GOOGLE_MAPS_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GOOGLE_MAPS_API_KEY", "")
        except Exception:
            key = ""
    return key.strip()


# ============================================================================
# Shared UI components
# ============================================================================

def page_header(kicker, title, subtitle, data=None):
    provenance = ""
    if data is not None:
        if data["source"] == "live":
            provenance = f'<span><span class="status-dot"></span>CAMS model · as of {data["now"]:%d %b %Y, %H:%M} IST</span>'
        else:
            provenance = '<span style="color:#fbbf24">Offline · synthetic demonstration data</span>'
    st.html(f"""
    <section class="hero">
      <div class="eyebrow">AirScope / {escape(kicker)}</div>
      <h1>{escape(title)}</h1><p class="lead">{escape(subtitle)}</p>
    </section>
    <div class="context-bar">{provenance}<span>Particulate-only US EPA AQI · not CPCB AQI</span></div>""")
    if data is not None and data["source"] != "live":
        st.warning("Live CAMS data is unavailable. All charts, forecasts and evaluation metrics on this page use synthetic demo data.")


def kpi_cards(items):
    html = "".join(
        f'<div class="kpi"><div class="kpi-label">{escape(str(i["label"]))}</div>'
        f'<div class="kpi-value">{escape(str(i["value"]))}</div><div class="kpi-sub">{escape(str(i.get("sub", "")))}</div></div>'
        for i in items)
    st.markdown(f'<div class="kpi-grid">{html}</div>', unsafe_allow_html=True)


def city_card(rec, title=None, subtitle=None):
    name, color, advice = aqi_category(rec["AQI"])
    cells = "".join(
        f'<div class="pcell"><small>{p} · {UNITS[p]}</small><b>{rec[p]:.2f}</b></div>' if p == "CO"
        else f'<div class="pcell"><small>{p} · {UNITS[p]}</small><b>{rec[p]:.0f}</b></div>'
        for p in DEFAULT_FEATURES)
    profile = f'<div class="kpi-sub">Pattern: <b>{escape(str(rec["Profile"]))}</b></div>' if "Profile" in rec and pd.notna(rec.get("Profile")) else ""
    st.markdown(f"""
    <div class="city-card">
      <div class="eyebrow">{escape(subtitle or CITY_INDEX.get(rec.get("city"), ("", "India"))[1])}</div>
      <h4>{escape(title or rec["city"])}</h4>
      <div class="city-aqi" style="color:{color}">{rec["AQI"]:.0f}</div>
      {badge(rec["AQI"])} <span class="kpi-sub">PM-only AQI · 24h mean / daily forecast</span>
      {profile}
      <div class="pgrid">{cells}</div>
      <div class="advice">{advice}</div>
    </div>""", unsafe_allow_html=True)


def aqi_bubble_map(df, value_col="AQI", height=560, style="carto-darkmatter", zoom=3.6, center=None,
                   animation_frame=None, title=None):
    keep = ["city", "lat", "lon", "AQI", "PM2.5", "PM10", "NO2", "O3", "CO", value_col] + ([animation_frame] if animation_frame else [])
    df = df[[c for c in dict.fromkeys(keep) if c in df]].copy()
    num = df.select_dtypes("number").columns.difference(["lat", "lon"])
    df[num] = df[num].round(2)
    df["Category"] = [aqi_category(a)[0] for a in df["AQI"]]
    df["size"] = df[value_col].clip(lower=1) ** 0.8
    hover = {c: (":.0f" if c != "CO" else ":.2f") for c in ["AQI", "PM2.5", "PM10", "NO2", "O3", "CO"] if c in df}
    hover.update({"lat": False, "lon": False, "size": False})
    if value_col == "AQI":
        fig = px.scatter_map(df, lat="lat", lon="lon", color="Category", size="size", size_max=34,
                             hover_name="city", hover_data=hover, color_discrete_map=AQI_COLOR_MAP,
                             category_orders={"Category": AQI_ORDER}, animation_frame=animation_frame,
                             zoom=zoom, center=center or {"lat": 22.5, "lon": 80.5}, title=title)
    else:
        fig = px.scatter_map(df, lat="lat", lon="lon", color=value_col, size="size", size_max=34,
                             hover_name="city", hover_data=hover, color_continuous_scale="Turbo",
                             animation_frame=animation_frame, zoom=zoom,
                             center=center or {"lat": 22.5, "lon": 80.5}, title=title)
    fig.update_traces(marker=dict(opacity=.85))
    if style == "satellite":
        fig.update_layout(map_style="white-bg", map_layers=[{
            "below": "traces", "sourcetype": "raster", "sourceattribution": "Esri World Imagery",
            "source": ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"]}])
    else:
        fig.update_layout(map_style=style)
    style_fig(fig, height=height, margin=dict(l=0, r=0, t=40 if title else 0, b=0))
    fig.update_layout(showlegend=False)
    return fig


# ============================================================================
# Google Maps
# ============================================================================
GOOGLE_MAP_TEMPLATE = """
<div id="wrap">
  <div id="map"></div>
  <div id="panel">
    <div class="ttl">Air quality · <span id="daylbl"></span></div>
    <input id="day" type="range" min="0" max="__MAXDAY__" value="__DAY__" step="1">
    <div class="ticks"><span>Now</span><span>+__MAXDAY__ days</span></div>
    <div class="lg">__LEGEND__</div>
  </div>
</div>
<style>
  html, body { margin:0; height:100%; background:#070d1a; font-family: Inter, system-ui, sans-serif; }
  #wrap { position:relative; height:__HEIGHT__px; border-radius:18px; overflow:hidden; border:1px solid rgba(148,163,184,.18); }
  #map { position:absolute; inset:0; }
  #panel { position:absolute; left:12px; bottom:22px; z-index:5; width:270px; padding:12px 14px; border-radius:14px;
    background:rgba(7,13,26,.86); color:#e2e8f0; backdrop-filter:blur(8px); border:1px solid rgba(148,163,184,.2); font-size:12px; }
  #panel .ttl { font-weight:700; font-size:13px; margin-bottom:6px; }
  #panel input { width:100%; accent-color:#38bdf8; }
  .ticks { display:flex; justify-content:space-between; color:#94a3b8; font-size:10px; }
  .lg { display:flex; flex-wrap:wrap; gap:6px 10px; margin-top:8px; }
  .lg span { display:inline-flex; align-items:center; gap:5px; color:#cbd5e1; font-size:10.5px; }
  .lg i { width:9px; height:9px; border-radius:3px; display:inline-block; }
  .iw { font-family: Inter, system-ui, sans-serif; color:#0f172a; min-width:190px; }
  .iw h3 { margin:0 0 2px; font-size:15px; } .iw .a { font-size:26px; font-weight:800; }
  .iw table { border-collapse:collapse; margin-top:6px; font-size:12px; } .iw td { padding:1px 8px 1px 0; }
</style>
<script>
const DATA = __DATA__;
const DARK = [
  {elementType:"geometry", stylers:[{color:"#0f1a2e"}]},
  {elementType:"labels.text.fill", stylers:[{color:"#94a3b8"}]},
  {elementType:"labels.text.stroke", stylers:[{color:"#070d1a"}]},
  {featureType:"administrative", elementType:"geometry.stroke", stylers:[{color:"#334155"}]},
  {featureType:"road", elementType:"geometry", stylers:[{color:"#1e293b"}]},
  {featureType:"poi", stylers:[{visibility:"off"}]},
  {featureType:"water", elementType:"geometry", stylers:[{color:"#0b2540"}]}
];
let markers = [], halos = [], info;
function render(d) {
  document.getElementById("daylbl").textContent = DATA.days[d];
  DATA.cities.forEach((c, i) => {
    const v = c.v[d];
    markers[i].setIcon({path: google.maps.SymbolPath.CIRCLE, scale: 15, fillColor: v.color, fillOpacity: .95,
                        strokeColor: "#ffffff", strokeWeight: 1.5});
    markers[i].setLabel({text: String(Math.round(v.aqi)), color: "#0b1220", fontSize: "11px", fontWeight: "800"});
    halos[i].setOptions({fillColor: v.color, strokeColor: v.color, radius: 9000 + v.aqi * 260});
  });
}
function initMap() {
  const map = new google.maps.Map(document.getElementById("map"), {
    center: __CENTER__, zoom: __ZOOM__, mapTypeId: "__MAPTYPE__", styles: (__STYLES__ === "__USE_DARK__" ? DARK : __STYLES__),
    mapTypeControl: true, streetViewControl: true, fullscreenControl: true,
  });
  info = new google.maps.InfoWindow();
  DATA.cities.forEach((c) => {
    halos.push(new google.maps.Circle({map, center: {lat: c.lat, lng: c.lon}, fillOpacity: .18, strokeOpacity: .5,
                                       strokeWeight: 1, clickable: false}));
    const m = new google.maps.Marker({map, position: {lat: c.lat, lng: c.lon}, title: c.name});
    m.addListener("click", () => {
      const d = +document.getElementById("day").value, v = c.v[d];
      info.setContent(`<div class="iw"><h3>${c.name}</h3><div style="color:#64748b;font-size:11px">${DATA.days[d]}</div>
        <div class="a" style="color:${v.color}">${Math.round(v.aqi)}</div><b>${v.cat}</b>
        <table><tr><td>PM2.5</td><td><b>${v.pm25.toFixed(0)}</b> µg/m³</td></tr><tr><td>PM10</td><td><b>${v.pm10.toFixed(0)}</b> µg/m³</td></tr>
        <tr><td>NO2</td><td><b>${v.no2.toFixed(0)}</b> µg/m³</td></tr><tr><td>O3</td><td><b>${v.o3.toFixed(0)}</b> µg/m³</td></tr></table></div>`);
      info.open({map, anchor: m});
    });
    markers.push(m);
  });
  const slider = document.getElementById("day");
  slider.addEventListener("input", () => render(+slider.value));
  render(+slider.value);
}
window.gm_authFailure = () => { document.getElementById("map").innerHTML =
  '<div style="color:#fca5a5;padding:24px;font-size:14px">Google Maps rejected the API key. Check that the Maps JavaScript API is enabled and the key allows this site.</div>'; };
</script>
<script src="https://maps.googleapis.com/maps/api/js?key=__KEY__&callback=initMap&loading=async" async defer></script>
"""


def google_map_payload(data, n_days=7):
    snap, outlook = data["snap"], data["outlook"]
    days = sorted(outlook["date"].unique())[:n_days]
    labels = ["Now (24h mean)"] + [pd.Timestamp(d).strftime("%a %d %b") + " · forecast" for d in days]
    cities = []
    for _, rec in snap.iterrows():
        fc = outlook[outlook["city"] == rec["city"]].set_index("date")
        values = []
        for row in [rec] + [fc.loc[d] for d in days]:
            name, color, _ = aqi_category(row["AQI"])
            values.append({"aqi": float(row["AQI"]), "cat": name, "color": color, "pm25": float(row["PM2.5"]),
                           "pm10": float(row["PM10"]), "no2": float(row["NO2"]), "o3": float(row["O3"])})
        cities.append({"name": rec["city"], "lat": float(rec["lat"]), "lon": float(rec["lon"]), "v": values})
    return {"days": labels, "cities": cities}


def render_google_map(data, key, day_index=0, height=600, map_type="roadmap", dark=True, center=None, zoom=5):
    payload = google_map_payload(data)
    legend = "".join(f'<span><i style="background:{c}"></i>{n}</span>' for _, n, c, _ in AQI_CATEGORIES)
    html = (GOOGLE_MAP_TEMPLATE
            .replace("__DATA__", json.dumps(payload))
            .replace("__MAXDAY__", str(len(payload["days"]) - 1))
            .replace("__DAY__", str(day_index))
            .replace("__LEGEND__", legend)
            .replace("__HEIGHT__", str(height))
            .replace("__CENTER__", json.dumps(center or {"lat": 22.5, "lng": 80.5}))
            .replace("__ZOOM__", str(zoom))
            .replace("__MAPTYPE__", map_type)
            .replace("__STYLES__", json.dumps(DARK_STYLE_FLAG if dark else []))
            .replace("__KEY__", quote(key, safe="")))
    components.html(html, height=height + 4)


DARK_STYLE_FLAG = "__USE_DARK__"


def google_embed(lat, lon, zoom=11, satellite=False, height=360, query=None):
    """Keyless Google Maps embed for a single location."""
    q = requests.utils.quote(query) if query else f"{lat},{lon}"
    t = "k" if satellite else "m"
    src = f"https://maps.google.com/maps?q={q}&ll={lat},{lon}&t={t}&z={zoom}&ie=UTF8&iwloc=&output=embed"
    components.html(
        f'<iframe src="{src}" width="100%" height="{height}" style="border:0;border-radius:16px" loading="lazy" '
        f'referrerpolicy="no-referrer-when-downgrade" allowfullscreen></iframe>', height=height + 6)


# ============================================================================
# Pages
# ============================================================================

@st.fragment
def _dashboard_trend(history, outlook, today, city):
    st.markdown("### History & outlook")
    controls = st.columns([1.4, 1])
    metric = controls[0].selectbox("Trend pollutant", ["AQI"] + DEFAULT_FEATURES, key="dash_metric")
    period = controls[1].segmented_control("History window", [14, 30, 90], default=30, key="dash_period") or 30
    if city != "All cities":
        history = history[history["city"] == city]
        outlook = outlook[outlook["city"] == city]
    obs = history.groupby("date")[metric].mean().reset_index().tail(period)
    fc = outlook.groupby("date")[metric].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=obs["date"], y=obs[metric], name="Historical model estimates",
                             line=dict(color="#38bdf8", width=2.5), fill="tozeroy", fillcolor="rgba(56,189,248,.05)"))
    fig.add_trace(go.Scatter(x=fc["date"], y=fc[metric], name="Forecast outlook",
                             line=dict(color="#2dd4bf", width=2, dash="dash"), mode="lines+markers", marker_size=4))
    fig.add_vline(x=today, line=dict(color="#8191a3", dash="dot"))
    style_fig(fig, height=330, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(title=metric + ("" if metric == "AQI" else f" ({UNITS[metric]})"), rangemode="tozero")
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="dash_trend")
    st.caption("Dotted boundary: today. Historical data is archived CAMS output, not station observations.")


@st.fragment
def _dashboard_comparison(data):
    st.markdown("### Compare & explore")
    view = st.segmented_control("Comparison view", ["City comparison", "Pollutant fingerprint", "AQI distribution"],
                                 default="City comparison", key="dash_compare_view") or "City comparison"
    snap = data["snap"]
    if view == "City comparison":
        cols = st.columns([1, 2])
        pol = cols[0].selectbox("Compare pollutant", ["AQI"] + DEFAULT_FEATURES, key="dash_compare_pol")
        cities = cols[1].multiselect("Compare cities", list(CITY_INDEX), default=["Delhi", "Mumbai", "Bengaluru"],
                                     max_selections=5, key="dash_compare_cities")
        if not cities:
            st.info("Select at least one city to compare.")
            return
        frame = data["history"][data["history"]["city"].isin(cities)].groupby("city").tail(30)
        fig = px.line(frame, x="date", y=pol, color="city",
                      color_discrete_sequence=["#38bdf8", "#2dd4bf", "#a5b4fc", "#fbbf24", "#fb7185"])
        fig.update_yaxes(title=pol + ("" if pol == "AQI" else f" ({UNITS[pol]})"))
        fig.update_xaxes(title=None)
        st.caption("Last 30 complete days · all series use the same units; CO is mg/m³.")
    elif view == "Pollutant fingerprint":
        heat = snap.set_index("city")[DEFAULT_FEATURES].sort_values("PM2.5", ascending=False)
        rel = heat / heat.max().replace(0, np.nan)
        fig = go.Figure(go.Heatmap(z=rel.values, x=DEFAULT_FEATURES, y=rel.index,
                                   colorscale=[[0, "#11171f"], [.5, "#175c78"], [1, "#67e8f9"]],
                                   customdata=heat.values, hovertemplate="%{y} · %{x}: %{customdata:.2f}<extra></extra>",
                                   colorbar=dict(title="Relative", thickness=10)))
        fig.update_yaxes(autorange="reversed", tickfont_size=10)
        st.caption("Normalized per pollutant across cities (0–1); hover for concentrations in each pollutant's units.")
    else:
        mix = snap["Category"].value_counts().reindex(AQI_ORDER).dropna().reset_index()
        mix.columns = ["Category", "Cities"]
        fig = px.bar(mix, x="Category", y="Cities", color="Category", color_discrete_map=AQI_COLOR_MAP)
        fig.update_xaxes(title=None)
    st.plotly_chart(style_fig(fig, height=400), config=PLOT_CONFIG, theme=None, key="dash_comparison")


def page_dashboard():
    data = get_data()
    snap, outlook, history = data["snap"], data["outlook"], data["history"]

    # New editorial observatory layout: headline -> live signal -> trend -> geography.
    page_header("AIR QUALITY OBSERVATORY", "See the air differently.",
                "A live view of India's modelled air quality, the changes behind it, and the outlook ahead.", data)

    left, right = st.columns([1.65, 1], gap="large")
    with left:
        city = st.selectbox("Explore a location", ["All cities"] + list(CITY_INDEX), key="overview_city",
                            label_visibility="visible")
    with right:
        st.download_button("Export readings", snap.drop(columns="Color").round(2).to_csv(index=False).encode(),
                           "airscope_current_readings.csv", "text/csv", icon=":material/download:", width="stretch")

    selected = snap if city == "All cities" else snap[snap["city"] == city]
    scope = "India · 26-city view" if city == "All cities" else f"{city} · {CITY_INDEX[city][1]}"
    current = selected[DEFAULT_FEATURES].mean()
    aqi = float(round(selected["AQI"].mean()))
    category, color, advice = aqi_category(aqi)
    delta = float(selected["AQI_change"].mean())
    future = outlook if city == "All cities" else outlook[outlook["city"] == city]
    tomorrow = future[future["date"] == data["today"] + pd.Timedelta(days=1)]
    next_aqi = float(tomorrow["AQI"].mean()) if not tomorrow.empty else aqi
    next_category, next_color, _ = aqi_category(next_aqi)
    method = "ML + CAMS" if data["source"] == "live" else "Synthetic demo"
    scale = "".join(f'<span style="background:{c}"></span>' for _, _, c, _ in AQI_CATEGORIES)

    st.html(f"""
    <section class="observatory-hero">
      <div class="observatory-copy">
        <div class="observatory-kicker"><span class="live-indicator"></span> CURRENT CONDITIONS <span class="observatory-sep">/</span> {escape(scope.upper())}</div>
        <div class="observatory-mainline">
          <div class="observatory-score" style="color:{color}">{aqi:.0f}</div>
          <div class="observatory-score-meta"><span class="observatory-unit">PARTICULATE AQI <span class="observatory-muted">/ 500</span></span>
            {badge(aqi)}<div class="observatory-delta">{delta:+.0f} points over the previous 24 hours</div></div>
        </div>
        <div class="observatory-advice">{escape(advice)}</div>
        <div class="observatory-scale">{scale}</div>
        <div class="observatory-scale-labels"><span>0 · Good</span><span>500 · Hazardous</span></div>
      </div>
      <div class="observatory-outlook">
        <div class="observatory-kicker">THE NEXT 24 HOURS</div>
        <div class="outlook-date">Tomorrow</div>
        <div class="observatory-next" style="color:{next_color}">{next_aqi:.0f}<span> AQI</span></div>
        <div class="outlook-status">{escape(next_category)}</div>
        <div class="outlook-change">{next_aqi - aqi:+.0f} <span>vs. current reading</span></div>
        <div class="outlook-foot">Forecast blend · {method}<br/>Up to 14 days available</div>
      </div>
    </section>
    """)

    worst, best = snap.loc[snap["AQI"].idxmax()], snap.loc[snap["AQI"].idxmin()]
    f = data["forecaster"]
    metrics = f["metrics"].iloc[0]
    st.html(f"""
    <div class="signal-strip">
      <div class="signal-item"><span class="signal-label">CITIES IN VIEW</span><strong>{len(snap):02d}</strong><small>Across India</small></div>
      <div class="signal-item"><span class="signal-label">HIGHEST READING</span><strong>{escape(str(worst['city']))}</strong><small>AQI {worst['AQI']:.0f} · {escape(str(worst['Category']))}</small></div>
      <div class="signal-item"><span class="signal-label">LOWEST READING</span><strong>{escape(str(best['city']))}</strong><small>AQI {best['AQI']:.0f} · {escape(str(best['Category']))}</small></div>
      <div class="signal-item"><span class="signal-label">ABOVE AQI 150</span><strong>{int((snap['AQI'] > 150).sum()):02d}<span class="signal-total"> / {len(snap)}</span></strong><small>City readings</small></div>
    </div>
    """)

    st.markdown('<div class="observatory-section-head"><div><span class="section-index">01</span><h2>Signals over time</h2></div><span class="section-aside">DAILY ESTIMATES · HISTORICAL + OUTLOOK</span></div>', unsafe_allow_html=True)
    with st.container(border=True):
        _dashboard_trend(history, outlook, data["today"], city)

    st.markdown('<div class="observatory-section-head"><div><span class="section-index">02</span><h2>Where the air changes</h2></div><span class="section-aside">GEOGRAPHIC DISTRIBUTION</span></div>', unsafe_allow_html=True)
    map_col, list_col = st.columns([1.35, 1], gap="large")
    with map_col:
        with st.container(border=True):
            st.caption("Modelled 24-hour particulate AQI across the selected city set.")
            st.plotly_chart(aqi_bubble_map(selected if city != "All cities" else snap, height=460, zoom=3.4), config=PLOT_CONFIG, theme=None, key="dash_map")
            st.html(aqi_legend_html())
            st.page_link(next(p for p in pages if p.title == "Map"), label="Open full map explorer", icon=":material/arrow_outward:")
    with list_col:
        with st.container(border=True):
            st.markdown("### City watch")
            st.caption("Ranked from highest to lowest particulate AQI.")
            ranked = snap.sort_values("AQI", ascending=False)
            st.dataframe(ranked[["city", "AQI", "Category", "AQI_change"]].rename(
                columns={"city": "City", "Category": "Category", "AQI_change": "24h change"}),
                hide_index=True, height=440, width="stretch",
                column_config={"AQI": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=500),
                               "24h change": st.column_config.NumberColumn(format="%+.0f")})
            st.page_link(next(p for p in pages if p.title == "Forecast"), label="Explore the 14-day forecast", icon=":material/trending_up:")

    st.markdown('<div class="observatory-section-head"><div><span class="section-index">03</span><h2>Pollutant fingerprint</h2></div><span class="section-aside">24-HOUR MEANS · LIVE MODEL ESTIMATES</span></div>', unsafe_allow_html=True)
    hourly = data["hourly"]
    if city != "All cities":
        hourly = hourly[hourly["city"] == city]
    recent = hourly[(hourly["time"] <= data["now"]) & (hourly["time"] > data["now"] - pd.Timedelta(hours=24))]
    with st.container(border=True):
        pollutant_cards(current, recent.groupby("time")[DEFAULT_FEATURES].mean())
    st.caption("City averages are unweighted, not population-weighted national statistics. This particulate-only AQI is not the official CPCB AQI. Gas pollutants are shown separately and are excluded from this index.")

    st.markdown('<div class="observatory-section-head"><div><span class="section-index">04</span><h2>How much to trust the model</h2></div><span class="section-aside">METHOD + LIMITATIONS</span></div>', unsafe_allow_html=True)
    st.html(f"""<div class="trust-row observatory-trust">
      <div><b>{'Demonstration clustering' if demo_mode else 'Saved clustering model'}</b>
        StandardScaler + K-Means · {len(features)} inputs. Clusters describe patterns, not health categories.</div>
      <div><b>1-day AQI MAE · {metrics['AQI MAE (model)']:.1f} points</b>
        Persistence baseline: {metrics['AQI MAE (persistence)']:.1f} · ML-only backtest, not ensemble accuracy.</div>
      <div><b>{f['holdout_days']}-day holdout · {f['n_train']:,} fitting rows</b>
        {'Archived CAMS model estimates' if data['source'] == 'live' else 'Synthetic demo data only'}. Historical errors do not guarantee future performance.</div>
    </div>""")
    st.caption("Data source: Copernicus CAMS via Open-Meteo when available. Offline fallback values are synthetic and are not live observations. Open Forecast and Methodology for full evaluation details.")
    

@st.fragment
def _map_explorer(data):
    snap, outlook = data["snap"], data["outlook"]
    key = google_maps_key()
    days = sorted(outlook["date"].unique())[:7]
    time_labels = ["Now"] + [pd.Timestamp(d).strftime("%a %d %b") for d in days]

    c1, c2, c3, c4 = st.columns([1.2, 1, 1.6, 1])
    engine = c1.segmented_control("Map engine", ["Google Maps", "Interactive"], default="Google Maps" if key else "Interactive",
                                  key="map_engine") or "Interactive"
    layer = c2.selectbox("Layer", ["AQI"] + DEFAULT_FEATURES, key="map_layer")
    when = c3.select_slider("Time", time_labels, value="Now", key="map_time")
    style = c4.selectbox("Base map", ["Dark", "Streets", "Satellite"], key="map_style")
    day_idx = time_labels.index(when)
    frame = snap if day_idx == 0 else outlook[outlook["date"] == days[day_idx - 1]]

    map_col, side = st.columns([2.1, 1], gap="large")
    with side:
        cities = list(snap.sort_values("AQI", ascending=False)["city"])
        sel = st.selectbox("Focus city", cities, key="map_city")
        rec = frame[frame["city"] == sel].iloc[0].to_dict()
        city_card(rec, subtitle=f"{CITY_INDEX[sel][1]} · {when}")
        lat, lon = CITY_INDEX[sel][2], CITY_INDEX[sel][3]
        st.markdown(f"[Open {sel} in Google Maps ↗](https://www.google.com/maps/search/?api=1&query={lat},{lon})")

    with map_col:
        if engine == "Google Maps":
            if key:
                render_google_map(data, key, day_index=day_idx, height=620,
                                  map_type="satellite" if style == "Satellite" else "roadmap", dark=style == "Dark")
            else:
                st.info("Showing the free Google Maps embed for the focus city. Add a Google Maps JavaScript API key in the "
                        "sidebar (or set `GOOGLE_MAPS_API_KEY`) to see every city's AQI as live markers on Google Maps.",
                        icon=":material/map:")
                google_embed(lat, lon, zoom=11, satellite=style == "Satellite", height=580, query=f"{sel}, India")
        else:
            st.markdown(aqi_legend_html() if layer == "AQI" else "", unsafe_allow_html=True)
            fig = aqi_bubble_map(frame, value_col=layer, height=620, zoom=3.9,
                                 style={"Dark": "carto-darkmatter", "Streets": "carto-voyager", "Satellite": "satellite"}[style])
            st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="map_interactive")

    if st.toggle(f"Street-level satellite view of {sel} (Google Maps)", key="map_street"):
        google_embed(lat, lon, zoom=12, satellite=True, height=420, query=f"{sel}, India")


def page_map():
    data = get_data()
    outlook = data["outlook"]
    page_header("Geospatial view", "Pollution map",
                "Explore every monitored city on Google Maps or the interactive overlay. Slide through time to see the predicted outlook.",
                data)
    _map_explorer(data)

    st.markdown("### Animated outlook")
    st.caption("Watch the predicted AQI evolve across India over the next 14 days.")
    if st.toggle("Load animated 14-day outlook", key="map_anim_on"):
        st.markdown(aqi_legend_html(), unsafe_allow_html=True)
        anim = outlook.copy()
        anim["Day"] = anim["date"].dt.strftime("%a %d %b")
        st.plotly_chart(aqi_bubble_map(anim, height=560, zoom=3.6, animation_frame="Day"), config=PLOT_CONFIG, theme=None, key="map_anim")


def _forecast_location(data):
    mode = st.segmented_control("Location", ["City", "Custom coordinates"], default="City", key="fc_mode") or "City"
    if mode == "City":
        city = st.selectbox("City", list(CITY_INDEX), key="fc_city")
        hist = data["history"][data["history"]["city"] == city]
        cams = data["cams"][data["cams"]["city"] == city]
        ml = data["ml"][data["ml"]["city"] == city]
        outlook = data["outlook"][data["outlook"]["city"] == city]
        hourly = data["hourly"][data["hourly"]["city"] == city]
        return city, CITY_INDEX[city][2], CITY_INDEX[city][3], hist, cams, ml, outlook, hourly, data["now"]
    a, b, c = st.columns(3)
    name = a.text_input("Place name", "My location", max_chars=80, key="fc_name").strip() or "My location"
    lat = b.number_input("Latitude", -90.0, 90.0, 28.5355, format="%.4f", key="fc_lat")
    lon = c.number_input("Longitude", -180.0, 180.0, 77.3910, format="%.4f", key="fc_lon")
    try:
        hist, cams, ml, outlook, hourly, local_now = custom_location_forecast(
            name, round(lat, 4), round(lon, 4), data["forecaster"]["revision"], data["forecaster"])
    except Exception:
        st.error("Could not retrieve complete CAMS history for these coordinates. Try again later or select a supported city.")
        st.stop()
    return name, lat, lon, hist, cams, ml, outlook, hourly, local_now


@st.cache_data(ttl=1800, show_spinner="Forecasting this location...", max_entries=64)
def custom_location_forecast(name, lat, lon, model_revision, _forecaster):
    frames, offsets = fetch_open_meteo((lat,), (lon,))
    local_now = (pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(seconds=offsets[0])).floor("h")
    today = local_now.normalize()
    hourly = frames[0].assign(city=name, lat=lat, lon=lon)
    hourly[DEFAULT_FEATURES] = hourly[DEFAULT_FEATURES].interpolate(limit_direction="both")
    daily = to_daily(hourly)
    hist, cams = daily[daily["date"] < today], daily[daily["date"] >= today]
    ml = forecast_series(_forecaster, [(name, hist)], MAX_HORIZON)
    return hist, cams, ml, blend_outlook(ml, cams), hourly, local_now


def page_forecast():
    data = get_data()
    page_header("Future predicted view", "Air quality forecast",
                "A machine-learning model trained on 3 months of daily pollutant levels, ensembled with the CAMS atmospheric model, "
                "predicts the next two weeks for any city or coordinate.", data)
    top = st.columns([2.2, 1, 1])
    with top[0]:
        name, lat, lon, hist, cams, ml, outlook, hourly, local_now = _forecast_location(data)
    metric = top[1].selectbox("Show", ["AQI"] + DEFAULT_FEATURES, key="fc_metric")
    horizon = top[2].slider("Horizon (days)", 3, MAX_HORIZON, 7, key="fc_horizon")
    outlook, ml = outlook[outlook["horizon"] <= horizon], ml[ml["horizon"] <= horizon]

    st.subheader(f"{name} · daily outlook")
    st.caption(f"Local timeline · {local_now:%d %b %H:%M}. First {min(horizon, 7)} days shown below; the chart and CSV include the selected horizon.")
    forecast_cards(outlook, aqi_category)

    st.markdown("### Historical estimates & prediction")
    recent = hist.sort_values("date").tail(45)
    fig = go.Figure()
    lo_col, hi_col = (f"{metric}_lo", f"{metric}_hi")
    if lo_col in ml:
        fig.add_trace(go.Scatter(x=pd.concat([ml["date"], ml["date"][::-1]]), y=pd.concat([ml[hi_col], ml[lo_col][::-1]]),
                                 fill="toself", fillcolor="rgba(56,189,248,.10)", line=dict(width=0), name="ML indicative range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=recent["date"], y=recent[metric], name="Historical model estimates", line=dict(color="#38bdf8", width=2.5)))
    fig.add_trace(go.Scatter(x=ml["date"], y=ml[metric], name="AirScope ML", line=dict(color="#a78bfa", width=2, dash="dash")))
    shown_cams = cams[cams["date"].between(outlook["date"].min(), outlook["date"].max())]
    fig.add_trace(go.Scatter(x=shown_cams["date"], y=shown_cams[metric], name="CAMS model", line=dict(color="#fbbf24", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=outlook["date"], y=outlook[metric], name="Ensemble outlook", mode="lines+markers",
                             line=dict(color="#2dd4bf", width=3), marker=dict(size=5)))
    if metric == "AQI":
        for limit, cat, color, _ in AQI_CATEGORIES[:5]:
            fig.add_hline(y=limit, line=dict(color=color, width=1, dash="dot"), opacity=.45)
    fig.add_vline(x=local_now.normalize(), line=dict(color="rgba(226,232,240,.4)", dash="dot"))
    style_fig(fig, height=440)
    fig.update_yaxes(title=metric + ("" if metric == "AQI" else f" ({UNITS[metric]})"))
    st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="fc_main")
    st.caption("Shading is a heuristic ML-only range (±1.28 × holdout residual SD × √h in log space), not a calibrated confidence interval or ensemble range.")

    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.markdown("### Next hours · rolling particulate AQI")
        rolling = hourly.sort_values("time").copy()
        rolling[["PM2.5", "PM10"]] = rolling[["PM2.5", "PM10"]].rolling(24, min_periods=24).mean()
        future = rolling[rolling["time"] > local_now].head(120).copy()
        if not future.empty:
            future["AQI (hourly PM)"] = [compute_aqi(x, y) for x, y in zip(future["PM2.5"], future["PM10"])]
            fig = go.Figure(go.Bar(x=future["time"], y=future["AQI (hourly PM)"],
                                   marker_color=[aqi_category(v)[1] for v in future["AQI (hourly PM)"]],
                                   hovertemplate="%{x|%a %d %b %H:%M}<br>AQI %{y:.0f}<extra></extra>"))
            style_fig(fig, height=330, legend=False)
            fig.update_yaxes(title="PM-only AQI · rolling 24h")
            st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="fc_hourly")
            st.caption("CAMS hourly output aggregated over the preceding 24 hours, including the forecast hour.")
    with b:
        st.markdown("### Where it happens")
        google_embed(lat, lon, zoom=10, height=330, query=None if name not in CITY_INDEX else f"{name}, India")

    st.markdown("### Predicted profile and pollutant breakdown")
    show = outlook[["date", "AQI", "Category"] + DEFAULT_FEATURES + (["Profile"] if "Profile" in outlook else []) + ["source"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b")
    st.dataframe(show.rename(columns={"date": "Date", "source": "Method"}), hide_index=True, width="stretch",
                 column_config={p: st.column_config.NumberColumn(format="%.2f" if p == "CO" else "%.0f") for p in DEFAULT_FEATURES + ["AQI"]})
    st.download_button("Download forecast as CSV", show.to_csv(index=False).encode("utf-8"), "airscope_forecast.csv", "text/csv")
    st.download_button(
        "Download trained forecasting model (.pkl)",
        pickle.dumps(data["forecaster"], protocol=pickle.HIGHEST_PROTOCOL),
        "air_pollution_forecast.pkl",
        "application/octet-stream",
        help="Saved Ridge models, residual ranges, backtest metrics and training revision. Only load pickle files you created or trust.",
    )
    st.caption("The app trains from its current historical CAMS data, saves the model bundle when the host permits, and reuses it only when the training history matches exactly.")

    st.markdown("### India outlook ranking")
    window = data["outlook"][data["outlook"]["horizon"] <= horizon]
    peak = window.loc[window.groupby("city")["AQI"].idxmax()].sort_values("AQI", ascending=False)
    avg = window.groupby("city")["AQI"].mean()
    rank = pd.DataFrame({"City": peak["city"], "Peak AQI": peak["AQI"], "Peak day": peak["date"].dt.strftime("%a %d %b"),
                         "Average AQI": peak["city"].map(avg), "Peak category": peak["Category"]})
    st.dataframe(rank, hide_index=True, width="stretch",
                 column_config={"Peak AQI": st.column_config.ProgressColumn(format="%.0f", min_value=0, max_value=max(300, float(rank["Peak AQI"].max()))),
                                "Average AQI": st.column_config.NumberColumn(format="%.0f")})

    with st.expander("How accurate is the forecast?"):
        f = data["forecaster"]
        st.write(f"Back-tested on the last {f['holdout_days']} days across all cities ({f['n_train']:,} training rows). "
                 "Persistence (\"tomorrow equals today\") is the standard baseline; positive skill means the model beats it.")
        st.dataframe(f["metrics"], hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.1f") for c in f["metrics"].columns if "MAE" in c} |
                     {"AQI skill vs persistence": st.column_config.NumberColumn(format="percent")})
        st.caption("These evaluate ML alone against archived CAMS data (or synthetic data when offline), not monitoring stations or the ensemble. "
                   "The outlook averages ML and CAMS on overlapping dates and uses ML alone afterward. Forecasts are indicative, not official advisories.")


def page_patterns():
    data = get_data()
    page_header("Machine learning", "Pollution pattern explorer",
                "K-Means clustering groups pollution readings into recurring atmospheric profiles. Explore them in 3D, 2D and PCA space.", data)
    city_days = "CAMS city-days" if data["source"] == "live" else "Offline city-days"
    source = st.segmented_control("Data", [city_days, "Synthetic demo"], default=city_days, key="pat_src") or city_days
    if source == city_days:
        frame = data["history"].copy()
        hover = ["city", "date"]
    else:
        n = st.slider("Sample size", 150, 1500, 650, 50)
        frame, hover = make_demo_data(n), []
    if not can_cluster(frame):
        st.warning("The loaded model's features are not available in this dataset.")
        return
    labels, scaled = assign_clusters(frame)
    frame[features] = frame[features].round(2)
    frame["Cluster"] = [f"{c} · {cluster_name(c)}" for c in labels]
    counts = frame["Cluster"].value_counts()
    kpi_cards([
        {"label": "Samples", "value": f"{len(frame):,}"},
        {"label": "Clusters", "value": str(len(counts)), "color": "#a78bfa"},
        {"label": "Features", "value": str(len(features)), "color": "#34d399"},
        {"label": "Algorithm", "value": "K-Means", "color": "#fbbf24", "sub": "Demo model" if demo_mode else "Loaded artifacts"},
    ])
    axes = st.columns(3)
    x = axes[0].selectbox("X axis", features, index=0)
    y = axes[1].selectbox("Y axis", features, index=1)
    z = axes[2].selectbox("Z axis", features, index=2)
    view = st.segmented_control("View", ["3D cluster space", "2D relationship", "PCA projection", "Cluster mix", "Cluster map"],
                                default="3D cluster space", key="pat_view", label_visibility="collapsed") or "3D cluster space"
    if view == "3D cluster space":
        fig = px.scatter_3d(frame, x=x, y=y, z=z, color="Cluster", hover_data=hover, color_discrete_sequence=CLUSTER_COLORS, opacity=.82)
        fig.update_traces(marker=dict(size=3.6))
        style_fig(fig, height=640, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)", xaxis=dict(gridcolor="rgba(148,163,184,.15)"),
                                     yaxis=dict(gridcolor="rgba(148,163,184,.15)"), zaxis=dict(gridcolor="rgba(148,163,184,.15)")))
        st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="pat_3d")
    elif view == "2D relationship":
        fig = px.scatter(frame, x=x, y=y, color="Cluster", hover_data=hover, color_discrete_sequence=CLUSTER_COLORS, opacity=.75)
        st.plotly_chart(style_fig(fig, height=540), config=PLOT_CONFIG, theme=None, key="pat_2d")
    elif view == "PCA projection":
        pca = PCA(n_components=3).fit(scaled)
        coords = pca.transform(scaled)
        var = pca.explained_variance_ratio_ * 100
        pdf = pd.DataFrame(coords, columns=["PC1", "PC2", "PC3"]).assign(Cluster=frame["Cluster"].values)
        st.caption(f"Variance explained: {var.sum():.1f}% (PC1 {var[0]:.1f}%, PC2 {var[1]:.1f}%, PC3 {var[2]:.1f}%)")
        fig = px.scatter_3d(pdf, x="PC1", y="PC2", z="PC3", color="Cluster", color_discrete_sequence=CLUSTER_COLORS, opacity=.82)
        fig.update_traces(marker=dict(size=3.6))
        style_fig(fig, height=600, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_layout(scene=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="pat_pca")
    elif view == "Cluster mix":
        c1, c2 = st.columns(2)
        fig = px.pie(values=counts.values, names=counts.index, hole=.6, color_discrete_sequence=CLUSTER_COLORS)
        c1.plotly_chart(style_fig(fig, height=420), config=PLOT_CONFIG, theme=None, key="pat_pie")
        means = frame.groupby("Cluster")[features].mean()
        fig = go.Figure(go.Heatmap(z=(means / means.max()).values, x=features, y=means.index, colorscale="Viridis",
                                   customdata=means.values, hovertemplate="%{y}<br>%{x}: %{customdata:.2f}<extra></extra>"))
        c2.plotly_chart(style_fig(fig, height=420, legend=False), config=PLOT_CONFIG, theme=None, key="pat_means")
    else:
        snap = data["snap"].copy()
        if "Cluster" in snap:
            snap["Profile"] = [f"{c} · {cluster_name(c)}" for c in snap["Cluster"]]
            snap["size"] = snap["AQI"]
            fig = px.scatter_map(snap, lat="lat", lon="lon", color="Profile", size="size", size_max=30, hover_name="city",
                                 hover_data={"AQI": ":.0f", "size": False, "lat": False, "lon": False},
                                 color_discrete_sequence=CLUSTER_COLORS, zoom=3.7, center={"lat": 22.5, "lon": 80.5}, map_style="carto-darkmatter")
            style_fig(fig, height=560, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="pat_map")
            st.caption("Each city coloured by the pattern its latest 24-hour readings belong to.")
    st.caption("Clusters are pattern groups learned by the model, not official AQI categories.")


def page_analyze():
    data = get_data()
    page_header("Instant analysis", "Analyze a sample",
                "Enter pollutant concentrations, or start from a city's model estimates, to find its pattern profile and AQI.", data)
    st.caption("For comparable particulate AQI, enter 24-hour mean PM2.5 and PM10 in µg/m³. CO uses mg/m³. Cluster assignment is not a health classification.")
    presets = {
        "Lower readings": {"PM2.5": 18.0, "PM10": 42.0, "NO2": 12.0, "NOx": 20.0, "NH3": 10.0, "CO": 0.45, "SO2": 7.0, "O3": 22.0},
        "Balanced preview": {"PM2.5": 55.0, "PM10": 95.0, "NO2": 28.0, "NOx": 43.0, "NH3": 18.0, "CO": 0.9, "SO2": 13.0, "O3": 32.0},
        "Higher readings": {"PM2.5": 145.0, "PM10": 245.0, "NO2": 58.0, "NOx": 82.0, "NH3": 34.0, "CO": 2.3, "SO2": 30.0, "O3": 48.0},
    }
    for _, r in data["snap"].sort_values("AQI", ascending=False).iterrows():
        presets[f"{'CAMS' if data['source'] == 'live' else 'Demo'} · {r['city']}"] = {p: round(float(r[p]), 2) for p in DEFAULT_FEATURES}
    preset = st.selectbox("Start from", list(presets), index=1)
    with st.form("sample_form"):
        cols = st.columns(4)
        values = {}
        for i, feat in enumerate(features):
            with cols[i % 4]:
                values[feat] = st.number_input(f"{feat} ({UNITS.get(feat, '')})", min_value=0.0, max_value=10000.0,
                                               value=float(presets[preset].get(feat, 0.0)), step=1.0, key=f"an_{preset}_{feat}")
        submitted = st.form_submit_button("Analyze pollution profile", type="primary", width="stretch")
    if submitted:
        try:
            frame = prepare_input(values)
            st.session_state["last_analysis"] = {"values": frame.iloc[0].to_dict(), "cluster": int(assign_clusters(frame)[0][0])}
        except Exception as e:
            st.error(f"Could not analyze this sample: {e}")
    if "last_analysis" not in st.session_state:
        return
    res = st.session_state["last_analysis"]
    frame = pd.DataFrame([res["values"]], columns=features)
    cid = res["cluster"]
    aqi = compute_aqi(res["values"].get("PM2.5", np.nan), res["values"].get("PM10", np.nan))
    cat, color, advice = aqi_category(aqi)
    c1, c2, c3 = st.columns([1, 1, 1.4], gap="large")
    with c1:
        st.markdown(f"""<div class="result-card"><div class="result-eyebrow">PATTERN PROFILE</div>
            <div class="result-title">Cluster {cid}</div><div class="result-pill">{escape(cluster_name(cid))}</div>
            <p class="result-desc">Assigned to the nearest K-Means centroid across {len(features)} standardized pollutant dimensions.</p></div>""",
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="result-card"><div class="result-eyebrow">ESTIMATED AQI</div>
            <div class="result-title" style="color:{color}">{aqi:.0f}</div>{badge(aqi)}
            <p class="result-desc">{advice}</p></div>""", unsafe_allow_html=True)
    with c3:
        try:
            centroid = scaler.inverse_transform(model.cluster_centers_[[cid]])[0]
            ref = np.maximum(np.maximum(frame.iloc[0].to_numpy(), centroid), 1e-9)
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(r=list(centroid / ref) + [centroid[0] / ref[0]], theta=features + [features[0]],
                                          fill="toself", name="Cluster centre", line_color="#a78bfa"))
            fig.add_trace(go.Scatterpolar(r=list(frame.iloc[0].to_numpy() / ref) + [frame.iloc[0, 0] / ref[0]],
                                          theta=features + [features[0]], fill="toself", name="Your sample", line_color="#38bdf8"))
            style_fig(fig, height=340, margin=dict(l=30, r=30, t=20, b=40))
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.05, x=0.5, xanchor="center"), polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=False)))
            st.plotly_chart(fig, config=PLOT_CONFIG, theme=None, key="an_radar")
        except Exception:
            st.dataframe(frame.T.rename(columns={0: "Value"}), width="stretch")
    similar = data["snap"].copy()
    if "Cluster" in similar:
        matches = similar[similar["Cluster"] == cid]["city"].tolist()
        st.markdown("**Cities currently in this profile:** " + (", ".join(matches) if matches else "none right now"))
    csv = frame.assign(Cluster=cid, Profile=cluster_name(cid), AQI=aqi).to_csv(index=False).encode("utf-8")
    st.download_button("Download this analysis as CSV", csv, "air_pollution_analysis.csv", "text/csv")


def page_batch():
    page_header("Bring your own data", "Batch analysis",
                "Upload a CSV with the model's pollutant columns to cluster many observations at once. Include latitude/longitude to map them.")
    template = make_demo_data(30).round(2)
    template = template.reindex(columns=features, fill_value=0.0)
    rng = np.random.default_rng(3)
    picks = rng.integers(0, len(CITIES), len(template))
    template["latitude"] = [CITIES[i][2] + rng.normal(0, .05) for i in picks]
    template["longitude"] = [CITIES[i][3] + rng.normal(0, .05) for i in picks]
    st.download_button("Download a sample CSV template", template.to_csv(index=False).encode("utf-8"), "airscope_template.csv", "text/csv")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    st.caption("Up to 10 MB / 50,000 rows. Invalid pollutant rows are reported rather than silently discarded. Coordinates are optional.")
    if not uploaded:
        st.info("Required columns: " + ", ".join(features))
        return
    try:
        if uploaded.size > 10 * 1024 * 1024:
            st.error("The file exceeds the 10 MB upload limit.")
            return
        raw = pd.read_csv(uploaded, nrows=50_001)
        if len(raw) > 50_000:
            st.error("The file exceeds the 50,000-row limit. Split it into smaller files.")
            return
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return
    missing = [c for c in features if c not in raw.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    try:
        clean = validate_measurements(raw, features)
    except ValueError as e:
        st.error(str(e))
        return
    results = raw.copy()
    results[features] = clean
    results["Cluster"] = assign_clusters(clean)[0]
    results["Profile"] = [cluster_name(c) for c in results["Cluster"]]
    if "PM2.5" in results and "PM10" in results:
        results["AQI"] = [compute_aqi(a, b) for a, b in zip(results["PM2.5"], results["PM10"])]
    st.success(f"Analyzed {len(clean):,} rows. No pollutant rows were discarded.")
    st.dataframe(results.head(200), width="stretch")
    cols = {str(c).strip().lower(): c for c in results.columns}
    lat_c = next((cols[n] for n in ("latitude", "lat") if n in cols), None)
    lon_c = next((cols[n] for n in ("longitude", "lon", "lng", "long") if n in cols), None)
    if lat_c and lon_c:
        geo = results.assign(lat=pd.to_numeric(results[lat_c], errors="coerce"), lon=pd.to_numeric(results[lon_c], errors="coerce"))
        geo = geo.dropna(subset=["lat", "lon"])
        geo = geo[geo["lat"].between(-90, 90) & geo["lon"].between(-180, 180)]
        if len(geo) < len(results):
            st.warning(f"{len(results) - len(geo):,} rows have missing/invalid coordinates. They remain in the export but are omitted from the map.")
        if len(geo) > 2000:
            geo = geo.sample(2000, random_state=42)
            st.caption("Map shows a reproducible 2,000-row sample; the CSV includes every analyzed row.")
        if not geo.empty:
            st.markdown("### Geographic cluster map")
            geo["Profile"] = [f"{c} · {cluster_name(c)}" for c in geo["Cluster"]]
            fig = px.scatter_map(geo, lat="lat", lon="lon", color="Profile", color_discrete_sequence=CLUSTER_COLORS,
                                 zoom=4, center={"lat": float(geo["lat"].median()), "lon": float(geo["lon"].median())},
                                 map_style="carto-darkmatter", hover_data={c: True for c in ["AQI"] if c in geo})
            fig.update_traces(marker=dict(size=11, opacity=.85))
            st.plotly_chart(style_fig(fig, height=540, margin=dict(l=0, r=0, t=0, b=0)), config=PLOT_CONFIG, theme=None, key="batch_map")
    a, b = st.columns(2)
    counts = results["Profile"].value_counts().reset_index()
    counts.columns = ["Profile", "Samples"]
    a.plotly_chart(style_fig(px.bar(counts, x="Profile", y="Samples", color="Profile", color_discrete_sequence=CLUSTER_COLORS), height=400, legend=False),
                   config=PLOT_CONFIG, theme=None, key="batch_bar")
    if "PM2.5" in results and "PM10" in results:
        b.plotly_chart(style_fig(px.scatter(results.sample(min(len(results), 2000), random_state=42), x="PM2.5", y="PM10", color="Profile", color_discrete_sequence=CLUSTER_COLORS), height=400),
                       config=PLOT_CONFIG, theme=None, key="batch_scatter")
    st.download_button("Download clustered dataset", csv_export(results), "air_pollution_clustered_results.csv", "text/csv")


def page_about():
    page_header("Methodology", "How AirScope works",
                "Data sources, models and how to plug in your own trained artifacts and Google Maps key.")
    steps = [
        ("Live data", "Hourly PM2.5, PM10, NO2, NO, CO, SO2 and O3 for 26 Indian cities from the Copernicus Atmosphere Monitoring Service (CAMS) via the free Open-Meteo API. NOx = NO2 + NO; NH3 is estimated from PM2.5 where CAMS has no ammonia."),
        ("AQI", "Particulate-only index using US EPA May 2024 breakpoints: truncate PM2.5 to 0.1 µg/m³ and PM10 to integer µg/m³, interpolate within discrete bands, round half up, take the larger index. Display is capped at 500. Not India's CPCB AQI or a full six-pollutant AQI."),
        ("Pattern discovery", "A StandardScaler + K-Means model assigns each reading to a learned atmospheric profile (your saved model is used automatically when present)."),
        ("Forecasting", "A ridge-regression model on log-scaled lagged pollutant levels (1, 2, 3 and 7 days, rolling means, seasonality and location) is trained across all cities and rolled forward recursively for 14 days."),
        ("Ensemble outlook", "ML is averaged with CAMS on overlapping forecast dates; later dates use ML alone. The shaded range belongs to ML alone: a heuristic based on holdout log residuals, not calibrated coverage. MAE and skill scores evaluate ML, not the ensemble, against archived CAMS estimates."),
        ("Maps", "Interactive maps run on MapLibre with CARTO and Esri tiles. Google Maps embeds work without a key; add a Maps JavaScript API key for full multi-city AQI markers on Google Maps."),
    ]
    st.markdown("".join(f'<div class="step"><div class="step-n">{i}</div><div><b>{t}</b><div>{d}</div></div></div>'
                        for i, (t, d) in enumerate(steps, 1)), unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("### Use your trained clustering model")
        st.markdown("Place these files in the `models/` folder next to `app.py` and restart:\n"
                    "- `air_pollution_model.pkl`\n- `air_pollution_scaler.pkl`\n- `feature_columns.pkl`\n- `cluster_profiles.pkl` (optional labels)")
        st.warning("Only load `.pkl` files from a trusted source. Pickle files can execute code when loaded.")
    with c2:
        st.markdown("### Enable full Google Maps")
        st.markdown("1. Create a key in Google Cloud Console and enable the **Maps JavaScript API**.\n"
                    "2. Paste it into the sidebar, or set it permanently in `.streamlit/secrets.toml`:")
        st.code('GOOGLE_MAPS_API_KEY = "your-key-here"', language="toml")
        st.markdown("Or as an environment variable: `GOOGLE_MAPS_API_KEY`.")
    st.markdown("### Current status")
    st.write(f"**Clustering model:** {'Synthetic demo model' if demo_mode else 'Loaded artifacts'} · **Features:** {', '.join(features)}")
    st.markdown("### Sources & limitations")
    st.markdown("- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) / "
                "[Copernicus CAMS](https://atmosphere.copernicus.eu/) (model estimates, not ground-station readings).\n"
                "- [US EPA AQI Technical Assistance Document, May 2024](https://www.airnow.gov/sites/default/files/2024-05/aqi-technical-assistance-document-may2024.pdf).\n"
                "- NH3 is estimated where unavailable; NOx uses NO2 alone if NO is missing. These proxies affect clustering and forecasting.\n"
                "- All-cities summaries are unweighted city averages, not population-weighted national statistics.\n"
                "- Only complete 24-hour days are used for daily forecasting. Custom locations use their local timezone; the pooled model is trained on Indian cities and may generalize poorly elsewhere.\n"
                "- Negative model skill means persistence performed better. Synthetic fallback metrics do not measure real-world accuracy.\n"
                "- Open-Meteo's free endpoint is for non-commercial use. Review its [terms](https://open-meteo.com/en/terms) before commercial deployment.")


# ============================================================================
# App shell
# ============================================================================
pages = [
    st.Page(page_dashboard, title="Overview", icon=":material/space_dashboard:", url_path="dashboard", default=True),
    st.Page(page_map, title="Map", icon=":material/map:", url_path="map"),
    st.Page(page_forecast, title="Forecast", icon=":material/insights:", url_path="forecast"),
    st.Page(page_patterns, title="Patterns", icon=":material/scatter_plot:", url_path="patterns"),
    st.Page(page_analyze, title="Analyze", icon=":material/science:", url_path="analyze"),
    st.Page(page_batch, title="Batch upload", icon=":material/upload_file:", url_path="batch"),
    st.Page(page_dataset_lab, title="Dataset lab", icon=":material/model_training:", url_path="dataset-lab"),
    st.Page(page_about, title="Methodology", icon=":material/info:", url_path="about"),
]
navigation = st.navigation(pages, position="hidden")
with st.sidebar:
    st.html("""<div class="brand"><div class="brand-mark">A</div>
        <div><div class="brand-name">AirScope<span style="color:#b5e86f"> 2</span></div>
        <div class="brand-sub">Environmental intelligence</div></div></div>""")
    st.html('<div class="nav-label">INTELLIGENCE</div>')
    for page in pages[:4]:
        st.page_link(page)
    st.html('<div class="nav-label">WORKSPACE</div>')
    for page in pages[4:]:
        st.page_link(page)
    st.markdown("---")
    st.html(f"""<div class="side-status"><b>{'Demo clustering model' if demo_mode else 'Clustering model ready'}</b>
        <small>{'Synthetic K-Means · add your artifacts' if demo_mode else 'Trusted saved artifacts loaded'}</small></div>""")
    if artifact_error:
        st.warning(artifact_error)
    with st.expander("Connection & settings", icon=":material/settings:"):
        st.caption("CAMS refreshes every 30 minutes. Refresh clears shared data caches for all visitors.")
        if st.button("Refresh data", icon=":material/refresh:", width="stretch"):
            st.cache_data.clear()
            get_data.clear()
            train_forecaster.clear()
            st.rerun()
        st.text_input("Google Maps API key", type="password", key="gmaps_key",
                      help="Browser-visible key. Restrict its HTTP referrers and API scope in Google Cloud.",
                      placeholder="Using env/secrets" if google_maps_key() else "Optional browser key")
    st.caption("CAMS / Open-Meteo · Model estimates\n\nExploratory tool, not an official advisory.")

navigation.run()

st.markdown("""<div class="app-footer"><span><b>AirScope</b> · Air quality intelligence for India</span>
<span>Data: Copernicus CAMS via Open-Meteo · Forecasts are indicative, not official advisories</span></div>""", unsafe_allow_html=True)
