import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
MODEL_PATH = MODEL_DIR / "air_pollution_model.pkl"
SCALER_PATH = MODEL_DIR / "air_pollution_scaler.pkl"
FEATURES_PATH = MODEL_DIR / "feature_columns.pkl"
PROFILES_PATH = MODEL_DIR / "cluster_profiles.pkl"

st.set_page_config(
    page_title="AirScope | Pollution Pattern Discovery",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#eaf3ff; --muted:#a9bbd0; --panel:rgba(17,31,55,.78); --line:rgba(148,180,218,.18); --cyan:#4de1d2; --blue:#6ba8ff; }
.stApp {
  background: radial-gradient(ellipse at 12% 4%, rgba(40,119,167,.28), transparent 32%),
              radial-gradient(ellipse at 88% 12%, rgba(21,164,153,.17), transparent 30%),
              linear-gradient(135deg,#07111f 0%,#0a1729 48%,#0c1b2e 100%);
  color:var(--ink); font-family:'DM Sans',sans-serif;
}
[data-testid="stHeader"] { background:rgba(7,17,31,.78); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1728 0%,#101f35 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:#eaf3ff; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#a9bbd0; }
[data-testid="stSidebar"] .stRadio label { padding: .35rem .2rem; transition:transform .2s ease,color .2s ease; }
[data-testid="stSidebar"] .stRadio label:hover { transform:translateX(4px); color:var(--cyan)!important; }
.block-container { padding-top:2.2rem; padding-bottom:3rem; max-width:1500px; animation:pageIn .7s ease both; }
@keyframes pageIn { from {opacity:0; transform:translateY(12px)} to {opacity:1; transform:translateY(0)} }
@keyframes floatGlow { 0%,100% {transform:translateY(0) scale(1)} 50% {transform:translateY(-7px) scale(1.015)} }
@keyframes pulseDot { 0%,100% {box-shadow:0 0 0 0 rgba(77,225,210,.5)} 50% {box-shadow:0 0 0 8px rgba(77,225,210,0)} }
.hero {
  position:relative; overflow:hidden; isolation:isolate;
  background:linear-gradient(115deg,rgba(18,47,79,.96),rgba(14,83,112,.9) 52%,rgba(10,119,117,.88));
  padding:2.1rem 2.2rem; border:1px solid rgba(140,220,240,.25); border-radius:24px;
  color:white; margin-bottom:1.3rem; box-shadow:0 18px 60px rgba(0,0,0,.22); animation:floatGlow 7s ease-in-out infinite;
}
.hero:before { content:''; position:absolute; z-index:-1; width:290px;height:290px;right:-65px;top:-140px;border-radius:50%;background:rgba(77,225,210,.16);filter:blur(2px); }
.hero:after { content:''; position:absolute; z-index:-1; width:180px;height:180px;right:150px;bottom:-135px;border-radius:50%;background:rgba(107,168,255,.2); }
.hero h1 { color:white; margin:0 0 .45rem 0; font-family:'Space Grotesk',sans-serif; font-size:clamp(2.2rem,4vw,3.5rem); letter-spacing:-.055em; }
.hero p { color:#d9f3fa; margin:0; font-size:1.02rem; }
.hero-kicker { color:#7ef5e5; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin-bottom:.55rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(21,39,65,.94),rgba(12,27,46,.92)); border:1px solid var(--line); padding:18px 18px; border-radius:18px; box-shadow:0 10px 28px rgba(0,0,0,.12); transition:transform .22s ease,border-color .22s ease,box-shadow .22s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-4px); border-color:rgba(77,225,210,.48); box-shadow:0 16px 34px rgba(0,0,0,.2); }
[data-testid="stMetricLabel"] { color:#a9bbd0!important; }
[data-testid="stMetricValue"] { color:#f5fbff!important; font-family:'Space Grotesk',sans-serif; }
[data-testid="stPlotlyChart"] { background:rgba(14,29,49,.65); border:1px solid var(--line); border-radius:18px; padding:8px; box-shadow:0 12px 30px rgba(0,0,0,.12); transition:transform .22s ease; }
[data-testid="stPlotlyChart"]:hover { transform:translateY(-2px); }
[data-testid="stDataFrame"], [data-testid="stFileUploader"] section { border:1px solid var(--line); border-radius:16px; }
.stMarkdown h2,.stMarkdown h3,.stMarkdown h4 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; color:#eaf3ff; }
.stMarkdown p,.stCaption { color:#b6c7d9; }
.stButton>button, .stDownloadButton>button, [data-testid="stFormSubmitButton"] button { border:1px solid rgba(77,225,210,.32); border-radius:12px; font-weight:700; color:#061a28; background:linear-gradient(105deg,#74f0df,#75b7ff); transition:transform .18s ease,box-shadow .18s ease,filter .18s ease; }
.stButton>button:hover, .stDownloadButton>button:hover, [data-testid="stFormSubmitButton"] button:hover { transform:translateY(-2px); box-shadow:0 8px 24px rgba(77,225,210,.18); filter:brightness(1.06); }
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] { background:#10213a; color:#eff8ff; border-color:#29435f; border-radius:10px; }
[data-testid="stAlert"] { border-radius:14px; }
hr { border-color:var(--line); }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation:none!important; transition:none!important; scroll-behavior:auto!important; } }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🌫️ AirScope")
st.sidebar.caption("✦  AIR POLLUTION INTELLIGENCE")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Analyze a sample", "Explore dataset", "About the model"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model status**")
if MODEL_PATH.exists() and SCALER_PATH.exists() and FEATURES_PATH.exists():
    st.sidebar.success("Model files detected")
else:
    st.sidebar.warning("Demo model files missing")
st.sidebar.caption("Demo mode uses synthetic example data. Do not treat it as official air-quality advice.")

# The features must match the model's training feature list.
DEFAULT_FEATURES = ["PM2.5", "PM10", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]

def load_artifacts():
    needed = [MODEL_PATH, SCALER_PATH, FEATURES_PATH]
    if not all(p.exists() for p in needed):
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
    return model, scaler, list(features), profiles

artifacts = load_artifacts()

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
    data = np.vstack(blocks)
    data = np.clip(data, [0, 0, 0, 0, 0, .05, 0, 0], None)
    return pd.DataFrame(data, columns=DEFAULT_FEATURES)

def pollution_label(profile, cluster_id):
    if profile is not None and isinstance(profile, dict):
        label = profile.get(int(cluster_id), {}).get("label")
        if label:
            return label
    return f"Cluster {int(cluster_id)}"

def infer_profile_labels(data, labels):
    """Provide tentative descriptive labels based on relative PM2.5 cluster means."""
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

if artifacts:
    model, scaler, features, profiles = artifacts
    demo_mode = False
else:
    # Create a temporary, deterministic demo model in memory if artifacts are missing.
    demo_frame = make_demo_data()
    features = DEFAULT_FEATURES
    scaler = StandardScaler()
    scaled = scaler.fit_transform(demo_frame[features])
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    model.fit(scaled)
    profiles = None
    demo_mode = True

def prepare_input(values):
    frame = pd.DataFrame([values], columns=features)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Please enter a valid numeric value for every required pollutant.")
    if (frame < 0).any().any():
        raise ValueError("Pollutant measurements cannot be negative.")
    return frame

def assign_clusters(frame):
    scaled = scaler.transform(frame[features])
    labels = model.predict(scaled)
    return labels, scaled

def show_hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">AIR QUALITY • MACHINE INTELLIGENCE</div>
      <h1>AirScope<span style="color:#7ef5e5">.</span></h1>
      <p>Discover hidden pollution patterns with unsupervised machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

if page == "Overview":
    show_hero()
    if demo_mode:
        st.info("You are viewing the temporary demo model. It is trained on synthetic sample values, not official monitoring data. Replace the model files in the models folder when your friend's trained files are ready.")
    else:
        st.success("Your saved model artifacts are loaded from the models folder.")
    demo = make_demo_data(650)
    labels, _ = assign_clusters(demo[features])
    demo["Cluster"] = labels.astype(str)
    cluster_counts = demo["Cluster"].value_counts()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Samples analyzed", f"{len(demo):,}")
    c2.metric("Discovered clusters", str(len(set(labels))))
    c3.metric("Features used", str(len(features)))
    c4.metric("Model type", "K-Means")
    st.markdown("### Pollution profile overview")
    left, right = st.columns([1.15, 1])
    with left:
        fig = px.scatter(
            demo, x="PM2.5", y="PM10", color="Cluster",
            hover_data=["NO2", "CO", "SO2", "O3"],
            title="PM2.5 vs PM10 by discovered cluster",
            template="plotly_white",
        )
        fig.update_layout(legend_title_text="Cluster", margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig2 = px.pie(
            values=cluster_counts.values, names=cluster_counts.index,
            hole=.55, title="Share of samples in each cluster", template="plotly_white"
        )
        fig2.update_layout(margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)
    st.caption("Charts on this page use generated demo samples to preview the dashboard. They are not measurements from real cities.")

elif page == "Analyze a sample":
    show_hero()
    st.write("Enter pollutant measurements to assign the sample to one of the model's discovered clusters.")
    if demo_mode:
        st.warning("Demo model active: results are illustrative only. The real model will be used automatically once valid model artifacts are placed in `models/`.")
    with st.form("sample_form"):
        st.markdown("#### Pollutant concentrations")
        default_values = {
            "PM2.5": 55.0, "PM10": 95.0, "NO2": 28.0, "NOx": 43.0,
            "NH3": 18.0, "CO": 0.9, "SO2": 13.0, "O3": 32.0
        }
        cols = st.columns(4)
        values = {}
        for i, feature in enumerate(features):
            with cols[i % 4]:
                values[feature] = st.number_input(
                    feature, min_value=0.0, max_value=10000.0,
                    value=float(default_values.get(feature, 0.0)), step=1.0,
                    help="Use the same units and measurement conventions as the model's training data."
                )
        submitted = st.form_submit_button("Analyze pollution profile", type="primary", use_container_width=True)
    if submitted:
        try:
            frame = prepare_input(values)
            labels, scaled = assign_clusters(frame)
            cluster_id = int(labels[0])
            st.markdown("---")
            st.markdown("### Analysis result")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Assigned cluster", f"Cluster {cluster_id}")
                st.write(pollution_label(profiles, cluster_id))
            with col2:
                st.write("**Input summary**")
                st.dataframe(frame.T.rename(columns={0: "Your value"}), use_container_width=True)
            st.caption("A cluster is a pattern group learned by the model, not an official AQI category or health-risk diagnosis.")
            csv = frame.assign(Cluster=cluster_id).to_csv(index=False).encode("utf-8")
            st.download_button("Download this analysis as CSV", csv, "air_pollution_analysis.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not analyze this sample: {e}")

elif page == "Explore dataset":
    show_hero()
    st.write("Upload a CSV file containing the model's pollutant feature columns to cluster many observations at once.")
    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            missing = [c for c in features if c not in raw.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
                st.write("Required features:", ", ".join(features))
            else:
                numeric = raw[features].apply(pd.to_numeric, errors="coerce")
                valid_mask = numeric.notna().all(axis=1) & (numeric >= 0).all(axis=1)
                clean = numeric.loc[valid_mask].copy()
                if clean.empty:
                    st.error("No valid rows remain after checking missing and negative values.")
                else:
                    labels, scaled = assign_clusters(clean)
                    results = raw.loc[clean.index].copy()
                    results["Cluster"] = labels
                    st.success(f"Analyzed {len(clean):,} rows; skipped {len(raw) - len(clean):,} rows with missing or negative required values.")
                    st.dataframe(results.head(100), use_container_width=True)
                    a, b = st.columns(2)
                    with a:
                        counts = results["Cluster"].astype(str).value_counts().rename_axis("Cluster").reset_index(name="Samples")
                        fig = px.bar(counts, x="Cluster", y="Samples", title="Samples per cluster", template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    with b:
                        if "PM2.5" in results and "PM10" in results:
                            fig = px.scatter(results, x="PM2.5", y="PM10", color=results["Cluster"].astype(str),
                                             title="PM2.5 vs PM10", template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                    st.download_button("Download clustered dataset", results.to_csv(index=False).encode("utf-8"),
                                       "air_pollution_clustered_results.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not read or analyze the file: {e}")
    else:
        st.info("For the demo, use a CSV with these columns: " + ", ".join(features))

elif page == "About the model":
    show_hero()
    st.markdown("### How the system works")
    st.markdown("""
    1. **Input:** pollutant concentration measurements.
    2. **Preprocessing:** the saved scaler transforms values using the training-time scaling.
    3. **Clustering:** K-Means assigns each observation to its nearest learned centroid.
    4. **Interpretation:** pollutant profiles can be compared after clustering to describe the groups.
    """)
    st.markdown("### Current model status")
    st.write("**Mode:** " + ("Temporary synthetic demo model" if demo_mode else "Loaded model artifacts"))
    st.write("**Algorithm:** K-Means clustering")
    st.write("**Features expected:** " + ", ".join(features))
    st.markdown("### Replace the demo model")
    st.markdown("""
    Put the three files from your friend into the `models` folder using these exact names:
    - `air_pollution_model.pkl`
    - `air_pollution_scaler.pkl`
    - `feature_columns.pkl`

    Restart Streamlit. The app will load those artifacts automatically. The scaler and feature list must be from the same training run as the model.
    """)
    st.warning("Only load `.pkl` files from a trusted source. Pickle files can execute code when loaded.")
