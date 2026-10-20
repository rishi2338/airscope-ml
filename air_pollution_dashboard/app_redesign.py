from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AirScope | Environmental Intelligence", page_icon="A", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root{--bg:#07111b;--panel:#0d1b29;--line:rgba(160,195,211,.16);--ink:#edf8fb;--muted:#9bb2c0;--teal:#62e6d1;--blue:#8ac8ff}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:radial-gradient(ellipse at 12% 0%,rgba(28,116,139,.22),transparent 35%),linear-gradient(145deg,#06101a,#091724 55%,#07131d);color:var(--ink)}
[data-testid="stHeader"]{background:rgba(5,15,24,.72);backdrop-filter:blur(14px)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#081521,#0b1c2a);border-right:1px solid var(--line)}
.block-container{max-width:1480px;padding-top:2rem;padding-bottom:3rem;animation:rise .5s ease both}
@keyframes rise{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
.hero{padding:clamp(1.5rem,3vw,2.7rem);border:1px solid rgba(98,230,209,.24);border-radius:24px;background:linear-gradient(112deg,#0e2636,#0b404b 62%,#0c5853);margin-bottom:1.4rem}
.kicker{color:var(--teal);font-size:.72rem;font-weight:700;letter-spacing:.18em;text-transform:uppercase}
.hero h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(2.7rem,5vw,4.4rem);line-height:1;letter-spacing:-.06em;margin:.65rem 0}
.hero p{max-width:700px;color:#c7e4e9;line-height:1.7}
[data-testid="stMetric"]{background:linear-gradient(145deg,#102535,#0a1a27);border:1px solid var(--line);border-radius:17px;padding:1rem;transition:transform .2s,border-color .2s}
[data-testid="stMetric"]:hover{transform:translateY(-3px);border-color:rgba(98,230,209,.4)}
[data-testid="stMetricLabel"]{color:var(--muted)!important}
[data-testid="stMetricValue"],h1,h2,h3{font-family:'Space Grotesk',sans-serif}
[data-testid="stPlotlyChart"]{border:1px solid var(--line);border-radius:17px;padding:.35rem;background:rgba(10,26,39,.6)}
.stButton>button,.stDownloadButton>button{border:1px solid rgba(98,230,209,.3);border-radius:11px;font-weight:700;color:#061a22;background:linear-gradient(105deg,#7af0dc,#8ac8ff)}
.stButton>button:hover,.stDownloadButton>button:hover{transform:translateY(-2px)}
hr{border-color:var(--line)}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}
</style>
""", unsafe_allow_html=True)

POLLUTANTS = ["PM2.5", "PM10", "NO2", "NOx", "NH3", "CO", "SO2", "O3"]
CITIES = ["Chennai", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Pune", "Madurai"]

def demo_data(n=480, seed=42):
    rng = np.random.default_rng(seed)
    city = rng.choice(CITIES, n)
    regimes = rng.choice([0, 1, 2], n, p=[.42, .36, .22])
    base = np.array([22, 48, 18, 25, 12, .45, 8, 28], dtype=float)
    offsets = np.array([[0,0,0,0,0,0,0,0],[42,65,20,27,8,.55,5,8],[95,120,40,48,20,1.2,14,18]])
    values = base + offsets[regimes] + rng.normal(0, [8,15,5,8,4,.18,3,9], (n, 8))
    frame = pd.DataFrame(np.maximum(values, .02), columns=POLLUTANTS)
    frame.insert(0, "City", city)
    frame["Sample"] = np.arange(1, n + 1)
    return frame

def cluster_data(frame, count):
    scaler = StandardScaler()
    x = scaler.fit_transform(frame[POLLUTANTS])
    model = KMeans(n_clusters=count, random_state=42, n_init=10)
    labels = model.fit_predict(x)
    projected = PCA(n_components=2, random_state=42).fit_transform(x)
    result = frame.copy()
    result["Pattern"] = labels.astype(str)
    result["PC1"], result["PC2"] = projected[:, 0], projected[:, 1]
    return result

st.sidebar.markdown("## AirScope")
st.sidebar.caption("ENVIRONMENTAL INTELLIGENCE")
page = st.sidebar.radio("Navigate", ["Overview", "Explore patterns", "Analyze data", "Methodology"])
st.sidebar.markdown("---")
st.sidebar.caption("Exploratory analysis only. Demo records are synthetic, not official monitoring-station measurements.")

st.markdown("""
<div class="hero">
  <div class="kicker">Air quality · Pattern discovery</div>
  <h1>See the air<br>in a new light.</h1>
  <p>Explore pollutant relationships, discover recurring air-quality patterns, and turn environmental data into clearer decisions.</p>
</div>
""", unsafe_allow_html=True)

if "air_data" not in st.session_state:
    st.session_state.air_data = demo_data()
data = st.session_state.air_data

if page == "Overview":
    st.caption("AIR QUALITY INTELLIGENCE / OVERVIEW")
    a, b, c, d = st.columns(4)
    a.metric("Records", f"{len(data):,}")
    b.metric("Pollutants", len(POLLUTANTS))
    c.metric("Cities represented", data["City"].nunique())
    d.metric("Analysis mode", "Demo data" if "Source" not in data else "Uploaded data")
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Pollutant profile")
        profile = data[POLLUTANTS].mean().sort_values().rename_axis("Pollutant").reset_index(name="Mean")
        fig = px.bar(profile, x="Mean", y="Pollutant", orientation="h", color="Mean", color_continuous_scale=["#287b87", "#62e6d1"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Pollutant relationships")
        corr = data[POLLUTANTS].corr()
        fig = px.imshow(corr, text_auto=".1f", color_continuous_scale="Tealrose", zmin=-1, zmax=1, aspect="auto")
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent records")
    st.dataframe(data.head(12), use_container_width=True, hide_index=True)

elif page == "Explore patterns":
    st.caption("UNSUPERVISED LEARNING / CLUSTER EXPLORER")
    k = st.slider("Number of patterns (clusters)", 2, 6, 3)
    clustered = cluster_data(data, k)
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Pattern map")
        fig = px.scatter(clustered, x="PC1", y="PC2", color="Pattern", hover_data=["City"] + POLLUTANTS[:3], color_discrete_sequence=["#62e6d1", "#8ac8ff", "#f3b66d", "#d69cff", "#ff829d", "#b9d6a5"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=5,r=5,t=15,b=5))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Pattern signatures")
        means = clustered.groupby("Pattern")[POLLUTANTS].mean().round(2)
        st.dataframe(means, use_container_width=True)
    st.caption("K-Means groups records by similarity after standardization. PCA projects multiple pollutant dimensions into two axes for exploration; clusters are not official AQI categories.")

elif page == "Analyze data":
    st.caption("YOUR DATA / CSV ANALYSIS")
    uploaded = st.file_uploader("Upload a CSV containing pollutant columns", type=["csv"])
    if uploaded:
        try:
            incoming = pd.read_csv(uploaded)
            incoming.columns = [str(col).strip() for col in incoming.columns]
            present = [col for col in POLLUTANTS if col in incoming.columns]
            if len(present) < 2:
                st.error("The CSV needs at least two recognized pollutant columns: " + ", ".join(POLLUTANTS))
            else:
                clean = incoming.copy()
                for col in present:
                    clean[col] = pd.to_numeric(clean[col], errors="coerce")
                clean = clean.dropna(subset=present)
                if len(clean) < 3:
                    st.error("At least three complete rows are required for analysis.")
                else:
                    st.success(f"Loaded {len(clean):,} usable rows and {len(present)} pollutant features.")
                    st.dataframe(clean.head(20), use_container_width=True, hide_index=True)
                    st.download_button("Download cleaned CSV", clean.to_csv(index=False).encode("utf-8"), "airscope_cleaned_data.csv", "text/csv")
                    st.subheader("Pollutant distribution")
                    selected = st.selectbox("Choose a pollutant", present)
                    fig = px.histogram(clean, x=selected, nbins=30, marginal="box", color_discrete_sequence=["#62e6d1"])
                    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                    if len(present) >= 2:
                        if st.button("Discover patterns in uploaded data"):
                            work = clean[present].copy()
                            for col in POLLUTANTS:
                                if col not in work:
                                    work[col] = 0.0
                            temp = clean.copy()
                            for col in POLLUTANTS:
                                temp[col] = clean[col] if col in clean else 0.0
                            result = cluster_data(temp, min(4, max(2, len(temp) // 2)))
                            st.dataframe(result[["Pattern"] + present].head(50), use_container_width=True)
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    else:
        st.info("Upload a CSV to inspect pollutant columns, review distributions, clean missing values, and export usable records.")

else:
    st.caption("MODEL NOTES / TRANSPARENCY")
    st.subheader("How AirScope works")
    st.markdown("""
    - **Input:** pollutant concentration columns from a CSV or clearly labelled synthetic demo records.
    - **Preparation:** numeric conversion, missing-value handling, and feature standardization.
    - **Pattern discovery:** K-Means groups records with similar pollutant profiles.
    - **Visualization:** PCA compresses the feature space into two dimensions for inspection.
    - **Interpretation:** clusters indicate statistical similarity, not causation, regulatory classifications, or medical advice.
    """)
    st.warning("This redesign uses synthetic sample records by default. Do not present demo values as live sensor readings or official city measurements.")
    st.markdown("**Libraries:** Streamlit · Pandas · NumPy · Scikit-learn · Plotly")
    st.markdown("**Accessibility:** animations respect the system's reduced-motion preference.")
