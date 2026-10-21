"""Independent multi-dataset K-Means exploration for AirScope."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000
MAX_FILES = 8


def _read_dataset(upload):
    if upload.size > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 10 MB per-file limit.")
    suffix = upload.name.lower().rsplit(".", 1)[-1]
    if suffix == "csv":
        frame = pd.read_csv(upload, low_memory=False, nrows=MAX_ROWS + 1)
    elif suffix in {"xlsx", "xls"}:
        frame = pd.read_excel(upload, nrows=MAX_ROWS + 1)
    else:
        raise ValueError("Unsupported file type.")
    if len(frame) > MAX_ROWS:
        raise ValueError(f"File exceeds the {MAX_ROWS:,}-row limit.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    return frame


def _numeric_features(frame):
    numeric = frame.select_dtypes(include="number").copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.dropna(axis=1, how="all")

    # Exclude likely row identifiers, which usually add noise to distance-based clustering.
    excluded = []
    for column in numeric.columns:
        normalized = str(column).strip().lower().replace(" ", "_")
        unique_ratio = numeric[column].nunique(dropna=True) / max(1, numeric[column].notna().sum())
        sequential_index = False
        if len(numeric) >= 50 and numeric[column].notna().all():
            values = numeric[column].to_numpy()
            sequential_index = (
                np.array_equal(values, np.arange(len(values)))
                or np.array_equal(values, np.arange(1, len(values) + 1))
            )
        if (
            normalized in {"id", "index", "row_id", "record_id"}
            or normalized.startswith("unnamed:")
            or sequential_index
        ):
            excluded.append(column)
    numeric = numeric.drop(columns=excluded, errors="ignore")
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    return numeric, excluded


def _analyze(frame):
    numeric, excluded = _numeric_features(frame)
    valid = numeric.notna().any(axis=1)
    numeric = numeric.loc[valid]
    source = frame.loc[numeric.index].copy()

    if numeric.shape[1] < 2:
        raise ValueError("At least two useful numerical features are required.")
    if len(numeric) < 3:
        raise ValueError("At least three usable rows are required.")

    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(numeric)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    max_k = min(8, len(numeric) - 1)
    candidates = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        if len(np.unique(labels)) < 2:
            continue
        if len(scaled) > 10_000:
            score = silhouette_score(
                scaled, labels, sample_size=10_000, random_state=42
            )
        else:
            score = silhouette_score(scaled, labels)
        candidates.append((k, model, labels, float(score)))

    if not candidates:
        raise ValueError("Could not find a valid clustering configuration.")

    best_k, best_model, labels, best_score = max(candidates, key=lambda item: item[3])
    results = source.copy()
    results["Cluster"] = labels
    profile_source = numeric.copy()
    profile_source["Cluster"] = labels
    profiles = profile_source.groupby("Cluster").mean(numeric_only=True)

    pca = PCA(n_components=2, random_state=42)
    coordinates = pca.fit_transform(scaled)
    projection = pd.DataFrame({
        "Component 1": coordinates[:, 0],
        "Component 2": coordinates[:, 1],
        "Cluster": labels.astype(str),
    })
    scores = pd.DataFrame({
        "K": [item[0] for item in candidates],
        "Inertia": [item[1].inertia_ for item in candidates],
        "Silhouette score": [item[3] for item in candidates],
    })

    return {
        "results": results,
        "profiles": profiles,
        "features": list(numeric.columns),
        "excluded": [str(c) for c in excluded],
        "best_k": best_k,
        "silhouette": best_score,
        "scores": scores,
        "projection": projection,
        "explained": pca.explained_variance_ratio_ * 100,
    }


def page_dataset_lab():
    st.markdown(
        '<div class="hero"><div class="eyebrow">AIRScope / UNSUPERVISED LEARNING</div>'
        '<h1>Multi-dataset ML lab</h1>'
        '<p class="lead">Run an independent K-Means experiment on every dataset you upload. '
        'Compare cluster quality, inspect feature profiles, and export all results.</p></div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "This lab trains a separate K-Means model for each dataset. "
        "For a hosted app, upload files from your computer; the app cannot directly read "
        "a private local folder such as D:\\Machine learning project\\data sets."
    )
    uploads = st.file_uploader(
        "Upload all datasets",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="CSV and Excel files; up to 8 files, 10 MB and 50,000 rows per file.",
    )
    st.caption("Numerical features are scaled before clustering. Likely ID columns and constant columns are excluded.")
    if not uploads:
        st.info("Upload one or more datasets to begin. Each dataset is clustered independently.")
        return
    if len(uploads) > MAX_FILES:
        st.error(f"Upload up to {MAX_FILES} datasets at a time.")
        return

    reports = {}
    summary = []
    progress = st.progress(0, text="Preparing datasets…")
    for index, upload in enumerate(uploads):
        try:
            frame = _read_dataset(upload)
            report = _analyze(frame)
            reports[upload.name] = report
            summary.append({
                "Dataset": upload.name,
                "Rows analyzed": len(report["results"]),
                "Features": len(report["features"]),
                "Recommended K": report["best_k"],
                "Silhouette score": report["silhouette"],
                "Status": "Completed",
            })
        except Exception as exc:
            summary.append({
                "Dataset": upload.name,
                "Rows analyzed": 0,
                "Features": 0,
                "Recommended K": None,
                "Silhouette score": None,
                "Status": f"Skipped: {exc}",
            })
        progress.progress((index + 1) / len(uploads), text=f"Processed {index + 1} of {len(uploads)} datasets")
    progress.empty()

    summary_frame = pd.DataFrame(summary)
    st.markdown("### Batch summary")
    done = len(reports)
    a, b, c = st.columns(3)
    a.metric("Datasets completed", f"{done} / {len(uploads)}")
    c.metric("Rows analyzed", f"{sum(len(r['results']) for r in reports.values()):,}")
    b.metric("Total clusters", sum(r["best_k"] for r in reports.values()))
    st.dataframe(
        summary_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "Silhouette score": st.column_config.NumberColumn(format="%.4f"),
            "Recommended K": st.column_config.NumberColumn(format="%d"),
        },
    )

    if not reports:
        st.warning("None of the uploaded datasets could be clustered. Review the status column above.")
        return

    archive_buffer = BytesIO()
    with ZipFile(archive_buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("all_datasets_summary.csv", summary_frame.to_csv(index=False))
        for name, report in reports.items():
            stem = name.rsplit(".", 1)[0]
            safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]
            prefix = safe_stem or "dataset"
            archive.writestr(f"{prefix}/clustered_data.csv", report["results"].to_csv(index=False))
            archive.writestr(f"{prefix}/cluster_profiles.csv", report["profiles"].to_csv())
            archive.writestr(f"{prefix}/cluster_scores.csv", report["scores"].to_csv(index=False))
    st.download_button(
        "Download all results (.zip)",
        data=archive_buffer.getvalue(),
        file_name="airscope_multi_dataset_kmeans.zip",
        mime="application/zip",
        type="primary",
        icon=":material/download:",
    )

    selected_name = st.selectbox("Explore a dataset", list(reports), key="dataset_lab_selected")
    report = reports[selected_name]
    st.markdown("### Clustering diagnostics")
    k1, k2, k3 = st.columns(3)
    k1.metric("Recommended clusters", report["best_k"])
    k2.metric("Silhouette score", f"{report['silhouette']:.4f}")
    k3.metric("Numerical features", len(report["features"]))
    st.caption(
        "Silhouette scores range from −1 to 1; higher values generally indicate better-separated clusters. "
        "This is a recommendation, not proof that the clusters are meaningful."
    )
    if report["excluded"]:
        st.caption("Excluded likely identifier columns: " + ", ".join(report["excluded"]))

    tab_projection, tab_elbow, tab_silhouette, tab_profiles, tab_data = st.tabs(
        ["PCA projection", "Elbow method", "Silhouette", "Cluster profiles", "Clustered rows"]
    )
    with tab_projection:
        projection = report["projection"]
        fig = px.scatter(
            projection, x="Component 1", y="Component 2", color="Cluster",
            color_discrete_sequence=["#38bdf8", "#a78bfa", "#fbbf24", "#34d399", "#f472b6", "#fb7185", "#60a5fa", "#fb923c"],
            opacity=0.78, title="Clusters projected into two PCA dimensions",
        )
        fig.update_traces(marker=dict(size=7))
        st.plotly_chart(fig, use_container_width=True, theme=None)
        st.caption(
            f"Explained variance: PC1 {report['explained'][0]:.1f}% · "
            f"PC2 {report['explained'][1]:.1f}%."
        )
    with tab_elbow:
        fig = px.line(report["scores"], x="K", y="Inertia", markers=True, title="Elbow method")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with tab_silhouette:
        fig = px.line(report["scores"], x="K", y="Silhouette score", markers=True, title="Silhouette score by K")
        fig.add_vline(x=report["best_k"], line_dash="dot", line_color="#34d399")
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with tab_profiles:
        st.dataframe(report["profiles"].round(3), width="stretch")
        fig = px.imshow(
            report["profiles"].T,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Cluster", "y": "Feature", "color": "Mean"},
            title="Mean feature values by cluster",
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with tab_data:
        st.dataframe(report["results"].head(500), width="stretch", hide_index=True)
        st.caption("Preview capped at 500 rows; the ZIP download contains every processed row.")
        st.download_button(
            "Download this dataset's clustered CSV",
            data=report["results"].to_csv(index=False).encode("utf-8"),
            file_name="airscope_clustered_data.csv",
            mime="text/csv",
        )
