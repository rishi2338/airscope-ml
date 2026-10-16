"""Offline integration tests execute real page functions, not a substitute app."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import requests
from streamlit.testing.v1 import AppTest

from train_demo_model import train_demo_model

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "app.py").read_text(encoding="utf-8").replace(
    "APP_DIR = Path(__file__).resolve().parent", f"APP_DIR = Path({str(ROOT)!r})"
)


def run_page(page="page_dashboard", extra=""):
    code = SOURCE.replace("navigation.run()", f"{page}()\n{extra}")
    with patch("requests.get", side_effect=requests.ConnectionError("offline test")):
        at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def pipeline():
    code = SOURCE.replace("navigation.run()", 'st.session_state["test_data"] = get_data()\nst.session_state["test_ns"] = globals().copy()')
    with patch("requests.get", side_effect=requests.ConnectionError("offline test")):
        at = AppTest.from_string(code, default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    return at.session_state["test_ns"], at.session_state["test_data"]


@pytest.mark.parametrize("page", [
    "page_dashboard", "page_map", "page_forecast", "page_patterns",
    "page_analyze", "page_batch", "page_about",
])
def test_every_page_renders_offline(page):
    at = run_page(page)
    if page not in ("page_batch", "page_about"):
        assert any("synthetic" in w.value.lower() for w in at.warning)


def test_overview_city_and_compare_controls():
    at = run_page()
    at.selectbox(key="overview_city").select("Delhi").run()
    assert not at.exception
    at.selectbox(key="dash_metric").select("CO").run()
    assert not at.exception
    at.multiselect(key="dash_compare_cities").set_value([]).run()
    assert not at.exception
    assert any("at least one city" in i.value for i in at.info)


def test_sparkline_markup_uses_svg_capable_renderer():
    at = run_page()
    cards = next(element.value for element in at.markdown if 'class="pollutant-grid"' in element.value)
    assert cards.count('<svg class="sparkline"') == 6
    assert cards.count("<polyline") == 6


@pytest.mark.parametrize("view", ["2D relationship", "PCA projection", "Cluster mix", "Cluster map"])
def test_patterns_renders_only_selected_view(view):
    at = run_page("page_patterns")
    at.segmented_control(key="pat_view").set_value(view).run()
    assert not at.exception
    assert len(at.get("plotly_chart")) <= 2


def test_manual_analysis():
    at = run_page("page_analyze")
    next(button for button in at.button if button.label == "Analyze pollution profile").click().run()
    assert not at.exception
    assert "last_analysis" in at.session_state
    assert np.isfinite(list(at.session_state["last_analysis"]["values"].values())).all()


def test_forecast_and_backtest_are_finite_nonnegative(pipeline):
    ns, data = pipeline
    assert data["source"] == "synthetic"
    assert len(data["snap"]) == 26
    assert len(data["ml"]) == 26 * ns["MAX_HORIZON"]
    for frame in (data["snap"], data["history"], data["ml"], data["outlook"]):
        assert np.isfinite(frame[ns["DEFAULT_FEATURES"] + ["AQI"]]).all().all()
        assert (frame[ns["DEFAULT_FEATURES"]] >= 0).all().all()
    assert np.isfinite(data["forecaster"]["metrics"].select_dtypes("number")).all().all()
    assert list(data["forecaster"]["metrics"]["Horizon"]) == ["1 day", "3 days", "7 days"]
    assert data["ml"]["AQI_lo"].le(data["ml"]["AQI"]).all()
    assert data["ml"]["AQI_hi"].ge(data["ml"]["AQI"]).all()
    assert not any(c.endswith(("_lo", "_hi")) for c in data["outlook"])


def test_api_units_missing_nh3_and_nonfinite(pipeline):
    ns, _ = pipeline
    sample = {"time": ["2026-10-01T00:00", "2026-10-01T01:00"], "pm2_5": [10., 20.],
              "pm10": [20., 40.], "nitrogen_dioxide": [4., 8.], "carbon_monoxide": [1000., 2000.]}
    frame = ns["_om_to_features"](sample)
    assert frame["CO"].tolist() == [1., 2.]
    assert frame["NH3"].tolist() == [7.2, 8.4]
    assert frame["nh3_estimated"].all()
    assert frame["NOx"].tolist() == frame["NO2"].tolist()
    sample["pm2_5"] = [np.inf, -1]
    assert ns["_om_to_features"](sample)["PM2.5"].isna().all()
    sample["time"] = ["2026-10-01T00:00", "2026-10-01T00:00"]
    with pytest.raises(ValueError, match="duplicate"):
        ns["_om_to_features"](sample)


def test_blending_uses_cams_only_on_overlapping_days(pipeline):
    ns, data = pipeline
    ml = data["ml"].query("city == 'Delhi'").head(3)
    cams = ml[["city", "date"] + ns["DEFAULT_FEATURES"]].head(1).copy()
    cams[ns["DEFAULT_FEATURES"]] *= 2
    result = ns["blend_outlook"](ml, cams)
    assert result.iloc[0]["PM2.5"] == pytest.approx(ml.iloc[0]["PM2.5"] * 1.5)
    assert result.iloc[1]["PM2.5"] == ml.iloc[1]["PM2.5"]
    assert result["source"].tolist() == ["ML + CAMS ensemble", "ML extended", "ML extended"]


def test_no_partial_daily_averages(pipeline):
    ns, data = pipeline
    frame = data["hourly"].query("city == 'Delhi'").head(35)
    assert len(ns["to_daily"](frame)) == 1


def test_trusted_demo_artifact_roundtrip_and_corrupt_bundle(pipeline, tmp_path, monkeypatch):
    ns, _ = pipeline
    train_demo_model(tmp_path)
    loader = ns["load_artifacts"]
    for key, filename in [("MODEL_PATH", "air_pollution_model.pkl"), ("SCALER_PATH", "air_pollution_scaler.pkl"),
                          ("FEATURES_PATH", "feature_columns.pkl"), ("PROFILES_PATH", "cluster_profiles.pkl")]:
        monkeypatch.setitem(loader.__globals__, key, tmp_path / filename)
    model, scaler, columns, _ = loader()
    assert model._airscope_demo
    assert columns == ns["DEFAULT_FEATURES"]
    assert scaler.n_features_in_ == len(columns)
    (tmp_path / "feature_columns.pkl").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        loader()


def test_forecast_rejects_short_history(pipeline):
    ns, data = pipeline
    with pytest.raises(ValueError, match="23 complete"):
        ns["train_forecaster"](data["history"].groupby("city").head(10))
