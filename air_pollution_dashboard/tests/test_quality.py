import io

import numpy as np
import pandas as pd
import pytest

from quality import PM10_BANDS, PM25_BANDS, compute_aqi, csv_export, validate_measurements
from ui import sparkline


@pytest.mark.parametrize("lower,upper,aqi_lower,aqi_upper", PM25_BANDS)
def test_pm25_band_endpoints(lower, upper, aqi_lower, aqi_upper):
    assert compute_aqi(lower) == aqi_lower
    assert compute_aqi(upper) == aqi_upper


@pytest.mark.parametrize("lower,upper,aqi_lower,aqi_upper", PM10_BANDS)
def test_pm10_band_endpoints(lower, upper, aqi_lower, aqi_upper):
    assert compute_aqi(np.nan, lower) == aqi_lower
    assert compute_aqi(np.nan, upper) == aqi_upper


def test_truncation_maximum_invalid_and_cap():
    assert compute_aqi(9.09) == 50
    assert compute_aqi(np.nan, 54.99) == 50
    assert compute_aqi(0, 155) == 101
    assert compute_aqi(1000, 1000) == 500
    assert np.isnan(compute_aqi(np.nan, None))
    assert np.isnan(compute_aqi(-1, np.inf))


def test_aqi_is_monotonic():
    values = [compute_aqi(v) for v in np.arange(0, 400, .1)]
    assert np.isfinite(values).all()
    assert (np.diff(values) >= 0).all()


@pytest.mark.parametrize("value", [None, "", "bad", -1, np.inf, -np.inf, np.nan])
def test_invalid_measurements_report_row(value):
    with pytest.raises(ValueError, match=r"row\(s\): 2"):
        validate_measurements(pd.DataFrame({"PM2.5": [10, value]}), ["PM2.5"])


def test_numeric_strings_empty_and_missing():
    assert validate_measurements(pd.DataFrame({"CO": ["1.25"]}), ["CO"]).iloc[0, 0] == 1.25
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_measurements(pd.DataFrame({"AQI": [50]}), ["PM2.5"])
    with pytest.raises(ValueError, match="no data rows"):
        validate_measurements(pd.DataFrame(columns=["CO"]), ["CO"])


def test_csv_formulas_are_neutralized_without_altering_numeric_values():
    frame = pd.DataFrame({"place": ["=CMD()", " +SUM(1)", "@unsafe", "Delhi"], "CO": [1.2] * 4})
    output = pd.read_csv(io.BytesIO(csv_export(frame)))
    assert output["place"].tolist() == ["'=CMD()", "' +SUM(1)", "'@unsafe", "Delhi"]
    assert output["CO"].tolist() == [1.2] * 4


def test_sparkline_empty_constant_and_missing():
    assert "<polyline" not in sparkline([])
    assert "nan" not in sparkline([2, np.nan, 2])
    assert "inf" not in sparkline([2, np.inf, 2])
