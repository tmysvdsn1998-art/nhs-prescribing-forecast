import pytest

from nhs_forecast.utils import calculate_growth


def test_positive_growth():
    result = calculate_growth(1000, 1200)
    assert result == 20.0
def test_negative_growth():
    result = calculate_growth(1000, 800)
    assert result == -20.0
def test_error_growth():
    with pytest.raises(ValueError):
            calculate_growth(0,1200)
def test_zero_growth():
    result = calculate_growth(500, 500)
    assert result == 0.0