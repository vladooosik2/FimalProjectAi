import pytest
from predictor import WeatherPredictor


@pytest.fixture
def predictor():
    return WeatherPredictor("artifacts")


def valid_sample(predictor):
    sample = {k: v["median"] for k, v in predictor.bounds["numeric"].items()}
    for k, opts in predictor.bounds["categorical"].items():
        sample[k] = opts[0]
    return sample


def test_prediction_returns_valid_probabilities(predictor):
    res = predictor.validate_and_predict(valid_sample(predictor))
    assert res["prediction"] in predictor.bounds["classes"]
    assert abs(sum(res["probabilities"].values()) - 1.0) < 1e-2
    assert len(res["warnings"]) == 0


def test_out_of_bounds_triggers_warning(predictor):
    sample = valid_sample(predictor)
    sample["Temperature"] = 999.0
    res = predictor.validate_and_predict(sample)
    assert len(res["warnings"]) >= 1


def test_empty_field_raises_value_error(predictor):
    sample = valid_sample(predictor)
    sample["Humidity"] = None
    with pytest.raises(ValueError):
        predictor.validate_and_predict(sample)
