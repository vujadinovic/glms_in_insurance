import numpy as np
import pytest

from insurance_pricing import metrics


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    n = 5000
    exposure = rng.uniform(0.1, 1.0, n)
    rate = rng.gamma(2.0, 0.05, n)
    y = rng.poisson(rate * exposure).astype(float)
    return y, rate * exposure, exposure


def test_perfect_fit(sample):
    y, _, exposure = sample
    m = metrics.frequency_metrics(y, y, exposure, base_rate=y.sum() / exposure.sum())
    assert m["mean_deviance"] == pytest.approx(0.0)
    assert m["D2"] == pytest.approx(1.0)
    assert m["MSE"] == pytest.approx(0.0)
    assert m["MAE"] == pytest.approx(0.0)
    assert m["pred_obs_ratio"] == pytest.approx(1.0)


def test_constant_model(sample):
    y, _, exposure = sample
    rate = y.sum() / exposure.sum()
    m = metrics.frequency_metrics(y, rate * exposure, exposure, base_rate=rate)
    assert m["D2"] == pytest.approx(0.0)
    assert m["pred_obs_ratio"] == pytest.approx(1.0)
    assert m["gini"] == pytest.approx(0.0)


def test_true_mean_beats_constant(sample):
    y, mu, exposure = sample
    rate = y.sum() / exposure.sum()
    m = metrics.frequency_metrics(y, mu, exposure, base_rate=rate)
    assert m["D2"] > 0
    assert m["gini"] > 0


def test_gini_invariant_to_scaling(sample):
    y, mu, exposure = sample
    assert metrics.gini(y, 10 * mu, exposure) == pytest.approx(metrics.gini(y, mu, exposure))


def test_invalid_predictions():
    y = np.array([0.0, 1.0, 2.0])
    exposure = np.ones(3)
    assert metrics.mean_poisson_deviance(y, np.zeros(3)) == np.inf
    assert np.isnan(metrics.mean_poisson_deviance(y, np.array([0.5, -0.1, 1.0])))


def test_calibration_groups_have_equal_exposure(sample):
    y, mu, exposure = sample
    table = metrics.calibration_table(y, mu, exposure, n_groups=10)
    assert len(table) == 10
    share = table["exposure"] / exposure.sum()
    assert np.allclose(share, 0.1, atol=0.01)
    assert table["predicted"].sum() == pytest.approx(mu.sum())