import numpy as np
import pandas as pd


def _as_arrays(*arrays):
    return [np.asarray(a, dtype=float) for a in arrays]


def poisson_unit_deviance(y, mu) -> np.ndarray:
    y, mu = _as_arrays(y, mu)
    out = np.full_like(mu, np.nan)
    valid = mu >= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        y_log = np.where(y > 0, y * np.log(y / mu), 0.0)
    out[valid] = 2.0 * (y_log[valid] - y[valid] + mu[valid])
    return out


def mean_poisson_deviance(y, mu) -> float:
    d = poisson_unit_deviance(y, mu)
    if np.isnan(d).any():
        return np.nan
    return float(d.mean())


def deviance_explained(y, mu, exposure, base_rate: float) -> float:
    exposure = np.asarray(exposure, dtype=float)
    d_model = mean_poisson_deviance(y, mu)
    d_const = mean_poisson_deviance(y, base_rate * exposure)
    return 1.0 - d_model / d_const


def lorenz_curve(y, mu, exposure) -> tuple[np.ndarray, np.ndarray]:
    y, mu, exposure = _as_arrays(y, mu, exposure)
    rate = np.round(mu / exposure, 12)
    levels, inverse = np.unique(rate, return_inverse=True)
    exp_by_level = np.bincount(inverse, weights=exposure, minlength=len(levels))
    claims_by_level = np.bincount(inverse, weights=y, minlength=len(levels))
    x = np.concatenate([[0.0], np.cumsum(exp_by_level) / exposure.sum()])
    lorenz = np.concatenate([[0.0], np.cumsum(claims_by_level) / y.sum()])
    return x, lorenz


def gini(y, mu, exposure) -> float:
    if np.isnan(np.asarray(mu, dtype=float)).any():
        return np.nan
    x, lorenz = lorenz_curve(y, mu, exposure)
    return float(1.0 - 2.0 * np.trapezoid(lorenz, x))


def frequency_metrics(y, mu, exposure, base_rate: float) -> dict:
    y, mu, exposure = _as_arrays(y, mu, exposure)
    return {
        "mean_deviance": mean_poisson_deviance(y, mu),
        "D2": deviance_explained(y, mu, exposure, base_rate),
        "MSE": float(np.mean((y - mu) ** 2)),
        "MAE": float(np.mean(np.abs(y - mu))),
        "pred_obs_ratio": float(mu.sum() / y.sum()),
        "gini": gini(y, mu, exposure),
    }


def metrics_table(y, exposure, predictions: dict, base_rate: float) -> pd.DataFrame:
    rows = {
        name: frequency_metrics(y, mu, exposure, base_rate)
        for name, mu in predictions.items()
    }
    return pd.DataFrame(rows).T


def calibration_table(y, mu, exposure, n_groups: int = 10) -> pd.DataFrame:
    y, mu, exposure = _as_arrays(y, mu, exposure)
    order = np.argsort(np.round(mu / exposure, 12), kind="stable")
    y, mu, exposure = y[order], mu[order], exposure[order]

    midpoint_share = (np.cumsum(exposure) - exposure / 2.0) / exposure.sum()
    group = np.minimum((midpoint_share * n_groups).astype(int), n_groups - 1)

    table = (
        pd.DataFrame({"group": group, "claims": y, "predicted": mu, "exposure": exposure})
        .groupby("group")
        .sum()
    )
    table["observed_freq"] = table["claims"] / table["exposure"]
    table["predicted_freq"] = table["predicted"] / table["exposure"]
    return table