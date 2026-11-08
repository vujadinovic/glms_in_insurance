import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge

from insurance_pricing import config


def base_rate(df: pd.DataFrame) -> float:
    return float(df[config.N_CLAIMS_COL].sum() / df[config.EXPOSURE_COL].sum())


class AlwaysZero:
    def fit(self, X: pd.DataFrame, df: pd.DataFrame) -> "AlwaysZero":
        return self

    def predict(self, X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


class ConstantFrequency:
    def fit(self, X: pd.DataFrame, df: pd.DataFrame) -> "ConstantFrequency":
        self.rate_ = base_rate(df)
        return self

    def predict(self, X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
        return self.rate_ * df[config.EXPOSURE_COL].to_numpy()


class RidgeFrequency:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, df: pd.DataFrame) -> "RidgeFrequency":
        exposure = df[config.EXPOSURE_COL].to_numpy()
        frequency = df[config.N_CLAIMS_COL].to_numpy() / exposure
        self.model_ = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model_.fit(X.drop(columns=config.INTERCEPT_COL), frequency, sample_weight=exposure)
        return self

    def predict(self, X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
        rate = self.model_.predict(X.drop(columns=config.INTERCEPT_COL))
        return rate * df[config.EXPOSURE_COL].to_numpy()


class PoissonGLM:
    def fit(self, X: pd.DataFrame, df: pd.DataFrame) -> "PoissonGLM":
        self.result_ = sm.GLM(
            df[config.N_CLAIMS_COL].to_numpy(),
            X,
            family=sm.families.Poisson(),
            offset=np.log(df[config.EXPOSURE_COL].to_numpy()),
        ).fit()
        return self

    def predict(self, X: pd.DataFrame, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.result_.predict(X, offset=np.log(df[config.EXPOSURE_COL].to_numpy()))
        )