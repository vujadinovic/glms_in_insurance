import numpy as np
import pandas as pd

from insurance_pricing import config


class GLMPreprocessor:
    def __init__(
        self,
        binned_features: list[str] = config.GLM_BINNED_FEATURES,
        categorical_features: list[str] = config.GLM_CATEGORICAL_FEATURES,
        n_bins: int = config.N_BINS,
    ):
        self.binned_features = list(binned_features)
        self.categorical_features = list(categorical_features)
        self.n_bins = n_bins
        self.bin_edges_: dict[str, np.ndarray] = {}
        self.levels_: dict[str, list[str]] = {}
        self.reference_levels_: dict[str, str] = {}
        self.feature_names_: list[str] = []

    @property
    def features(self) -> list[str]:
        return self.binned_features + self.categorical_features

    def fit(self, df: pd.DataFrame) -> "GLMPreprocessor":
        self.bin_edges_ = {col: self._quantile_edges(df[col]) for col in self.binned_features}
        levels_df = self._to_levels(df)
        weights = df[config.EXPOSURE_COL]

        self.levels_ = {}
        self.reference_levels_ = {}
        for col in self.features:
            exposure_by_level = weights.groupby(levels_df[col]).sum()
            if col in self.binned_features:
                order = self._bin_labels(self.bin_edges_[col]) + [config.UNKNOWN_LEVEL]
                levels = [lvl for lvl in order if lvl in exposure_by_level.index]
            else:
                levels = sorted(exposure_by_level.index)
            self.levels_[col] = levels
            self.reference_levels_[col] = exposure_by_level.idxmax()

        self.feature_names_ = [config.INTERCEPT_COL] + [
            self._column_name(col, lvl)
            for col in self.features
            for lvl in self.levels_[col]
            if lvl != self.reference_levels_[col]
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.feature_names_:
            raise RuntimeError("GLMPreprocessor is not fitted")

        levels_df = self._to_levels(df)
        columns = {config.INTERCEPT_COL: np.ones(len(df))}
        for col in self.features:
            values = self._map_unseen(col, levels_df[col])
            for lvl in self.levels_[col]:
                if lvl != self.reference_levels_[col]:
                    columns[self._column_name(col, lvl)] = (values == lvl).to_numpy(dtype=float)

        return pd.DataFrame(columns, index=df.index)[self.feature_names_]

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _quantile_edges(self, values: pd.Series) -> np.ndarray:
        probs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.unique(np.quantile(values.dropna().to_numpy(dtype=float), probs))
        if len(edges) < 2:
            return np.array([-np.inf, np.inf])
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    @staticmethod
    def _bin_labels(edges: np.ndarray) -> list[str]:
        return [f"({a:.4g}, {b:.4g}]" for a, b in zip(edges[:-1], edges[1:])]

    @staticmethod
    def _column_name(col: str, level: str) -> str:
        return f"{col}={level}"

    @staticmethod
    def _as_level_strings(values: pd.Series) -> pd.Series:
        values = values.astype(object)
        return values.where(values.notna(), config.UNKNOWN_LEVEL).astype(str)

    def _to_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for col in self.binned_features:
            edges = self.bin_edges_[col]
            binned = pd.cut(df[col], bins=edges, labels=self._bin_labels(edges))
            out[col] = self._as_level_strings(binned)
        for col in self.categorical_features:
            out[col] = self._as_level_strings(df[col])
        return pd.DataFrame(out, index=df.index)

    def _map_unseen(self, col: str, values: pd.Series) -> pd.Series:
        levels = self.levels_[col]
        fallback = (
            config.UNKNOWN_LEVEL if config.UNKNOWN_LEVEL in levels else self.reference_levels_[col]
        )
        return values.where(values.isin(levels), fallback)