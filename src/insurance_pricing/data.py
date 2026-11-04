from pathlib import Path

import numpy as np
import pandas as pd

from insurance_pricing import config


def load_raw(path: Path = config.RAW_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, sep=config.CSV_SEPARATOR)
    for col in config.DATE_COLS:
        df[col] = pd.to_datetime(df[col], format=config.DATE_FORMAT, errors="coerce")
    return df


def _years_between(start: pd.Series, end: pd.Series) -> pd.Series:
    return (end - start).dt.days / config.DAYS_PER_YEAR


def compute_exposure(df: pd.DataFrame) -> pd.Series:
    start = df[config.PERIOD_START_COL]
    period_end = df[config.PERIOD_END_COL]
    lapse = df[config.LAPSE_DATE_COL]

    lapse_in_period = lapse.notna() & (lapse > start) & (lapse < period_end)
    end = period_end.where(~lapse_in_period, lapse)
    end = end.where(end < config.OBSERVATION_END, config.OBSERVATION_END)

    exposure = _years_between(start, end)
    return exposure.clip(lower=config.MIN_EXPOSURE, upper=1.0)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    start = df[config.PERIOD_START_COL]
    df[config.EXPOSURE_COL] = compute_exposure(df)
    df[config.DRIVER_AGE_COL] = _years_between(df["Date_birth"], start)
    df[config.LICENCE_YEARS_COL] = _years_between(df["Date_driving_licence"], start)
    df[config.VEHICLE_AGE_COL] = start.dt.year - df["Year_matriculation"]
    df[config.RENEWAL_YEAR_COL] = start.dt.year
    return df


def anomaly_report(df: pd.DataFrame) -> pd.Series:
    age = df[config.DRIVER_AGE_COL]
    licence = df[config.LICENCE_YEARS_COL]
    vehicle_age = df[config.VEHICLE_AGE_COL]
    return pd.Series(
        {
            "driver_age_below_18": int((age < config.MIN_DRIVING_AGE).sum()),
            "licence_years_negative": int((licence < 0).sum()),
            "licence_before_18": int((licence > age - config.MIN_DRIVING_AGE).sum()),
            "vehicle_age_negative": int((vehicle_age < 0).sum()),
            "missing_dates": int(
                df[[config.PERIOD_START_COL, "Date_birth", "Date_driving_licence"]]
                .isna()
                .any(axis=1)
                .sum()
            ),
        }
    )


def fix_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    max_licence = (df[config.DRIVER_AGE_COL] - config.MIN_DRIVING_AGE).clip(lower=0)
    df[config.LICENCE_YEARS_COL] = np.minimum(
        df[config.LICENCE_YEARS_COL].clip(lower=0), max_licence
    )
    df[config.VEHICLE_AGE_COL] = df[config.VEHICLE_AGE_COL].clip(lower=0)
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    raw_date_cols = [c for c in config.DATE_COLS if c != config.PERIOD_START_COL]
    to_drop = set(config.DROPPED_COLUMNS) | set(raw_date_cols) | {"Year_matriculation"}
    return df.drop(columns=[c for c in to_drop if c in df.columns])


def build_dataset(path: Path = config.RAW_FILE) -> pd.DataFrame:
    df = load_raw(path)
    df = add_derived_features(df)
    df = fix_anomalies(df)
    df = drop_unused_columns(df)
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, path: Path = config.PROCESSED_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_processed(path: Path = config.PROCESSED_FILE) -> pd.DataFrame:
    return pd.read_parquet(path)