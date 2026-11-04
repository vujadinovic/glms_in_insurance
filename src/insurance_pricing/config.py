from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_FILE = RAW_DIR / "Motor vehicle insurance data.csv"
PROCESSED_FILE = PROCESSED_DIR / "policy_years.parquet"

CSV_SEPARATOR = ";"
DATE_FORMAT = "%d/%m/%Y"

SEED = 42
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_COL = "split"

OBSERVATION_END = pd.Timestamp("2018-12-31")
DAYS_PER_YEAR = 365.0
MIN_EXPOSURE = 1.0 / DAYS_PER_YEAR
MIN_DRIVING_AGE = 18

ID_COL = "ID"
N_CLAIMS_COL = "N_claims_year"
COST_CLAIMS_COL = "Cost_claims_year"
EXPOSURE_COL = "exposure"

DATE_COLS = [
    "Date_start_contract",
    "Date_last_renewal",
    "Date_next_renewal",
    "Date_birth",
    "Date_driving_licence",
    "Date_lapse",
]

PERIOD_START_COL = "Date_last_renewal"
PERIOD_END_COL = "Date_next_renewal"
LAPSE_DATE_COL = "Date_lapse"

DRIVER_AGE_COL = "driver_age"
LICENCE_YEARS_COL = "licence_years"
VEHICLE_AGE_COL = "vehicle_age"
RENEWAL_YEAR_COL = "renewal_year"

DROPPED_COLUMNS = {
    "N_claims_history": "contains the target: end-of-observation snapshot with N_claims_history >= N_claims_year",
    "R_Claims_history": "contains the target: derived from claims history",
    "Seniority": "end-of-observation snapshot, constant within ID",
    "Max_policies": "end-of-observation snapshot, constant within ID",
    "Max_products": "end-of-observation snapshot, constant within ID",
    "Lapse": "future information, used only for exposure",
    "Date_lapse": "future information, used only for exposure",
    "Premium": "output of the existing tariff, unknown before pricing",
    "Policies_in_force": "unclear whether it refers to the whole period, dropped conservatively",
}

KEEP_FOR_TARGET = [N_CLAIMS_COL, COST_CLAIMS_COL, EXPOSURE_COL]

GLM_BINNED_FEATURES = [
    DRIVER_AGE_COL,
    LICENCE_YEARS_COL,
    VEHICLE_AGE_COL,
    "Power",
    "Value_vehicle",
    "Weight",
]

GLM_CATEGORICAL_FEATURES = [
    "Type_risk",
    "Area",
    "Second_driver",
    "Type_fuel",
    "Distribution_channel",
    "Payment",
    RENEWAL_YEAR_COL,
]

GLM_EXCLUDED_FEATURES = {
    "Length": "strongly correlated with Weight and Cylinder_capacity, missing for about 10% of rows",
}

N_BINS = 6
UNKNOWN_LEVEL = "unknown"
INTERCEPT_COL = "intercept"

XGB_NUMERIC_FEATURES = [
    DRIVER_AGE_COL,
    LICENCE_YEARS_COL,
    VEHICLE_AGE_COL,
    "Power",
    "Cylinder_capacity",
    "Value_vehicle",
    "N_doors",
    "Length",
    "Weight",
    "Area",
    "Second_driver",
    "Distribution_channel",
    "Payment",
    RENEWAL_YEAR_COL,
]

XGB_CATEGORICAL_FEATURES = [
    "Type_risk",
    "Type_fuel",
]
