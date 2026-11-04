import numpy as np
import pandas as pd

from insurance_pricing import config

SPLIT_ORDER = ["train", "val", "test"]


def assign_split(
    df: pd.DataFrame,
    group_col: str = config.ID_COL,
    fractions: dict = config.SPLIT_FRACTIONS,
    seed: int = config.SEED,
) -> pd.DataFrame:
    df = df.copy()
    groups = np.sort(df[group_col].unique())
    shuffled = np.random.default_rng(seed).permutation(groups)

    n = len(shuffled)
    n_train = int(fractions["train"] * n)
    n_val = int(fractions["val"] * n)

    labels = np.empty(n, dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "val"
    labels[n_train + n_val :] = "test"

    mapping = pd.Series(labels, index=shuffled)
    df[config.SPLIT_COL] = df[group_col].map(mapping)
    return df


def groups_are_disjoint(df: pd.DataFrame, group_col: str = config.ID_COL) -> bool:
    return bool((df.groupby(group_col)[config.SPLIT_COL].nunique() == 1).all())


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby(config.SPLIT_COL).agg(
        num_of_ids=(config.ID_COL, "nunique"),
        num_of_rows=(config.ID_COL, "size"),
        sum_claims=(config.N_CLAIMS_COL, "sum"),
        sum_exposure=(config.EXPOSURE_COL, "sum"),
    )
    summary["row_share"] = summary["num_of_rows"] / summary["num_of_rows"].sum()
    summary["frequency"] = summary["sum_claims"] / summary["sum_exposure"]
    return summary.loc[SPLIT_ORDER]


def get_split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df[config.SPLIT_COL] == name].reset_index(drop=True)