"""
data/preprocessing.py
---------------------
Skill 1 — pandas
Skill 2 — scikit-learn (Pipeline, ColumnTransformer)

Responsibilities:
  1. Load raw Excel data
  2. Derive treatment flag and outcome label
  3. Engineer features from RFM time-windows and date columns
  4. Build a scikit-learn preprocessing pipeline
  5. Produce a clean train/test split, stratified on treatment
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from uplift_model.config import (
    CAMPAIGN_START_DATE,
    CUSTOMER_ID_COL,
    ENROLL_DATE_COL,
    ENGINEERED_FEATURES,
    LAST_TRANS_COL,
    MEMBERSHIP_COL,
    NET_REV_COL,
    OUTCOME_COL,
    RANDOM_STATE,
    RAW_CATEGORICAL_FEATURES,
    RAW_NUMERIC_FEATURES,
    STRATIFY_ON,
    TEST_SIZE,
    TREATMENT_COL,
    TREATMENT_COL_RAW,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_raw(path: Path) -> pd.DataFrame:
    """Load the campaign Excel table and parse date columns."""
    df = pd.read_excel(path, sheet_name="Table", parse_dates=[ENROLL_DATE_COL, LAST_TRANS_COL])
    print(f"[load] {df.shape[0]:,} rows × {df.shape[1]} columns loaded.")
    return df


# ── 2. Derive treatment flag and outcome ──────────────────────────────────────

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive two binary columns:
      - treated  : 1 if customer received the 15% discount, 0 otherwise
      - purchased: 1 if customer made at least one purchase during the campaign
    """
    df = df.copy()
    df[TREATMENT_COL] = (df[TREATMENT_COL_RAW] > 0).astype(int)
    df[OUTCOME_COL]   = (df["CMPGN_NUM_TRANS"] > 0).astype(int)

    treat_n   = df[TREATMENT_COL].sum()
    control_n = (df[TREATMENT_COL] == 0).sum()
    print(f"[labels] Treatment: {treat_n:,} ({treat_n/len(df):.1%})  "
          f"Control: {control_n:,} ({control_n/len(df):.1%})")
    print(f"[labels] Overall purchase rate: {df[OUTCOME_COL].mean():.4%}")
    return df


# ── 3. Feature engineering ────────────────────────────────────────────────────

def _safe_divide(numerator: pd.Series, denominator: pd.Series,
                 fill_value: float = 0.0) -> pd.Series:
    """Division that returns fill_value wherever denominator is zero."""
    return np.where(denominator != 0, numerator / denominator, fill_value)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build engineered features on top of the raw RFM columns.

    All features are derived solely from pre-campaign data; no campaign
    outcome variables (CMPGN_*) are used here.
    """
    df = df.copy()
    campaign_date = pd.Timestamp(CAMPAIGN_START_DATE)

    # ── Recency ───────────────────────────────────────────────────────────────
    # Customers with no prior transaction get a very large recency value (999 days)
    # to signal inactivity, rather than being dropped.
    df["recency_days"] = (
        (campaign_date - df[LAST_TRANS_COL])
        .dt.days
        .fillna(999)
        .clip(lower=0)
    )

    # ── Loyalty tenure ────────────────────────────────────────────────────────
    df["loyalty_tenure_days"] = (campaign_date - df[ENROLL_DATE_COL]).dt.days.clip(lower=0)

    # ── Revenue momentum: L3M relative to L12M ────────────────────────────────
    # > 1 means recent spending is accelerating, < 1 means decelerating.
    df["revenue_trend"] = _safe_divide(
        df["L3M_NET_REV"] * 4,   # annualise 3-month window
        df["L12M_NET_REV"],
        fill_value=0.0,
    )

    # ── Average basket size (gross and net) ───────────────────────────────────
    df["avg_basket_gross"] = _safe_divide(df["L12M_GROSS_REV"], df["L12M_NUM_TRANS"])
    df["avg_basket_net"]   = _safe_divide(df["L12M_NET_REV"],   df["L12M_NUM_TRANS"])

    # ── Revenue per visit: very recent window ─────────────────────────────────
    df["revenue_per_visit_3m"] = _safe_divide(df["L3M_NET_REV"], df["L3M_NUM_TRANS"])

    # ── Purchase velocity: L3M transactions as fraction of L12M ───────────────
    # Customers shopping faster recently than their annual average → > 0.25
    df["purchase_velocity_ratio"] = _safe_divide(
        df["L3M_NUM_TRANS"] * 4,   # annualised
        df["L12M_NUM_TRANS"],
        fill_value=0.0,
    )

    # ── Transaction growth: compare L6M run-rate to L12M run-rate ─────────────
    l12m_half = df["L12M_NUM_TRANS"] / 2.0
    df["trans_growth_rate"] = _safe_divide(
        df["L6M_NUM_TRANS"] - l12m_half,
        l12m_half.replace(0, np.nan),
        fill_value=0.0,
    )

    # ── Margin proxy: net / gross revenue ratio ───────────────────────────────
    df["gross_to_net_ratio_12m"] = _safe_divide(
        df["L12M_NET_REV"], df["L12M_GROSS_REV"], fill_value=0.0
    )

    # ── Binary: customer has never transacted before campaign ─────────────────
    df["never_transacted"] = df[LAST_TRANS_COL].isna().astype(int)

    print(f"[features] Engineered {len(ENGINEERED_FEATURES)} additional features.")
    return df


# ── 4. Preprocessing pipeline (scikit-learn) ──────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Returns a ColumnTransformer that:
      - Median-imputes then standard-scales numeric features
      - Ordinal-encodes the membership tier (already integer, but kept
        explicit so the pipeline is transparent and extensible)
    
    The transformer is fit only on training data to prevent leakage.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,      RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES),
            ("cat", categorical_pipeline,  RAW_CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return ordered feature names after transformation (for SHAP labelling)."""
    return RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + RAW_CATEGORICAL_FEATURES


# ── 5. Train / test split ─────────────────────────────────────────────────────

def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratify on the treatment column so that the 80/20 treatment/control
    ratio is preserved in both train and test sets.

    Returns (df_train, df_test) as raw DataFrames; the preprocessor is
    applied separately to avoid any leakage.
    """
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[STRATIFY_ON],
    )
    print(f"[split] Train: {len(df_train):,}  Test: {len(df_test):,}")
    print(f"[split] Train treatment rate: {df_train[TREATMENT_COL].mean():.3f}  "
          f"Test treatment rate: {df_test[TREATMENT_COL].mean():.3f}")
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# ── 6. Master pipeline function ───────────────────────────────────────────────

def prepare_data(path: Path) -> dict:
    """
    Full data preparation pipeline. Returns a dict with:
      - df_train, df_test   : raw DataFrames with all columns + labels
      - X_train, X_test     : numpy arrays (preprocessed features)
      - y_train, y_test     : binary outcome arrays
      - t_train, t_test     : binary treatment arrays
      - rev_train, rev_test : net revenue arrays (for economic evaluation)
      - preprocessor        : fitted ColumnTransformer
      - feature_names       : list of feature names matching X columns
    """
    df = load_raw(path)
    df = add_labels(df)
    df = engineer_features(df)

    df_train, df_test = split_data(df)

    preprocessor = build_preprocessor()
    feature_cols  = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES + RAW_CATEGORICAL_FEATURES

    X_train = preprocessor.fit_transform(df_train[feature_cols])
    X_test  = preprocessor.transform(df_test[feature_cols])

    return {
        "df_train":      df_train,
        "df_test":       df_test,
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       df_train[OUTCOME_COL].values,
        "y_test":        df_test[OUTCOME_COL].values,
        "t_train":       df_train[TREATMENT_COL].values,
        "t_test":        df_test[TREATMENT_COL].values,
        "rev_train":     df_train[NET_REV_COL].values,
        "rev_test":      df_test[NET_REV_COL].values,
        "preprocessor":  preprocessor,
        "feature_names": get_feature_names(preprocessor),
    }
