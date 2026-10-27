"""
config.py
---------
Central configuration: paths, column names, model hyperparameters,
and campaign economics. Edit this file to adapt the pipeline to
new data or discount levels.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH = Path("data/campaign_data.xlsx")
OUTPUT_DIR = Path("outputs")
MODEL_DIR  = Path("outputs/models")

# ── Column name constants ──────────────────────────────────────────────────────
CUSTOMER_ID_COL   = "CUSTOMER_ID"
TREATMENT_COL_RAW = "TARGETED_DISCOUNT"   # raw: 0.15 or 0
TREATMENT_COL     = "treated"             # derived binary flag
OUTCOME_COL       = "purchased"           # derived: 1 if CMPGN_NUM_TRANS > 0
NET_REV_COL       = "CMPGN_NET_REV"
GROSS_REV_COL     = "CMPGN_GROSS_REV"
ENROLL_DATE_COL   = "ENROLL_DT"
LAST_TRANS_COL    = "LST_TRANS_DT"
MEMBERSHIP_COL    = "MBRSHP_TIER"

# Reference date for recency calculations (first day of campaign)
CAMPAIGN_START_DATE = "2024-01-01"        # update to actual campaign date

# ── Raw feature columns (from dataset, excluding ID/dates/targets) ─────────────
RFM_WINDOWS = ["L36M", "L24M", "L12M", "L6M", "L3M", "L1M"]

RAW_NUMERIC_FEATURES = [
    f"{w}_{suffix}"
    for w in RFM_WINDOWS
    for suffix in ["NUM_TRANS", "TRANS_SIZE", "GROSS_REV", "NET_REV"]
]  # 24 columns

RAW_CATEGORICAL_FEATURES = [MEMBERSHIP_COL]  # ordinal 1/2/3/...

# ── Engineered feature names ───────────────────────────────────────────────────
ENGINEERED_FEATURES = [
    "recency_days",           # days since last transaction (NaN → very large)
    "loyalty_tenure_days",    # days since enrolment
    "revenue_trend",          # L3M_NET_REV / L12M_NET_REV (momentum)
    "avg_basket_gross",       # L12M_GROSS_REV / L12M_NUM_TRANS
    "avg_basket_net",         # L12M_NET_REV  / L12M_NUM_TRANS
    "revenue_per_visit_3m",   # L3M_NET_REV   / L3M_NUM_TRANS
    "purchase_velocity_ratio",# L3M_NUM_TRANS / L12M_NUM_TRANS (recent vs annual)
    "trans_growth_rate",      # (L6M_NUM_TRANS - L12M_NUM_TRANS/2) normalised
    "gross_to_net_ratio_12m", # margin proxy: L12M_NET_REV / L12M_GROSS_REV
    "never_transacted",       # binary: LST_TRANS_DT is null
]

ALL_FEATURES = RAW_NUMERIC_FEATURES + RAW_CATEGORICAL_FEATURES + ENGINEERED_FEATURES

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE       = 0.20
RANDOM_STATE    = 42
STRATIFY_ON     = TREATMENT_COL   # preserve 80/20 treatment ratio in splits

# ── Model hyperparameters ─────────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 370,   # ~(1-0.0027)/0.0027; compensates for 0.27% purchase rate
    "eval_metric":      "auc",
    "use_label_encoder": False,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

LGBM_PARAMS = {
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "is_unbalance":     True,   # equivalent to scale_pos_weight for LightGBM
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1,
}

# ── Campaign economics ────────────────────────────────────────────────────────
DISCOUNT_RATE          = 0.15    # 15% off
# Break-even uplift threshold:
# The model must predict an incremental purchase probability large enough
# that the expected incremental revenue offsets the discount cost.
# τ_breakeven = (price × discount%) / avg_product_price_net
# Computed dynamically in evaluation/economics.py from the data.
# Set a fallback default here (can be overridden):
BREAKEVEN_UPLIFT_DEFAULT = 0.002  # 0.2 pp lift needed to break even

# ── Evaluation ────────────────────────────────────────────────────────────────
QINI_PERCENTILES = list(range(0, 101, 10))   # decile buckets for Qini curve
TOP_DECILE_FRACTION = 0.10
