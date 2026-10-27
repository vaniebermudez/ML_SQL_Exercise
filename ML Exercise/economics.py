"""
evaluation/economics.py
-----------------------
Campaign economics: compute the break-even uplift threshold and
translate raw uplift scores into a net-revenue uplift estimate
per customer.

Break-Even Logic
----------------
For a given customer, offering the 15% discount is net-positive when:

    E[Incremental Net Revenue] > 0

    τᵢ × P_net > DISCOUNT_RATE × P_gross

Where:
    τᵢ       = predicted uplift in purchase probability
    P_net    = average net revenue per purchase (from control group)
    P_gross  = average gross revenue per purchase (revenue before cost)
    DISCOUNT_RATE = 0.15

Rearranging:
    τᵢ_breakeven = DISCOUNT_RATE × avg_gross_rev_per_purchase
                   ──────────────────────────────────────────
                           avg_net_rev_per_purchase

In practice, we compute these averages from the control group in the
training set (to avoid using campaign results for targeting decisions).
"""

import numpy as np
import pandas as pd

from uplift_model.config import DISCOUNT_RATE, BREAKEVEN_UPLIFT_DEFAULT


def compute_breakeven_threshold(
    df_control: pd.DataFrame,
    gross_col: str = "CMPGN_GROSS_REV",
    net_col: str   = "CMPGN_NET_REV",
    trans_col: str = "CMPGN_NUM_TRANS",
) -> float:
    """
    Estimate the minimum uplift in purchase probability needed for
    the 15% discount to generate net-positive incremental revenue.

    Only control-group buyers are used to estimate baseline revenue
    per transaction (avoids contamination from the discount's effect
    on transaction values).

    Parameters
    ----------
    df_control : rows from the control arm (TARGETED_DISCOUNT == 0)

    Returns
    -------
    threshold : float — uplift scores above this are net-positive to target
    """
    buyers = df_control[df_control[trans_col] > 0]

    if len(buyers) < 10:
        print(f"[economics] Warning: too few control buyers ({len(buyers)}); "
              f"using default threshold {BREAKEVEN_UPLIFT_DEFAULT}.")
        return BREAKEVEN_UPLIFT_DEFAULT

    avg_gross = buyers[gross_col].mean()
    avg_net   = buyers[net_col].mean()

    if avg_net == 0:
        print("[economics] Warning: avg net revenue is 0; using default threshold.")
        return BREAKEVEN_UPLIFT_DEFAULT

    # Cost of the discount per persuaded buyer = DISCOUNT_RATE × avg gross revenue
    # Benefit per persuaded buyer               = avg net revenue per purchase
    threshold = (DISCOUNT_RATE * avg_gross) / avg_net

    print(f"[economics] Control buyers: {len(buyers):,}")
    print(f"[economics] Avg gross rev per purchase: {avg_gross:.2f}")
    print(f"[economics] Avg net rev per purchase:   {avg_net:.2f}")
    print(f"[economics] Break-even uplift threshold: {threshold:.5f}  "
          f"({threshold:.2%} increase in P(buy) needed)")
    return threshold


def add_targeting_decision(
    df: pd.DataFrame,
    uplift_scores: np.ndarray,
    threshold: float,
    score_col: str     = "uplift_score",
    decision_col: str  = "send_discount",
    tier_col: str      = "targeting_tier",
) -> pd.DataFrame:
    """
    Attach uplift scores and targeting decisions to the scored DataFrame.

    Three tiers:
      HIGH  : uplift > threshold × 1.5  → send discount confidently
      MID   : threshold ≤ uplift ≤ threshold × 1.5  → borderline; monitor
      NO    : uplift < threshold  → do not send

    Parameters
    ----------
    df             : DataFrame to annotate (typically df_test)
    uplift_scores  : predicted uplift scores from model.predict()
    threshold      : break-even threshold from compute_breakeven_threshold()
    """
    df = df.copy()
    df[score_col]    = uplift_scores
    df[decision_col] = (uplift_scores >= threshold).astype(int)

    high_band  = threshold * 1.5
    conditions = [
        uplift_scores >= high_band,
        (uplift_scores >= threshold) & (uplift_scores < high_band),
    ]
    choices = ["HIGH", "MID"]
    df[tier_col] = np.select(conditions, choices, default="NO")

    # Summary
    tier_counts = df[tier_col].value_counts()
    total = len(df)
    for tier in ["HIGH", "MID", "NO"]:
        n = tier_counts.get(tier, 0)
        print(f"[targeting] {tier:4s} tier: {n:>8,} customers ({n/total:.1%})")

    return df


def summarise_campaign_economics(
    df_scored: pd.DataFrame,
    treatment_col: str  = "treated",
    outcome_col: str    = "purchased",
    net_rev_col: str    = "CMPGN_NET_REV",
    score_col: str      = "uplift_score",
    tier_col: str       = "targeting_tier",
) -> pd.DataFrame:
    """
    Compare key metrics between:
      - Full blast (current approach: send to all 80% treated)
      - Targeted   (send only to HIGH uplift tier)
      - Control    (baseline)

    Returns a summary DataFrame for reporting.
    """
    rows = []

    def _stats(mask, label):
        sub = df_scored[mask]
        n   = len(sub)
        if n == 0:
            return
        purchase_rate = sub[outcome_col].mean()
        total_net_rev = sub[net_rev_col].sum()
        avg_net_rev   = sub[net_rev_col].mean()
        rows.append({
            "Segment":          label,
            "N Customers":      n,
            "Purchase Rate":    f"{purchase_rate:.4%}",
            "Total Net Rev":    f"${total_net_rev:,.2f}",
            "Avg Net Rev":      f"${avg_net_rev:.4f}",
        })

    _stats(df_scored[treatment_col] == 0,                        "Control (no discount)")
    _stats(df_scored[treatment_col] == 1,                        "Full blast (all treated)")
    _stats(df_scored[tier_col] == "HIGH",                        "Targeted — HIGH tier")
    _stats(df_scored[tier_col] == "MID",                         "Targeted — MID tier")
    _stats((df_scored[tier_col] == "NO") & (df_scored[treatment_col] == 1),
                                                                  "Would exclude (NO tier)")

    summary = pd.DataFrame(rows)
    print("\n[economics] Campaign comparison:")
    print(summary.to_string(index=False))
    return summary
