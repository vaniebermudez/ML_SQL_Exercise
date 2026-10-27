"""
evaluation/metrics.py
---------------------
Skill 6 — matplotlib / seaborn  (Qini curve, gain chart, score distribution)
Skill 7 — scipy / statsmodels   (bootstrap CI on Qini, t-tests on net revenue)

Uplift model evaluation differs fundamentally from standard classification:
we cannot observe the counterfactual for any individual, so we evaluate
using population-level rank-ordering metrics.

Key metrics implemented here:
  - Qini coefficient  (area between model curve and random baseline)
  - Uplift by decile  (average uplift score per population decile)
  - Net revenue lift  (actual CMPGN_NET_REV difference, treated vs control,
                       within top-N% of model-ranked customers)
  - Bootstrap CI on Qini  (resample to get confidence interval)
  - t-test on top-decile net revenue uplift  (for stakeholder reporting)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

from uplift_model.config import OUTPUT_DIR, TOP_DECILE_FRACTION

warnings.filterwarnings("ignore")

# ── Plot style ────────────────────────────────────────────────────────────────
PALETTE   = {"treated": "#2EC4B6", "control": "#F4A261", "model": "#1E2761", "random": "#AAAAAA"}
FONT_SANS = "DejaVu Sans"
plt.rcParams.update({
    "font.family":        FONT_SANS,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         120,
})


# ── Qini curve helpers ────────────────────────────────────────────────────────

def _qini_curve_data(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Qini curve: cumulative incremental buyers as a fraction
    of population targeted, sorted by descending uplift score.

    Returns (fraction_targeted, cumulative_incremental_gains) — both
    normalised to [0, 1] on the x-axis.
    """
    df = pd.DataFrame({"y": y, "t": t, "score": uplift_scores})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    n_treated = t.sum()
    n_control = (t == 0).sum()

    cum_treated_buyers = np.cumsum(df["y"] * df["t"])
    cum_control_buyers = np.cumsum(df["y"] * (1 - df["t"]))

    # Normalise control buyers by treatment/control ratio to put on same scale
    ratio = n_treated / n_control if n_control > 0 else 1.0
    cum_gains = cum_treated_buyers - ratio * cum_control_buyers

    x = np.arange(1, len(df) + 1) / len(df)
    return x, cum_gains


def qini_coefficient(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
) -> float:
    """
    Qini coefficient = area between model Qini curve and random baseline,
    normalised by the area of the perfect model.

    A positive Qini indicates the model ranks persuadables better than random.
    Value of 1.0 = perfect ranking; 0.0 = random.
    """
    x, gains = _qini_curve_data(y, t, uplift_scores)

    # Random baseline: linear from 0 to final gains value
    random_baseline = np.linspace(0, gains[-1], len(x))

    # Area under model curve minus area under random baseline
    model_area  = np.trapz(gains, x)
    random_area = np.trapz(random_baseline, x)

    # Perfect model area (upper bound): all incremental buyers ranked first
    n_treated_buyers = (y[t == 1]).sum()
    n_control_buyers = (y[t == 0]).sum()
    ratio = t.sum() / (t == 0).sum() if (t == 0).sum() > 0 else 1.0
    perfect_max = n_treated_buyers - ratio * n_control_buyers
    perfect_area = perfect_max / 2  # triangle approximation

    if perfect_area <= 0:
        return 0.0

    qini = (model_area - random_area) / perfect_area
    return float(qini)


def bootstrap_qini_ci(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
    n_bootstrap: int = 500,
    ci_level: float  = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """
    Skill 7: Bootstrap confidence interval for the Qini coefficient.

    Resample (with replacement) n_bootstrap times, compute Qini each
    time, return (point_estimate, lower_ci, upper_ci).
    """
    rng    = np.random.default_rng(random_state)
    n      = len(y)
    qinis  = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        q   = qini_coefficient(y[idx], t[idx], uplift_scores[idx])
        qinis.append(q)

    qinis = np.array(qinis)
    alpha = (1 - ci_level) / 2
    lower = np.quantile(qinis, alpha)
    upper = np.quantile(qinis, 1 - alpha)
    point = qini_coefficient(y, t, uplift_scores)

    print(f"[qini] Point estimate: {point:.4f}  "
          f"{ci_level:.0%} CI: [{lower:.4f}, {upper:.4f}]")
    return point, lower, upper


# ── Net revenue lift validation ───────────────────────────────────────────────

def top_decile_revenue_test(
    df_scored: pd.DataFrame,
    treatment_col: str = "treated",
    net_rev_col: str   = "CMPGN_NET_REV",
    score_col: str     = "uplift_score",
    top_frac: float    = TOP_DECILE_FRACTION,
) -> dict:
    """
    Skill 7: Two-sided t-test comparing net revenue between treated and
    control customers within the top model-ranked fraction.

    Hypothesis: customers with high predicted uplift who received the
    discount generated significantly more net revenue than controls with
    the same high uplift score.
    """
    n_top      = int(top_frac * len(df_scored))
    top_scored = df_scored.nlargest(n_top, score_col)

    treated_rev = top_scored[top_scored[treatment_col] == 1][net_rev_col].values
    control_rev = top_scored[top_scored[treatment_col] == 0][net_rev_col].values

    t_stat, p_val = stats.ttest_ind(treated_rev, control_rev, equal_var=False)

    result = {
        "n_top_treated":    len(treated_rev),
        "n_top_control":    len(control_rev),
        "mean_treated_rev": treated_rev.mean(),
        "mean_control_rev": control_rev.mean(),
        "mean_difference":  treated_rev.mean() - control_rev.mean(),
        "t_statistic":      t_stat,
        "p_value":          p_val,
        "significant_5pct": p_val < 0.05,
    }

    print(f"\n[t-test] Top {top_frac:.0%} of model-ranked customers:")
    print(f"  Treated  n={result['n_top_treated']:,}  "
          f"mean net rev = ${result['mean_treated_rev']:.4f}")
    print(f"  Control  n={result['n_top_control']:,}  "
          f"mean net rev = ${result['mean_control_rev']:.4f}")
    print(f"  Difference = ${result['mean_difference']:.4f}")
    print(f"  t = {t_stat:.3f},  p = {p_val:.4f}  "
          f"({'✓ significant' if result['significant_5pct'] else '✗ not significant'} at 5%)")
    return result


# ── Visualisations ────────────────────────────────────────────────────────────

def plot_qini_curve(
    y: np.ndarray,
    t: np.ndarray,
    uplift_scores: np.ndarray,
    model_name: str = "Model",
    save_path = None,
) -> plt.Figure:
    """
    Skill 6: Plot the Qini curve with random baseline and Qini coefficient.
    """
    x, gains = _qini_curve_data(y, t, uplift_scores)
    qini      = qini_coefficient(y, t, uplift_scores)
    random_bl = np.linspace(0, gains[-1], len(x))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x * 100, gains, color=PALETTE["model"],  lw=2.5, label=f"{model_name} (Qini={qini:.4f})")
    ax.plot(x * 100, random_bl, color=PALETTE["random"], lw=1.5, ls="--", label="Random baseline")
    ax.fill_between(x * 100, gains, random_bl, alpha=0.12, color=PALETTE["model"])

    ax.set_xlabel("% Population Targeted (ranked by uplift score)", fontsize=12)
    ax.set_ylabel("Cumulative Incremental Buyers", fontsize=12)
    ax.set_title(f"Qini Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Qini curve saved → {save_path}")
    return fig


def plot_uplift_by_decile(
    df_scored: pd.DataFrame,
    treatment_col: str = "treated",
    outcome_col: str   = "purchased",
    score_col: str     = "uplift_score",
    save_path = None,
) -> plt.Figure:
    """
    Skill 6: Bar chart of actual purchase rate lift (treated vs control)
    within each score decile.

    A good model concentrates positive lift in the top deciles.
    """
    df = df_scored.copy()
    df["decile"] = pd.qcut(df[score_col], q=10, labels=False, duplicates="drop") + 1

    decile_stats = (
        df.groupby(["decile", treatment_col])[outcome_col]
        .mean()
        .unstack(treatment_col)
        .rename(columns={0: "control_rate", 1: "treated_rate"})
    )
    decile_stats["uplift"] = decile_stats["treated_rate"] - decile_stats["control_rate"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: purchase rates side by side
    ax = axes[0]
    x  = np.arange(len(decile_stats))
    w  = 0.38
    ax.bar(x - w/2, decile_stats["treated_rate"] * 100, w,
           color=PALETTE["treated"], label="Treated", alpha=0.9)
    ax.bar(x + w/2, decile_stats["control_rate"] * 100, w,
           color=PALETTE["control"], label="Control", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{d}" for d in decile_stats.index])
    ax.set_xlabel("Uplift Score Decile (D1=lowest, D10=highest)")
    ax.set_ylabel("Purchase Rate (%)")
    ax.set_title("Purchase Rate by Decile", fontweight="bold")
    ax.legend(frameon=False)

    # Right: net uplift per decile
    ax2 = axes[1]
    colors = [PALETTE["model"] if v > 0 else "#E07A5F" for v in decile_stats["uplift"]]
    ax2.bar(x, decile_stats["uplift"] * 100, color=colors, alpha=0.9)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"D{d}" for d in decile_stats.index])
    ax2.set_xlabel("Uplift Score Decile")
    ax2.set_ylabel("Purchase Rate Lift (pp)")
    ax2.set_title("Incremental Lift by Decile", fontweight="bold")

    fig.suptitle("Uplift Model — Decile Analysis", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Decile chart saved → {save_path}")
    return fig


def plot_score_distribution(
    df_scored: pd.DataFrame,
    treatment_col: str = "treated",
    score_col: str     = "uplift_score",
    threshold: float   = None,
    save_path = None,
) -> plt.Figure:
    """
    Skill 6: KDE plot of uplift score distributions for treated vs control.
    Helps diagnose whether the model is separating the groups meaningfully.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for val, label, color in [(1, "Treated", PALETTE["treated"]),
                               (0, "Control",  PALETTE["control"])]:
        scores = df_scored[df_scored[treatment_col] == val][score_col]
        sns.kdeplot(scores, ax=ax, label=label, color=color, fill=True, alpha=0.3, lw=2)

    if threshold is not None:
        ax.axvline(threshold, color="red", ls="--", lw=1.5,
                   label=f"Break-even threshold ({threshold:.4f})")

    ax.set_xlabel("Predicted Uplift Score τ(x)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Uplift Score Distribution — Treated vs Control", fontsize=14, fontweight="bold")
    ax.legend(frameon=False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"[plot] Score distribution saved → {save_path}")
    return fig
