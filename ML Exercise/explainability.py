"""
evaluation/explainability.py
----------------------------
Skill 5 — shap

SHAP (SHapley Additive exPlanations) decomposes each uplift prediction
into per-feature contributions, making the model interpretable to the
Marketing team.

Because uplift = μ₁(x) − μ₀(x), SHAP is applied to the treated-group
model (μ₁). This tells us: for customers with high uplift scores, which
features drive the treated-group purchase probability up?

Functions:
  compute_shap_values     — compute raw SHAP values matrix
  plot_shap_summary       — beeswarm summary (global feature importance)
  plot_shap_waterfall     — single-customer prediction breakdown
  plot_shap_dependence    — feature interaction with uplift score
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from uplift_model.config import OUTPUT_DIR

warnings.filterwarnings("ignore")
shap.initjs()


def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 5_000,
) -> shap.Explanation:
    """
    Compute SHAP values for the treated-group model.

    TreeExplainer is exact and fast for XGBoost / LightGBM.
    We cap at max_samples for speed on the full 485K dataset;
    a random sample is sufficient for robust feature importance.

    Parameters
    ----------
    model          : fitted uplift model with a .treated_model property
    X              : preprocessed feature matrix (test set)
    feature_names  : list of feature names matching X columns
    max_samples    : cap sample size for speed

    Returns
    -------
    shap.Explanation object (supports .values, .data, .feature_names)
    """
    base_model = model.treated_model

    if X.shape[0] > max_samples:
        rng   = np.random.default_rng(42)
        idx   = rng.choice(X.shape[0], size=max_samples, replace=False)
        X_sub = X[idx]
        print(f"[SHAP] Subsampled to {max_samples:,} rows for speed.")
    else:
        X_sub = X

    explainer = shap.TreeExplainer(base_model)
    shap_vals = explainer(X_sub)

    # Attach feature names so plots are labelled correctly
    shap_vals.feature_names = feature_names
    return shap_vals


def plot_shap_summary(
    shap_explanation: shap.Explanation,
    max_display: int = 20,
    save_path = None,
) -> plt.Figure:
    """
    Beeswarm plot: global feature importance with direction of effect.

    Each dot is one customer; position on x-axis = SHAP value (impact
    on model output); colour = feature value (red=high, blue=low).

    Interpretation: features with wide SHAP distributions and rightward
    skew on high-value (red) customers are the key drivers of predicted
    purchase probability in the treated group.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_explanation.values,
        shap_explanation.data,
        feature_names=shap_explanation.feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title("SHAP Feature Importance — Treated-Group Model",
              fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SHAP] Summary plot saved → {save_path}")
    return plt.gcf()


def plot_shap_waterfall(
    shap_explanation: shap.Explanation,
    customer_idx: int = 0,
    save_path = None,
) -> plt.Figure:
    """
    Waterfall plot for a single customer: shows how each feature
    pushes the prediction up or down from the base value.

    Useful for explaining a specific targeting decision (e.g. "why
    was this customer selected for the discount?").
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_explanation[customer_idx], show=False)
    plt.title(f"SHAP Waterfall — Customer #{customer_idx}",
              fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SHAP] Waterfall plot saved → {save_path}")
    return plt.gcf()


def plot_shap_dependence(
    shap_explanation: shap.Explanation,
    feature: str,
    interaction_feature: str = "auto",
    save_path = None,
) -> plt.Figure:
    """
    Dependence plot: SHAP value for `feature` vs its actual value,
    coloured by an interaction feature.

    Reveals non-linearities and thresholds in how a feature affects
    the model (e.g. "L12M_NUM_TRANS has diminishing returns above 5").
    """
    feat_names = shap_explanation.feature_names
    feat_idx   = feat_names.index(feature)

    inter_idx  = "auto"
    if interaction_feature != "auto":
        inter_idx = feat_names.index(interaction_feature)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.dependence_plot(
        feat_idx,
        shap_explanation.values,
        shap_explanation.data,
        feature_names=feat_names,
        interaction_index=inter_idx,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence: {feature}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"[SHAP] Dependence plot saved → {save_path}")
    return fig


def top_uplift_drivers(
    shap_explanation: shap.Explanation,
    n: int = 10,
) -> pd.DataFrame:
    """
    Return a DataFrame of the top-n features by mean absolute SHAP value
    (treated model). Suitable for executive reporting.
    """
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    drivers = pd.DataFrame({
        "feature":         shap_explanation.feature_names,
        "mean_abs_shap":   mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(n).reset_index(drop=True)
    drivers.index += 1
    return drivers
