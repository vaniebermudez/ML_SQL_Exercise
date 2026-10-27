"""
main.py
-------
End-to-end pipeline orchestrator.

Usage:
    python main.py --model t_learner_xgb --data data/campaign_data.xlsx
    python main.py --model x_learner_xgb --skip-shap
    python main.py --model dr_learner_xgb --n-bootstrap 1000

This script ties together all modules:
  1. Data preparation  (preprocessing.py)
  2. Model training    (uplift_models.py)
  3. Scoring           (model.predict)
  4. Economics         (economics.py)
  5. Evaluation        (metrics.py)
  6. Explainability    (explainability.py)
  7. Output            (save model, plots, scored CSV)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from uplift_model.config import (
    DATA_PATH,
    MODEL_DIR,
    OUTPUT_DIR,
    TREATMENT_COL,
)
from uplift_model.data.preprocessing import prepare_data
from uplift_model.models.uplift_models import get_model
from uplift_model.models.persistence import save_model
from uplift_model.evaluation.economics import (
    compute_breakeven_threshold,
    add_targeting_decision,
    summarise_campaign_economics,
)
from uplift_model.evaluation.metrics import (
    qini_coefficient,
    bootstrap_qini_ci,
    top_decile_revenue_test,
    plot_qini_curve,
    plot_uplift_by_decile,
    plot_score_distribution,
)
from uplift_model.evaluation.explainability import (
    compute_shap_values,
    plot_shap_summary,
    top_uplift_drivers,
)


def run_pipeline(
    model_name: str  = "t_learner_xgb",
    data_path: Path  = DATA_PATH,
    n_bootstrap: int = 500,
    skip_shap: bool  = False,
) -> dict:
    """
    Full end-to-end uplift modeling pipeline.

    Returns a results dict with model, scores, metrics, and scored DataFrame.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Data preparation ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 1 — Data Preparation")
    print("="*60)
    data = prepare_data(data_path)

    X_train       = data["X_train"]
    X_test        = data["X_test"]
    y_train       = data["y_train"]
    y_test        = data["y_test"]
    t_train       = data["t_train"]
    t_test        = data["t_test"]
    df_train      = data["df_train"]
    df_test       = data["df_test"]
    feature_names = data["feature_names"]

    # ── Step 2: Model training ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"STEP 2 — Model Training: {model_name}")
    print("="*60)
    model = get_model(model_name)
    model.fit(X_train, y_train, t_train)
    save_model(model, model_name)

    # ── Step 3: Scoring ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3 — Scoring Test Set")
    print("="*60)
    uplift_scores = model.predict(X_test)
    print(f"[score] Score range: [{uplift_scores.min():.5f}, {uplift_scores.max():.5f}]")
    print(f"[score] Mean score:  {uplift_scores.mean():.5f}")
    print(f"[score] % positive:  {(uplift_scores > 0).mean():.1%}")

    # ── Step 4: Break-even economics ──────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4 — Campaign Economics")
    print("="*60)
    # Compute threshold from control arm of TRAINING set (not test — no leakage)
    df_train_control = df_train[df_train[TREATMENT_COL] == 0]
    threshold = compute_breakeven_threshold(df_train_control)

    df_test_scored = add_targeting_decision(df_test, uplift_scores, threshold)
    summary = summarise_campaign_economics(df_test_scored)

    # ── Step 5: Evaluation metrics ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5 — Uplift Model Evaluation")
    print("="*60)
    qini_point, qini_lo, qini_hi = bootstrap_qini_ci(
        y_test, t_test, uplift_scores, n_bootstrap=n_bootstrap
    )
    ttest_result = top_decile_revenue_test(df_test_scored)

    # ── Step 6: Visualisations ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 6 — Generating Plots")
    print("="*60)
    plot_qini_curve(
        y_test, t_test, uplift_scores,
        model_name=model_name,
        save_path=OUTPUT_DIR / f"qini_curve_{model_name}.png",
    )
    plot_uplift_by_decile(
        df_test_scored,
        save_path=OUTPUT_DIR / f"decile_chart_{model_name}.png",
    )
    plot_score_distribution(
        df_test_scored,
        threshold=threshold,
        save_path=OUTPUT_DIR / f"score_dist_{model_name}.png",
    )

    # ── Step 7: SHAP explainability ───────────────────────────────────────────
    if not skip_shap:
        print("\n" + "="*60)
        print("STEP 7 — SHAP Explainability")
        print("="*60)
        shap_vals = compute_shap_values(model, X_test, feature_names)
        plot_shap_summary(
            shap_vals,
            save_path=OUTPUT_DIR / f"shap_summary_{model_name}.png",
        )
        drivers = top_uplift_drivers(shap_vals)
        print("\n[SHAP] Top uplift drivers:")
        print(drivers.to_string())
        drivers.to_csv(OUTPUT_DIR / f"shap_drivers_{model_name}.csv", index=False)
    else:
        shap_vals = None
        drivers   = None

    # ── Step 8: Save scored output ────────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 8 — Saving Outputs")
    print("="*60)
    out_csv = OUTPUT_DIR / f"scored_customers_{model_name}.csv"
    cols_to_save = [
        "CUSTOMER_ID", TREATMENT_COL, "purchased", "CMPGN_NET_REV",
        "uplift_score", "send_discount", "targeting_tier",
    ]
    df_test_scored[cols_to_save].to_csv(out_csv, index=False)
    print(f"[output] Scored customers saved → {out_csv}")

    results = {
        "model":         model,
        "df_scored":     df_test_scored,
        "uplift_scores": uplift_scores,
        "threshold":     threshold,
        "qini_point":    qini_point,
        "qini_ci":       (qini_lo, qini_hi),
        "ttest":         ttest_result,
        "summary":       summary,
        "shap_values":   shap_vals,
        "shap_drivers":  drivers,
    }

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"  Model:          {model_name}")
    print(f"  Qini:           {qini_point:.4f}  [{qini_lo:.4f}, {qini_hi:.4f}]")
    print(f"  Break-even τ:   {threshold:.5f}")
    print(f"  Customers to target (HIGH): "
          f"{(df_test_scored['targeting_tier']=='HIGH').sum():,}")
    print("="*60)

    return results


# ── CLI entry point ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Uplift Modeling Pipeline")
    parser.add_argument("--model",       type=str,  default="t_learner_xgb",
                        help="Model to use (see MODEL_REGISTRY in uplift_models.py)")
    parser.add_argument("--data",        type=Path, default=DATA_PATH,
                        help="Path to campaign Excel file")
    parser.add_argument("--n-bootstrap", type=int,  default=500,
                        help="Bootstrap iterations for Qini CI")
    parser.add_argument("--skip-shap",   action="store_true",
                        help="Skip SHAP computation (faster, for dev runs)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        model_name  = args.model,
        data_path   = args.data,
        n_bootstrap = args.n_bootstrap,
        skip_shap   = args.skip_shap,
    )
