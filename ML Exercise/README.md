# Uplift Modeling Framework

This project implements an end-to-end uplift modeling pipeline for customer targeting and campaign optimization. It is designed to estimate the incremental impact of a treatment (for example, a discount) on customer behavior and identify which customers are most likely to respond positively to the intervention.

The pipeline prepares data, trains uplift models, scores customers, evaluates the model, estimates campaign economics, and produces explainability outputs such as SHAP plots and top uplift drivers.

## Overview

The main entry point is `main.py`, which orchestrates the full workflow:

1. Data preparation
2. Model training
3. Scoring
4. Campaign economics and targeting decisions
5. Evaluation metrics
6. Plot generation
7. SHAP explainability
8. Export of scored output

The project is configured around campaign data and discount targeting, with model behavior and evaluation thresholds controlled in `config.py`.

## Key Features

- Uplift modeling for treatment effect estimation
- Support for multiple uplift model variants, including T-learner, X-learner, and DR-learner setups
- Automated data preparation and feature engineering
- Customer-level uplift scoring
- Break-even targeting based on campaign economics
- Qini analysis and bootstrap confidence intervals
- Plots for uplift curves, decile performance, and score distributions
- SHAP-based model explainability
- Export of scored customer results to CSV

## Project Structure

- `main.py` — pipeline entry point and CLI runner
- `config.py` — central project configuration, feature definitions, and economic parameters
- `preprocessing.py` — data loading, cleaning, feature engineering, and train/test split logic
- `uplift_models.py` — model registry and uplift model definitions
- `economics.py` — break-even threshold and campaign targeting logic
- `metrics.py` — evaluation metrics and plotting utilities
- `explainability.py` — SHAP value computation and driver extraction
- `persistence.py` — model serialization utilities
- `data/` — campaign dataset folder (for example, `data/campaign_data.xlsx`)
- `outputs/` — generated model artifacts, plots, and scored outputs

## Data Inputs

The project expects a campaign dataset stored in `data/campaign_data.xlsx` by default. The configuration file defines the expected schema and derived fields used by the pipeline, including:

- `CUSTOMER_ID`
- `TARGETED_DISCOUNT` / `treated`
- `purchased`
- `CMPGN_NET_REV`
- campaign and transaction history fields for recency, purchase behavior, and revenue metrics

The configuration also includes engineered features such as recency, loyalty tenure, revenue trend, basket size, and purchase velocity.

## Installation

This project uses Python with common data science libraries.

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install numpy pandas scikit-learn xgboost shap matplotlib seaborn openpyxl
```

Depending on the environment, additional dependencies may also be required for the specific model library versions used in the project.

## Usage

Run the full pipeline from the project root:

```bash
python main.py --model t_learner_xgb --data data/campaign_data.xlsx
```

Other examples from the project:

```bash
python main.py --model x_learner_xgb --skip-shap
python main.py --model dr_learner_xgb --n-bootstrap 1000
```

### Command line options

- `--model` — uplift model name to train and evaluate
- `--data` — path to the input campaign Excel file
- `--n-bootstrap` — number of bootstrap iterations for Qini confidence intervals
- `--skip-shap` — skip SHAP calculation for faster development runs

## What the Pipeline Produces

The output pipeline generates:

- trained model artifacts in `outputs/models/`
- evaluation plots in `outputs/`
- scored customer results such as `scored_customers_<model_name>.csv`
- SHAP summary plots and top driver CSV files when SHAP is enabled

Typical generated outputs include:

- `qini_curve_<model_name>.png`
- `decile_chart_<model_name>.png`
- `score_dist_<model_name>.png`
- `shap_summary_<model_name>.png`
- `shap_drivers_<model_name>.csv`
- `scored_customers_<model_name>.csv`

## Evaluation Metrics

The project computes and reports:

- Qini coefficient
- bootstrap confidence interval for Qini
- top-decile revenue test
- break-even uplift threshold
- targeting summary by customer tier

These outputs help evaluate whether the uplift model is performing well enough to justify targeting customers with offers or discounts.

## Notes

This framework is tailored to campaign uplift use cases, where the objective is not only to predict conversion, but also to estimate the incremental effect of targeting a customer with a treatment versus leaving them untreated.

To adapt the project to a different campaign dataset, update the constants in `config.py` and ensure the target columns align with the expected treatment and outcome definitions.
