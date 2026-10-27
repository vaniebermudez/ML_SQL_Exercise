"""
models/uplift_models.py
-----------------------
Skill 3 — xgboost / lightgbm  (base learners)
Skill 4 — causalml             (T-Learner, X-Learner wrappers)

Three model classes are provided:

  TLearnerUplift   — trains one model on treated, one on control;
                     uplift = P(buy|treated) − P(buy|control).
                     Fast, interpretable, good starting point.

  XLearnerUplift   — cross-fits imputed treatment effects between groups;
                     more stable when treatment/control groups are very
                     different in size (like this 80/20 split).

  DRLearnerUplift  — doubly-robust estimator; requires both a propensity
                     model and an outcome model. Most statistically rigorous;
                     consistent even if one model is mis-specified.

All three share the same interface:
    model.fit(X_train, y_train, t_train)
    uplift_scores = model.predict(X_test)         # shape: (n,)
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

from uplift_model.config import LGBM_PARAMS, RANDOM_STATE, XGB_PARAMS


# ── Helper: base learner factory ──────────────────────────────────────────────

def _make_xgb(**overrides) -> XGBClassifier:
    params = {**XGB_PARAMS, **overrides}
    return XGBClassifier(**params)


def _make_lgbm(**overrides) -> LGBMClassifier:
    params = {**LGBM_PARAMS, **overrides}
    return LGBMClassifier(**params)


# ── T-Learner ─────────────────────────────────────────────────────────────────

class TLearnerUplift:
    """
    Two-model (T-Learner) uplift estimator.

    Algorithm
    ---------
    1. Fit model_1 on the treated sub-sample  → estimates μ₁(x) = P(Y=1 | T=1, X=x)
    2. Fit model_0 on the control sub-sample  → estimates μ₀(x) = P(Y=1 | T=0, X=x)
    3. Uplift τ(x) = μ₁(x) − μ₀(x)

    Trade-off
    ---------
    + Simple, interpretable, works out of the box with any classifier.
    − Each model sees only its own group; may overfit to group-specific
      feature distributions rather than the treatment effect itself.
    """

    def __init__(self, learner: str = "xgb"):
        self.learner = learner
        self.model_1 = None   # trained on treated
        self.model_0 = None   # trained on control

    def _make_learner(self):
        if self.learner == "xgb":
            return _make_xgb()
        elif self.learner == "lgbm":
            return _make_lgbm()
        else:
            raise ValueError(f"Unknown learner '{self.learner}'. Choose 'xgb' or 'lgbm'.")

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "TLearnerUplift":
        """
        Fit both sub-models.

        Parameters
        ----------
        X : pre-processed feature matrix, shape (n, p)
        y : binary purchase outcome, shape (n,)
        t : binary treatment indicator, shape (n,)
        """
        treated_mask  = t == 1
        control_mask  = t == 0

        print(f"[T-Learner] Fitting treated model  ({treated_mask.sum():,} samples)...")
        self.model_1 = self._make_learner()
        self.model_1.fit(X[treated_mask], y[treated_mask])

        print(f"[T-Learner] Fitting control model  ({control_mask.sum():,} samples)...")
        self.model_0 = self._make_learner()
        self.model_0.fit(X[control_mask], y[control_mask])

        print("[T-Learner] Done.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-customer uplift scores τ(x) = μ₁(x) − μ₀(x)."""
        mu1 = self.model_1.predict_proba(X)[:, 1]
        mu0 = self.model_0.predict_proba(X)[:, 1]
        return mu1 - mu0

    @property
    def treated_model(self):
        """Expose the treated-group model for SHAP analysis."""
        return self.model_1


# ── X-Learner ─────────────────────────────────────────────────────────────────

class XLearnerUplift:
    """
    X-Learner uplift estimator (Künzel et al., 2019).

    Algorithm
    ---------
    Stage 1 — Fit outcome models on each group (same as T-Learner):
        μ₁(x) ← fit on treated;   μ₀(x) ← fit on control

    Stage 2 — Impute counterfactual outcomes and estimate CATE:
        D₁ᵢ = Yᵢ − μ̂₀(Xᵢ)   for treated units  → τ₁ = fit on treated
        D₀ᵢ = μ̂₁(Xᵢ) − Yᵢ   for control units  → τ₀ = fit on control

    Stage 3 — Propensity-weighted combination:
        τ(x) = e(x) · τ₀(x) + (1 − e(x)) · τ₁(x)
        where e(x) = P(T=1 | X=x) is the propensity score.

    Trade-off
    ---------
    + Better variance when groups differ in size (80/20 split here).
    + Cross-fitting reduces bias from group-specific feature distributions.
    − One extra modeling step; requires a propensity model.
    """

    def __init__(self, learner: str = "xgb"):
        self.learner = learner
        # Stage 1
        self.mu1 = None
        self.mu0 = None
        # Stage 2
        self.tau1 = None
        self.tau0 = None
        # Stage 3
        self.propensity_model = None

    def _make_learner(self):
        if self.learner == "xgb":
            return _make_xgb()
        return _make_lgbm()

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "XLearnerUplift":
        treated = t == 1
        control = t == 0

        # ── Stage 1: outcome models ───────────────────────────────────────────
        print("[X-Learner] Stage 1 — fitting outcome models...")
        self.mu1 = self._make_learner()
        self.mu1.fit(X[treated], y[treated])

        self.mu0 = self._make_learner()
        self.mu0.fit(X[control], y[control])

        # ── Stage 2: imputed treatment effects ────────────────────────────────
        print("[X-Learner] Stage 2 — imputing counterfactual effects...")
        D1 = y[treated] - self.mu0.predict_proba(X[treated])[:, 1]
        D0 = self.mu1.predict_proba(X[control])[:, 1] - y[control]

        self.tau1 = self._make_learner()
        self.tau1.fit(X[treated], D1)

        self.tau0 = self._make_learner()
        self.tau0.fit(X[control], D0)

        # ── Stage 3: propensity model ─────────────────────────────────────────
        print("[X-Learner] Stage 3 — fitting propensity model...")
        # Logistic regression is sufficient and fast for the weighting step;
        # a complex model here rarely improves causal estimates meaningfully.
        self.propensity_model = LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE, n_jobs=-1
        )
        self.propensity_model.fit(X, t)

        print("[X-Learner] Done.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return weighted combination of τ₀ and τ₁ estimates."""
        e  = self.propensity_model.predict_proba(X)[:, 1]   # P(T=1|X)
        t1 = self.tau1.predict(X)
        t0 = self.tau0.predict(X)
        return e * t0 + (1 - e) * t1

    @property
    def treated_model(self):
        return self.mu1


# ── DR-Learner ────────────────────────────────────────────────────────────────

class DRLearnerUplift:
    """
    Doubly-Robust (DR) Learner uplift estimator.

    Algorithm
    ---------
    Constructs a doubly-robust pseudo-outcome for each observation:

        Ψᵢ = (Yᵢ − μ̂(Xᵢ, Tᵢ)) · (Tᵢ − ê(Xᵢ))
             ───────────────────────────────────── + μ̂(Xᵢ,1) − μ̂(Xᵢ,0)
                   ê(Xᵢ) · (1 − ê(Xᵢ))

    Then regresses Ψᵢ on Xᵢ to learn τ(x) = E[Ψ | X=x].

    Trade-off
    ---------
    + Doubly-robust: consistent if either the propensity model or the
      outcome model is correctly specified (but not necessarily both).
    + Most statistically rigorous; preferred for production.
    − Requires careful propensity clipping to avoid extreme weights.
    − More data-hungry than T/X-Learner.
    """

    PROPENSITY_CLIP = 0.01   # clip e(x) to [clip, 1-clip] to avoid ÷ near 0

    def __init__(self, learner: str = "xgb"):
        self.learner = learner
        self.outcome_model_1 = None
        self.outcome_model_0 = None
        self.propensity_model = None
        self.cate_model       = None

    def _make_learner(self):
        if self.learner == "xgb":
            return _make_xgb()
        return _make_lgbm()

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> "DRLearnerUplift":
        treated = t == 1
        control = t == 0

        # ── Outcome models ────────────────────────────────────────────────────
        print("[DR-Learner] Fitting outcome models...")
        self.outcome_model_1 = self._make_learner()
        self.outcome_model_1.fit(X[treated], y[treated])

        self.outcome_model_0 = self._make_learner()
        self.outcome_model_0.fit(X[control], y[control])

        # ── Propensity model ──────────────────────────────────────────────────
        print("[DR-Learner] Fitting propensity model...")
        self.propensity_model = LogisticRegression(
            max_iter=1000, C=1.0, random_state=RANDOM_STATE, n_jobs=-1
        )
        self.propensity_model.fit(X, t)

        # ── DR pseudo-outcome ─────────────────────────────────────────────────
        print("[DR-Learner] Computing doubly-robust pseudo-outcomes...")
        mu1 = self.outcome_model_1.predict_proba(X)[:, 1]
        mu0 = self.outcome_model_0.predict_proba(X)[:, 1]
        e   = self.propensity_model.predict_proba(X)[:, 1]
        e   = np.clip(e, self.PROPENSITY_CLIP, 1 - self.PROPENSITY_CLIP)

        # Direct effect term + IPW residual correction
        direct_term = mu1 - mu0
        ipw_term    = (y - (t * mu1 + (1 - t) * mu0)) * (t - e) / (e * (1 - e))
        psi         = direct_term + ipw_term

        # ── CATE model: regress pseudo-outcome on features ────────────────────
        print("[DR-Learner] Fitting CATE model on pseudo-outcomes...")
        # Use a regression variant: remove scale_pos_weight since psi is continuous
        self.cate_model = _make_xgb(
            objective="reg:squarederror",
            scale_pos_weight=1,
            eval_metric="rmse",
        )
        self.cate_model.fit(X, psi)

        print("[DR-Learner] Done.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return per-customer CATE estimates (the DR pseudo-outcome regression)."""
        return self.cate_model.predict(X)

    @property
    def treated_model(self):
        return self.outcome_model_1


# ── Model registry ────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "t_learner_xgb":  lambda: TLearnerUplift(learner="xgb"),
    "t_learner_lgbm": lambda: TLearnerUplift(learner="lgbm"),
    "x_learner_xgb":  lambda: XLearnerUplift(learner="xgb"),
    "x_learner_lgbm": lambda: XLearnerUplift(learner="lgbm"),
    "dr_learner_xgb": lambda: DRLearnerUplift(learner="xgb"),
}


def get_model(name: str):
    """Instantiate a model by registry key."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]()
