"""
Model training, evaluation and comparison engine.
Trains Logistic Regression, Random Forest, XGBoost, Decision Tree.

Compatibility: sklearn 1.2 – 1.8+
  - LogisticRegression: explicit penalty='l2' (multi_class removed in 1.8)
  - All predictions use safe_predict_proba() which sanitizes inputs to
    float64 and guards against NaN/Inf before calling model.predict_proba()
  - MODEL_REGISTRY built via factory to avoid shared mutable state between
    Streamlit reruns (each training run gets fresh model instances)
"""

import numpy as np
import pandas as pd
import joblib
import logging
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def _get_model_registry() -> Dict:
    """
    Return fresh model instances on every call.

    Why a factory instead of a module-level dict:
    - Prevents shared mutable state across Streamlit reruns / threads
    - Ensures each training run starts with clean, unfitted estimators
    - Avoids sklearn clone() issues with models that store runtime state

    Why explicit penalty='l2' on LogisticRegression:
    - sklearn 1.2 deprecated the multi_class parameter
    - sklearn 1.8 fully removed multi_class — accessing it raises AttributeError
    - Omitting penalty causes penalty='deprecated' in sklearn 1.8 internals,
      which can trigger downstream AttributeError in some call paths
    - Explicit penalty='l2' + solver='lbfgs' is safe across sklearn 1.0 – 1.8+
    """
    return {
        "Logistic Regression": LogisticRegression(
            penalty="l2",           # explicit — avoids 'deprecated' default in sklearn 1.8
            solver="lbfgs",         # compatible with l2, supports predict_proba
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=10,
            class_weight="balanced", random_state=42,
        ),
    }


# ---------------------------------------------------------------------------
# Safe prediction helpers
# ---------------------------------------------------------------------------

def _sanitize_X(X: np.ndarray) -> np.ndarray:
    """
    Convert input to float64 and replace any NaN / Inf with safe values.

    Why this is needed:
    - pd.get_dummies produces bool columns; bool arrays can fail sklearn
      internal dtype validation in certain sklearn 1.7/1.8 code paths.
    - User-uploaded CSVs may contain NaN that survives imputation for
      unseen categorical levels, causing predict_proba to raise ValueError
      which in some sklearn versions surfaces as an AttributeError on
      internal model attributes (e.g. multi_class).
    - np.nan_to_num replaces NaN→0, Inf→large finite, -Inf→large negative,
      which is safe because StandardScaler has already centred the data.
    """
    X = np.array(X, dtype=np.float64)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


def safe_predict_proba(model, X: np.ndarray) -> np.ndarray:
    """
    Model-agnostic predict_proba with defensive input sanitization.

    Works for: LogisticRegression, RandomForest, XGBoost, DecisionTree
    without assuming any model-specific attributes (multi_class, coef_, etc.)

    Returns churn probability (class-1 column), shape (n_samples,).
    Falls back to predict() if predict_proba is unavailable.
    """
    X = _sanitize_X(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            # Guard: some multiclass-trained models return > 2 columns
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return proba.ravel()
        # Fallback for models without predict_proba
        return model.predict(X).astype(float)


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Comprehensive model evaluation with sanitized inputs."""
    X_test = _sanitize_X(X_test)
    y_pred = model.predict(X_test)
    y_prob  = safe_predict_proba(model, X_test)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
        "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        "recall": round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
        "f1": round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
        "roc_auc": round(roc_auc_score(y_test, y_prob) * 100, 2),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def train_all_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: Optional[list] = None,
) -> Tuple[Dict, str, Dict]:
    """
    Train all models, evaluate them, return (results, best_name, trained_models).
    Uses fresh model instances from factory to avoid shared mutable state.
    All array inputs are sanitized to float64 before training/evaluation.
    """
    # Sanitize inputs once for the entire training run
    X_train = _sanitize_X(X_train)
    X_test  = _sanitize_X(X_test)

    results        = {}
    trained_models = {}
    registry       = _get_model_registry()   # fresh instances every call

    for name, model in registry.items():
        logger.info(f"Training {name}...")
        t0 = time.time()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)

            elapsed = round(time.time() - t0, 2)
            metrics = evaluate_model(model, X_test, y_test)
            metrics["train_time_sec"] = elapsed

            # Cross-validation (sanitized X_train already)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=cv, scoring="roc_auc", n_jobs=-1
                )
            metrics["cv_roc_auc_mean"] = round(cv_scores.mean() * 100, 2)
            metrics["cv_roc_auc_std"]  = round(cv_scores.std()  * 100, 2)

            # Feature importance — model-agnostic attribute checks (never assume)
            if feature_names:
                if hasattr(model, "feature_importances_"):
                    fi = dict(zip(feature_names, model.feature_importances_))
                    metrics["feature_importance"] = dict(
                        sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
                    )
                elif hasattr(model, "coef_") and model.coef_ is not None:
                    coef = model.coef_
                    # coef_ may be 2-D for multiclass; flatten safely
                    importance = np.abs(coef[0] if coef.ndim == 2 else coef)
                    fi = dict(zip(feature_names, importance))
                    metrics["feature_importance"] = dict(
                        sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
                    )

            results[name]        = metrics
            trained_models[name] = model
            logger.info(f"  {name}: AUC={metrics['roc_auc']}%, F1={metrics['f1']}%")

        except Exception as e:
            logger.error(f"Failed to train {name}: {e}", exc_info=True)
            continue

    if not results:
        raise RuntimeError("All models failed to train. Check logs for details.")

    # Select best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    logger.info(f"Best model: {best_name} (AUC={results[best_name]['roc_auc']}%)")

    return results, best_name, trained_models


def save_artifacts(
    preprocessor,
    model,
    model_name: str,
    results: Dict,
    feature_names: list,
    output_dir: Path,
):
    """Persist all artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, output_dir / "preprocessor.pkl")
    joblib.dump(model, output_dir / "best_model.pkl")
    joblib.dump({"name": model_name, "results": results, "feature_names": feature_names},
                output_dir / "model_metadata.pkl")

    logger.info(f"Artifacts saved to {output_dir}")


def load_artifacts(model_dir: Path):
    """Load persisted artifacts."""
    preprocessor = joblib.load(model_dir / "preprocessor.pkl")
    model = joblib.load(model_dir / "best_model.pkl")
    metadata = joblib.load(model_dir / "model_metadata.pkl")
    return preprocessor, model, metadata
