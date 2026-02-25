"""Model building, training, and nested cross-validation for the GPC Pipeline."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    RandomizedSearchCV,
    RepeatedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_metrics

logger = logging.getLogger("gpc_pipeline")


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(
    name: str,
    param_grid: dict,
    random_state: int = 42,
) -> Tuple[Any, dict]:
    """Build a model estimator and its hyperparameter grid."""
    if name == "ElasticNet":
        from sklearn.linear_model import ElasticNet as EN
        est = EN(random_state=random_state)
        grid = {
            "max_iter": param_grid.get("max_iter", [1000]),
            "alpha": param_grid.get("alpha", [0.01, 0.1, 1.0]),
            "l1_ratio": param_grid.get("l1_ratio", [0.5]),
        }
        return est, grid

    elif name == "RandomForest":
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(random_state=random_state, n_jobs=1)
        grid = {
            "n_estimators": param_grid.get("n_estimators", [200, 400]),
            "max_depth": param_grid.get("max_depth", [10, 20, None]),
            "min_samples_leaf": param_grid.get("min_samples_leaf", [1, 2, 4]),
        }
        return est, grid

    elif name == "XGBoost":
        try:
            from xgboost import XGBRegressor
            est = XGBRegressor(
                objective="reg:squarederror",
                verbosity=0,
                random_state=random_state,
                n_jobs=1,
            )
            grid = {
                "n_estimators": param_grid.get("n_estimators", [300, 600]),
                "max_depth": param_grid.get("max_depth", [4, 6, 8]),
                "learning_rate": param_grid.get("learning_rate", [0.03, 0.05, 0.1]),
                "subsample": param_grid.get("subsample", [0.7, 0.9]),
            }
            return est, grid
        except ImportError:
            logger.error("xgboost not installed — skipping XGBoost")
            return None, {}

    elif name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
            est = LGBMRegressor(
                verbose=-1,
                random_state=random_state,
                n_jobs=1,
            )
            grid = {
                "n_estimators": param_grid.get("n_estimators", [300, 600]),
                "max_depth": param_grid.get("max_depth", [10, 20, -1]),
                "learning_rate": param_grid.get("learning_rate", [0.03, 0.05, 0.1]),
                "num_leaves": param_grid.get("num_leaves", [31, 63]),
                "subsample": param_grid.get("subsample", [0.7, 0.9]),
            }
            return est, grid
        except ImportError:
            logger.error("lightgbm not installed — skipping LightGBM")
            return None, {}

    elif name == "CatBoost":
        try:
            from catboost import CatBoostRegressor
            est = CatBoostRegressor(
                verbose=0,
                random_state=random_state,
            )
            grid = {
                "iterations": param_grid.get("iterations", [250, 500]),
                "depth": param_grid.get("depth", [5, 8]),
                "learning_rate": param_grid.get("learning_rate", [0.01, 0.1]),
            }
            return est, grid
        except ImportError:
            logger.error("catboost not installed — skipping CatBoost")
            return None, {}

    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# Single model training
# ---------------------------------------------------------------------------

def train_single_model(
    name: str,
    estimator: Any,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    inner_cv_folds: int = 5,
    inner_cv_repeats: int = 3,
    random_state: int = 42,
    n_iter: int = 15,
) -> Dict[str, Any]:
    """Train a single model with hyperparameter search."""
    inner_cv = RepeatedKFold(
        n_splits=inner_cv_folds,
        n_repeats=inner_cv_repeats,
        random_state=random_state,
    )

    # Determine number of combinations
    n_combos = 1
    for v in param_grid.values():
        if isinstance(v, list):
            n_combos *= len(v)

    if n_combos <= n_iter:
        search = GridSearchCV(
            estimator, param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
    else:
        search = RandomizedSearchCV(
            estimator, param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
            random_state=random_state,
        )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    train_pred = best_model.predict(X_train)
    test_pred = best_model.predict(X_test)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    logger.info(
        "  %s — best params: %s | Test R²: %.4f | Test RMSE: %.4f",
        name, search.best_params_, test_metrics["R2"], test_metrics["RMSE"],
    )

    return {
        "name": name,
        "predictions": test_pred,
        "best_params": search.best_params_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "fitted_model": best_model,
    }


# ---------------------------------------------------------------------------
# Stacking ensemble
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Build a stacking ensemble: RF + LightGBM → Ridge.

    Returns dict with predictions, metrics, model.
    """
    from sklearn.ensemble import RandomForestRegressor, StackingRegressor
    from sklearn.linear_model import RidgeCV

    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        logger.warning("LightGBM not available for stacking")
        return None

    base_models = [
        ("rf", RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=random_state, n_jobs=1,
        )),
        ("lgbm", LGBMRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            random_state=random_state, n_jobs=1, verbose=-1,
        )),
    ]

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
        cv=5,
        passthrough=False,
        n_jobs=-1,
    )

    stacking.fit(X_train, y_train)
    test_pred = stacking.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred)

    logger.info(
        "  Stacking (RF+LGBM→Ridge) — Test R²: %.4f | Test RMSE: %.4f",
        test_metrics["R2"], test_metrics["RMSE"],
    )

    return {
        "name": "Stacking",
        "predictions": test_pred,
        "best_params": {"method": "RF+LGBM→Ridge"},
        "train_metrics": {},
        "test_metrics": test_metrics,
        "fitted_model": stacking,
    }


# ---------------------------------------------------------------------------
# Nested CV pipeline
# ---------------------------------------------------------------------------

def nested_cv_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict,
    run_feature_selection: bool = False,
) -> Dict[str, Any]:
    """Run nested CV with all enabled models. Optionally runs feature selection inside each fold."""
    mod_cfg = config["modeling"]
    cv_cfg = mod_cfg["cv"]
    hp_cfg = mod_cfg["hyperparameters"]
    model_switches = mod_cfg["models"]
    random_state = config.get("random_state", 42)

    outer_splits = cv_cfg["outer_splits"]
    inner_splits = cv_cfg["inner_splits"]
    inner_repeats = cv_cfg["inner_repeats"]

    # Determine which models to run
    active_models = [
        name for name, enabled in model_switches.items()
        if enabled and name != "Stacking"
    ]
    run_stacking = model_switches.get("Stacking", False)

    logger.info(
        "Nested CV: %d outer folds, models: %s%s",
        outer_splits, active_models,
        " + Stacking" if run_stacking else "",
    )

    feature_names = list(X.columns)
    X_arr = X.values.astype(float)

    # --- Unsupervised prescreening (runs ONCE, before CV, no data leakage) ---
    unsupervised_result = None
    if run_feature_selection:
        us_cfg = config.get("feature_selection", {}).get("unsupervised_prescreening", {})
        if us_cfg.get("enabled", False):
            from .feature_selection import unsupervised_prescreening

            X_full_df = pd.DataFrame(X_arr, columns=feature_names)
            us_features, unsupervised_result = unsupervised_prescreening(
                X_full_df, config,
            )
            us_feat_idx = [feature_names.index(f) for f in us_features]
            X_arr = X_arr[:, us_feat_idx]
            feature_names = us_features

    n_splits = min(outer_splits, len(np.unique(groups)))
    random_state = config.get("random_state", 42)

    # Use StratifiedKFold with binned target for balanced folds
    n_bins = min(5, n_splits)
    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    logger.info(
        "Outer CV: StratifiedKFold (shuffle=True, %d splits, %d protein bins)",
        n_splits, len(np.unique(y_binned)),
    )

    fold_results = []
    summary = {m: {} for m in active_models + (["Stacking"] if run_stacking else []) + ["Ensemble"]}
    per_fold_features: List[List[str]] = []

    # OOF predictions storage
    oof_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y_binned)):
        logger.info("--- Fold %d/%d ---", fold_idx + 1, outer_splits)

        X_train_raw, X_test_raw = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups[test_idx]

        # Feature selection inside the fold (train only)
        if run_feature_selection:
            from .feature_selection import feature_selection_pipeline

            groups_train = groups[train_idx]
            X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
            fs_result = feature_selection_pipeline(
                X_train_df, y_train, groups_train, config,
            )
            fold_selected = fs_result["selected_features"]
            fold_feat_idx = [feature_names.index(f) for f in fold_selected]
            X_train = X_train_raw[:, fold_feat_idx]
            X_test = X_test_raw[:, fold_feat_idx]
            per_fold_features.append(fold_selected)
            logger.info(
                "  Feature selection: %d → %d features",
                len(feature_names), len(fold_selected),
            )
        else:
            X_train, X_test = X_train_raw, X_test_raw

        # Handle NaN with training medians
        for j in range(X_train.shape[1]):
            train_col = X_train[:, j]
            median_val = np.nanmedian(train_col)
            median_val = median_val if np.isfinite(median_val) else 0.0
            X_train[np.isnan(X_train[:, j]), j] = median_val
            X_test[np.isnan(X_test[:, j]), j] = median_val

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        fold_models = {}

        # Train each enabled model
        for model_name in active_models:
            grid = hp_cfg.get(model_name, {})
            estimator, param_grid = build_model(model_name, grid, random_state)

            if estimator is None:
                continue

            result = train_single_model(
                model_name, estimator, param_grid,
                X_train_s, y_train, X_test_s, y_test,
                inner_cv_folds=inner_splits,
                inner_cv_repeats=inner_repeats,
                random_state=random_state,
            )
            fold_models[model_name] = result

            # Store metrics
            for metric, value in result["test_metrics"].items():
                summary[model_name].setdefault(metric, []).append(value)

        # Stacking
        if run_stacking:
            stacking_result = build_stacking_ensemble(
                X_train_s, y_train, X_test_s, y_test, random_state,
            )
            if stacking_result:
                fold_models["Stacking"] = stacking_result
                for metric, value in stacking_result["test_metrics"].items():
                    summary["Stacking"].setdefault(metric, []).append(value)

        # Ensemble (mean of all base models)
        if fold_models:
            base_preds = [
                r["predictions"] for name, r in fold_models.items()
                if name != "Stacking"
            ]
            if base_preds:
                ensemble_pred = np.mean(base_preds, axis=0)
                ensemble_metrics = compute_metrics(y_test, ensemble_pred)
                fold_models["Ensemble"] = {
                    "name": "Ensemble",
                    "predictions": ensemble_pred,
                    "test_metrics": ensemble_metrics,
                }
                for metric, value in ensemble_metrics.items():
                    summary["Ensemble"].setdefault(metric, []).append(value)

                logger.info(
                    "  Ensemble — Test R²: %.4f | Test RMSE: %.4f",
                    ensemble_metrics["R2"], ensemble_metrics["RMSE"],
                )

        fold_results.append(fold_models)

        # Store OOF predictions
        for model_name, result in fold_models.items():
            for i, idx in enumerate(test_idx):
                oof_data.append({
                    "fold": fold_idx,
                    "field_key": groups[idx],
                    "y_true": y_test[i],
                    f"y_pred_{model_name}": result["predictions"][i],
                })

    oof_df = pd.DataFrame(oof_data)

    result = {
        "fold_results": fold_results,
        "summary": summary,
        "oof_predictions": oof_df,
        "feature_names": feature_names,
        "unsupervised_result": unsupervised_result,
    }
    if run_feature_selection:
        result["per_fold_features"] = per_fold_features
    return result
