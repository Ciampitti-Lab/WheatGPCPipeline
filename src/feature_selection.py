"""Feature selection: prescreening, mRMR grid search, and RFE consensus."""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold, cross_val_score

logger = logging.getLogger("gpc_pipeline")


# ---------------------------------------------------------------------------
# Helper: get LightGBM estimator
# ---------------------------------------------------------------------------

def _get_cv_estimator(model_type="LightGBM", **kwargs):
    """Create a CV estimator for mRMR grid search."""
    model_type = model_type.lower()

    if model_type in ("lightgbm", "lgbm"):
        from lightgbm import LGBMRegressor
        defaults = {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        defaults.update(kwargs)
        return LGBMRegressor(**defaults)

    if model_type == "xgboost":
        from xgboost import XGBRegressor
        defaults = {
            "n_estimators": 200, "learning_rate": 0.05, "max_depth": 6,
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
        }
        defaults.update(kwargs)
        return XGBRegressor(**defaults)

    if model_type == "randomforest":
        from sklearn.ensemble import RandomForestRegressor
        defaults = {
            "n_estimators": 200, "max_depth": 10, "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        return RandomForestRegressor(**defaults)

    if model_type == "elasticnet":
        from sklearn.linear_model import ElasticNet
        defaults = {
            "alpha": 0.01, "l1_ratio": 0.5, "max_iter": 2000,
            "random_state": 42,
        }
        defaults.update(kwargs)
        return ElasticNet(**defaults)

    raise ValueError(f"Unknown cv_estimator model_type: {model_type}")


# ---------------------------------------------------------------------------
# Stage 1: Prescreening
# ---------------------------------------------------------------------------

def _remove_collinear(
    X: pd.DataFrame,
    y: np.ndarray,
    threshold: float = 0.85,
) -> List[str]:
    """
    Remove collinear features, keeping the one with highest |corr| to target.

    For each pair with |corr| > threshold, drop the feature less correlated
    with y.
    """
    # Target correlations (used to decide which to keep)
    target_corr = {}
    for col in X.columns:
        c = np.corrcoef(X[col].values, y)[0, 1]
        target_corr[col] = abs(c) if np.isfinite(c) else 0.0

    # Pairwise correlation matrix
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        correlated = upper.index[upper[col] > threshold].tolist()
        for other in correlated:
            if other in to_drop:
                continue
            # Drop the one less correlated with target
            if target_corr.get(col, 0) >= target_corr.get(other, 0):
                to_drop.add(other)
            else:
                to_drop.add(col)
                break  # col is dropped, move on

    kept = [f for f in X.columns if f not in to_drop]
    return kept


def _remove_collinear_unsupervised(
    X: pd.DataFrame,
    threshold: float = 0.90,
) -> List[str]:
    """
    Remove collinear features without using the target variable.

    For each pair with |corr| > threshold, drop the feature with lower
    variance (keep the more informative one in an unsupervised sense).
    """
    variances = X.var()

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    )

    to_drop: Set[str] = set()
    for col in upper.columns:
        if col in to_drop:
            continue
        # Use boolean indexing more carefully to avoid indexing errors
        col_values = upper[col]
        mask = col_values > threshold
        # Filter out NaN values that come from the where() operation
        mask = mask.fillna(False)
        correlated = upper.index[mask].tolist()
        for other in correlated:
            if other in to_drop:
                continue
            if variances[col] >= variances[other]:
                to_drop.add(other)
            else:
                to_drop.add(col)
                break

    kept = [f for f in X.columns if f not in to_drop]
    return kept


def unsupervised_prescreening(
    X: pd.DataFrame,
    config: dict,
) -> Tuple[List[str], dict]:
    """
    Unsupervised prescreening: variance filter + collinearity removal.

    Safe to run on full dataset (no target variable used).
    Intended to be called ONCE before the CV loop.
    """
    us_cfg = config["feature_selection"]["unsupervised_prescreening"]
    var_threshold = us_cfg.get("variance_threshold", 1e-6)
    collinear_thresh = us_cfg.get("collinear_threshold", None)

    n_input = X.shape[1]
    diagnostics: Dict[str, Any] = {"n_input": n_input}

    imputer = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index,
    )

    # Stage 1: Near-zero variance filter
    variances = X_clean.var()
    pass_var = variances[variances >= var_threshold].index.tolist()
    logger.info(
        "Unsupervised prescreening: variance filter (threshold=%.2e): %d → %d features",
        var_threshold, n_input, len(pass_var),
    )
    diagnostics["n_after_variance"] = len(pass_var)
    X_clean = X_clean[pass_var]

    # Stage 2: Collinearity removal (unsupervised — uses variance, not y)
    if collinear_thresh is not None and collinear_thresh > 0:
        n_before = X_clean.shape[1]
        passed = _remove_collinear_unsupervised(X_clean, threshold=collinear_thresh)
        logger.info(
            "Unsupervised prescreening: collinearity filter (threshold=%.2f): %d → %d features",
            collinear_thresh, n_before, len(passed),
        )
        diagnostics["n_after_collinearity"] = len(passed)
    else:
        passed = list(X_clean.columns)

    logger.info(
        "Unsupervised prescreening complete: %d → %d features",
        n_input, len(passed),
    )
    diagnostics["n_output"] = len(passed)
    return passed, diagnostics


def prescreening(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
) -> List[str]:
    """Pre-screen features using correlation, MI, and collinearity removal."""
    ps_cfg = config["feature_selection"]["prescreening"]
    corr_pct = ps_cfg["correlation_percentile"]
    mi_pct = ps_cfg["mi_percentile"]
    mi_neighbors = ps_cfg["mi_n_neighbors"]
    collinear_threshold = ps_cfg.get("collinear_threshold", None)
    random_state = config.get("random_state", 42)

    feature_names = list(X.columns)
    n_input = len(feature_names)

    # Correlation with target
    correlations = {}
    for col in feature_names:
        corr = np.corrcoef(X[col].values, y)[0, 1]
        correlations[col] = abs(corr) if np.isfinite(corr) else 0.0

    corr_threshold = np.percentile(list(correlations.values()), 100 - corr_pct)
    pass_corr = {f for f, c in correlations.items() if c >= corr_threshold}

    # Mutual Information
    logger.info("Prescreening: computing mutual information...")
    mi_scores = mutual_info_regression(
        X.values, y, random_state=random_state, n_neighbors=mi_neighbors,
    )
    mi_dict = dict(zip(feature_names, mi_scores))
    mi_threshold = np.percentile(mi_scores, 100 - mi_pct)
    pass_mi = {f for f, m in mi_dict.items() if m >= mi_threshold}

    # Intersection
    passed = sorted(pass_corr & pass_mi)

    logger.info(
        "Prescreening: %d → %d features (corr: %d pass, MI: %d pass, intersection: %d)",
        n_input, len(passed), len(pass_corr), len(pass_mi), len(passed),
    )

    # Collinearity removal (inter-feature)
    if collinear_threshold is not None and collinear_threshold > 0:
        n_before = len(passed)
        passed = _remove_collinear(X[passed], y, threshold=collinear_threshold)
        logger.info(
            "Prescreening: collinearity filter (threshold=%.2f): %d → %d features",
            collinear_threshold, n_before, len(passed),
        )

    return passed


# ---------------------------------------------------------------------------
# Stage 2: mRMR
# ---------------------------------------------------------------------------

def mrmr_select(
    X: pd.DataFrame,
    y: pd.Series,
    K: int,
) -> List[str]:
    """Select K features using mRMR."""
    from mrmr import mrmr_regression

    K = min(K, X.shape[1])
    selected = mrmr_regression(X=X, y=y, K=K, show_progress=False)
    return selected


def mrmr_grid_search(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Grid search over K values to find optimal mRMR feature count."""
    from mrmr import mrmr_regression

    fs_cfg = config["feature_selection"]["mrmr"]
    K_values = fs_cfg["K_values"]
    cv_folds = fs_cfg["cv_folds"]
    random_state = config.get("random_state", 42)
    cv_est_cfg = dict(fs_cfg.get("cv_estimator", {}))
    cv_model_type = cv_est_cfg.pop("model_type", "LightGBM")

    y_series = pd.Series(y, index=X.index)

    # Run mRMR with max K to get full ranking
    max_K = min(max(K_values), X.shape[1])
    logger.info("mRMR grid search: computing ranking (max K=%d)...", max_K)
    all_ranked = mrmr_regression(X=X, y=y_series, K=max_K, show_progress=False)
    logger.info("mRMR ranking complete. Evaluating K values with %s...", cv_model_type)

    estimator = _get_cv_estimator(cv_model_type, **cv_est_cfg)
    cv = GroupKFold(n_splits=min(cv_folds, len(np.unique(groups))))

    results = []
    for K in K_values:
        K = min(K, len(all_ranked))
        selected = all_ranked[:K]
        X_k = X[selected].values

        scores = cross_val_score(
            estimator, X_k, y, cv=cv, groups=groups,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )
        rmse = -scores.mean()
        std = scores.std()

        results.append({
            "K": K,
            "RMSE": rmse,
            "Std": std,
            "features": selected,
        })
        logger.info("  K=%3d → RMSE=%.4f (±%.4f)", K, rmse, std)

    results_df = pd.DataFrame([{
        "K": r["K"], "RMSE": r["RMSE"], "Std": r["Std"],
    } for r in results])

    best_idx = results_df["RMSE"].idxmin()
    best_K = int(results_df.loc[best_idx, "K"])
    best_rmse = float(results_df.loc[best_idx, "RMSE"])
    best_features = results[best_idx]["features"]

    logger.info("mRMR grid search: best K=%d, RMSE=%.4f", best_K, best_rmse)

    return {
        "best_K": best_K,
        "best_rmse": best_rmse,
        "results": results_df,
        "best_features": best_features,
        "all_ranked": all_ranked,
        "all_results": results,
    }


# ---------------------------------------------------------------------------
# Stage 3: RFE Consensus
# ---------------------------------------------------------------------------

def build_rfe_models(config: dict) -> Dict[str, Any]:
    """Build the 3 estimators for RFE consensus."""
    rfe_cfg = config["feature_selection"]["rfe_consensus"]["models"]
    random_state = config.get("random_state", 42)
    models = {}

    # LightGBM
    lgbm_cfg = rfe_cfg.get("LightGBM", {})
    from lightgbm import LGBMRegressor
    models["LightGBM"] = LGBMRegressor(
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **lgbm_cfg,
    )

    # RandomForest
    rf_cfg = rfe_cfg.get("RandomForest", {})
    from sklearn.ensemble import RandomForestRegressor
    models["RandomForest"] = RandomForestRegressor(
        random_state=random_state,
        n_jobs=-1,
        **rf_cfg,
    )

    # XGBoost
    xgb_cfg = rfe_cfg.get("XGBoost", {})
    from xgboost import XGBRegressor
    models["XGBoost"] = XGBRegressor(
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
        **xgb_cfg,
    )

    return models


def rfe_single_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    estimator: Any,
    feature_names: List[str],
    step: int = 5,
    min_features: int = 20,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """Run RFECV for a single model."""
    cv = GroupKFold(n_splits=min(cv_folds, len(np.unique(groups))))

    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        min_features_to_select=min_features,
        n_jobs=-1,
    )

    rfecv.fit(X, y, groups=groups)

    selected = [f for f, s in zip(feature_names, rfecv.support_) if s]
    best_rmse = -rfecv.cv_results_["mean_test_score"].max()

    return {
        "n_features": rfecv.n_features_,
        "best_rmse": best_rmse,
        "selected_features": selected,
        "ranking": rfecv.ranking_,
        "support": rfecv.support_,
        "cv_results": rfecv.cv_results_,
    }


def rfe_consensus(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Run RFE consensus: RFECV on 3 models, then intersect."""
    rfe_cfg = config["feature_selection"]["rfe_consensus"]
    step = rfe_cfg["step"]
    min_features = rfe_cfg["min_features"]
    cv_folds = rfe_cfg["cv_folds"]
    consensus_rule = rfe_cfg.get("consensus_rule", "all")

    feature_names = list(X.columns)
    X_arr = X.values

    # Build models
    models = build_rfe_models(config)

    per_model_results = {}

    for model_name, estimator in models.items():
        logger.info("RFE Consensus: running RFECV with %s...", model_name)
        result = rfe_single_model(
            X_arr, y, groups, estimator, feature_names,
            step=step, min_features=min_features, cv_folds=cv_folds,
        )
        per_model_results[model_name] = result
        logger.info(
            "  %s: %d features selected, RMSE=%.4f",
            model_name, result["n_features"], result["best_rmse"],
        )

    # Consensus voting
    all_sets = [set(r["selected_features"]) for r in per_model_results.values()]
    model_names = list(per_model_results.keys())

    # Intersection of all models
    consensus_all = all_sets[0]
    for s in all_sets[1:]:
        consensus_all = consensus_all & s

    # Features selected by at least 2 models
    any_two = set()
    for i in range(len(all_sets)):
        for j in range(i + 1, len(all_sets)):
            any_two |= (all_sets[i] & all_sets[j])
    any_two_features = sorted(any_two)

    # Apply consensus rule
    if consensus_rule == "majority":
        consensus_features = any_two_features
    else:
        consensus_features = sorted(consensus_all)

    # Overlap stats
    overlap_stats = {
        "total_unique": len(set().union(*all_sets)),
        "all_three": len(consensus_features),
        "at_least_two": len(any_two_features),
    }
    for name, result in per_model_results.items():
        overlap_stats[f"{name}_count"] = result["n_features"]

    logger.info(
        "RFE Consensus: ALL 3 models → %d features, at least 2 → %d features",
        len(consensus_features), len(any_two_features),
    )

    return {
        "consensus_features": consensus_features,
        "any_two_features": any_two_features,
        "per_model_results": per_model_results,
        "overlap_stats": overlap_stats,
    }


# ---------------------------------------------------------------------------
# Stage 2b: PLS-VIP feature selection
# ---------------------------------------------------------------------------

def _compute_vip(pls_model, X: np.ndarray) -> np.ndarray:
    """Compute VIP scores from a fitted PLS model."""
    # x_weights_ shape: (p, n_components)
    W = pls_model.x_weights_
    # x_scores_ shape: (n_samples, n_components)  = X @ x_rotations_
    T = pls_model.x_scores_
    # y_loadings_ shape: (1, n_components) for univariate y
    Q = pls_model.y_loadings_

    p, n_comp = W.shape
    # SS_a = (t_a' t_a)(q_a' q_a) for each component a
    SS = np.zeros(n_comp)
    for a in range(n_comp):
        SS[a] = (T[:, a] @ T[:, a]) * (Q[0, a] ** 2)

    total_SS = SS.sum()
    if total_SS == 0:
        return np.ones(p)

    # VIP for each feature
    vip = np.zeros(p)
    for j in range(p):
        s = 0.0
        for a in range(n_comp):
            s += W[j, a] ** 2 * SS[a]
        vip[j] = np.sqrt(p * s / total_SS)

    return vip


def pls_vip_select(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """PLS-VIP feature selection with CV-tuned n_components."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

    fs_cfg = config["feature_selection"]
    pls_cfg = fs_cfg.get("pls_vip", {})

    max_components = pls_cfg.get("max_components", 20)
    vip_threshold = pls_cfg.get("vip_threshold", 1.0)
    cv_folds = pls_cfg.get("cv_folds", 5)

    feature_names = list(X.columns)
    X_arr = X.values.copy()
    n_samples, n_features = X_arr.shape

    # Cap max_components at min(n_samples, n_features) - 1
    max_comp = min(max_components, n_features, n_samples - 1)
    comp_range = list(range(1, max_comp + 1))

    # --- Step 1: Select n_components via inner GroupKFold CV ---
    n_unique_groups = len(np.unique(groups))
    inner_cv = GroupKFold(n_splits=min(cv_folds, n_unique_groups))

    cv_rmse = {nc: [] for nc in comp_range}

    for train_idx, val_idx in inner_cv.split(X_arr, y, groups):
        X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        for nc in comp_range:
            pls = PLSRegression(n_components=nc, scale=False, max_iter=500)
            pls.fit(X_tr_sc, y_tr)
            y_pred = pls.predict(X_val_sc).ravel()
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            cv_rmse[nc].append(rmse)

    # Mean RMSE per n_components
    cv_results = pd.DataFrame([
        {"n_components": nc, "RMSE_mean": np.mean(scores), "RMSE_std": np.std(scores)}
        for nc, scores in cv_rmse.items()
    ])
    best_nc = int(cv_results.loc[cv_results["RMSE_mean"].idxmin(), "n_components"])
    best_rmse = float(cv_results.loc[cv_results["RMSE_mean"].idxmin(), "RMSE_mean"])

    logger.info(
        "PLS-VIP: optimal n_components=%d (RMSE=%.4f) from range [1, %d]",
        best_nc, best_rmse, max_comp,
    )

    # --- Step 2: Fit final PLS on all data → compute VIP ---
    scaler_full = StandardScaler()
    X_sc = scaler_full.fit_transform(X_arr)

    pls_final = PLSRegression(n_components=best_nc, scale=False, max_iter=500)
    pls_final.fit(X_sc, y)

    vip_scores = _compute_vip(pls_final, X_sc)
    vip_dict = dict(zip(feature_names, vip_scores))

    # --- Step 3: Select features with VIP > threshold ---
    selected = [f for f, v in vip_dict.items() if v >= vip_threshold]

    # Sort by VIP descending
    selected.sort(key=lambda f: vip_dict[f], reverse=True)

    # Fallback: if too few features, take top N by VIP
    min_features = pls_cfg.get("min_features", 10)
    if len(selected) < min_features:
        ranked = sorted(feature_names, key=lambda f: vip_dict[f], reverse=True)
        selected = ranked[:min_features]
        logger.warning(
            "PLS-VIP: only %d features above threshold %.2f, using top %d by VIP",
            sum(1 for v in vip_scores if v >= vip_threshold),
            vip_threshold, min_features,
        )

    logger.info(
        "PLS-VIP: %d features with VIP >= %.2f (out of %d)",
        len(selected), vip_threshold, n_features,
    )

    return {
        "selected_features": selected,
        "vip_scores": vip_dict,
        "n_components": best_nc,
        "cv_results": cv_results,
        "best_rmse": best_rmse,
        "vip_threshold": vip_threshold,
    }


# ---------------------------------------------------------------------------
# Stage 2c: Boruta feature selection
# ---------------------------------------------------------------------------

def boruta_select(
    X: pd.DataFrame,
    y: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """
    Boruta feature selection using Random Forest.

    Compares each feature against shadow features (shuffled copies).
    Keeps only features statistically better than random noise.
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    fs_cfg = config["feature_selection"]
    boruta_cfg = fs_cfg.get("boruta", {})
    random_state = config.get("random_state", 42)

    n_estimators = boruta_cfg.get("n_estimators", 200)
    max_depth = boruta_cfg.get("max_depth", 7)
    alpha = boruta_cfg.get("alpha", 0.05)
    max_iter = boruta_cfg.get("max_iter", 100)
    perc = boruta_cfg.get("perc", 100)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=random_state,
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators="auto",
        perc=perc,
        alpha=alpha,
        max_iter=max_iter,
        random_state=random_state,
        verbose=0,
    )

    boruta.fit(X.values, y)

    confirmed = X.columns[boruta.support_].tolist()
    tentative = X.columns[boruta.support_weak_].tolist()

    # Include tentative features if configured
    include_tentative = boruta_cfg.get("include_tentative", True)
    if include_tentative:
        selected = confirmed + tentative
    else:
        selected = confirmed

    # Fallback: if Boruta finds nothing, take top features by ranking
    if len(selected) == 0:
        ranking = boruta.ranking_
        min_rank = ranking.min()
        selected = X.columns[ranking == min_rank].tolist()
        logger.warning(
            "Boruta found no confirmed features. Using top-ranked: %d features",
            len(selected),
        )

    logger.info(
        "Boruta: %d confirmed, %d tentative → %d selected features",
        len(confirmed), len(tentative), len(selected),
    )

    return {
        "selected_features": selected,
        "confirmed": confirmed,
        "tentative": tentative,
        "ranking": boruta.ranking_.tolist(),
        "n_iterations": boruta.n_features_,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def feature_selection_pipeline(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Run the full feature selection pipeline (prescreening → mRMR/Boruta/PLS-VIP → RFE)."""
    fs_cfg = config["feature_selection"]
    random_state = config.get("random_state", 42)

    n_input = X.shape[1]
    logger.info("Feature selection pipeline: %d input features, %d samples", n_input, len(y))

    # Handle NaN
    imputer = SimpleImputer(strategy="median")
    X_clean = pd.DataFrame(
        imputer.fit_transform(X), columns=X.columns, index=X.index,
    )

    # Stage 1: Prescreening
    prescreening_result = None
    if fs_cfg["prescreening"].get("enabled", True):
        passed = prescreening(X_clean, y, config)
        prescreening_result = {
            "n_input": n_input,
            "n_passed": len(passed),
            "features": passed,
        }
        X_clean = X_clean[passed]
        logger.info("After prescreening: %d features", X_clean.shape[1])
    else:
        logger.info("Prescreening: disabled")

    # Stage 2: Feature selection method (mRMR, Boruta, or PLS-VIP)
    method = fs_cfg.get("method", "mrmr")  # "mrmr", "boruta", or "pls_vip"
    mrmr_result = None
    boruta_result = None
    pls_vip_result = None
    rfe_result = None

    if method == "pls_vip":
        # --- PLS-VIP ---
        pls_vip_result = pls_vip_select(X_clean, y, groups, config)
        selected_features = pls_vip_result["selected_features"]
        logger.info("After PLS-VIP: %d features", len(selected_features))

    elif method == "boruta":
        # --- Boruta ---
        boruta_result = boruta_select(X_clean, y, config)
        selected_features = boruta_result["selected_features"]
        logger.info("After Boruta: %d features", len(selected_features))

    else:
        # --- mRMR (default) ---
        override_K = fs_cfg["mrmr"].get("override_K", None)
        if override_K is not None and override_K > 0:
            K = min(override_K, X_clean.shape[1])
            logger.info("mRMR: fixed K=%d (skipping grid search)", K)
            ranked = mrmr_select(X_clean, pd.Series(y, index=X_clean.index), K)
            mrmr_features = ranked
            mrmr_result = {
                "best_K": K,
                "best_rmse": None,
                "best_features": ranked,
                "all_ranked": ranked,
            }
        else:
            mrmr_result = mrmr_grid_search(X_clean, y, groups, config)
            mrmr_features = mrmr_result["best_features"]

        logger.info("After mRMR (K=%d): %d features", len(mrmr_features), len(mrmr_features))

        # Stage 3: RFE Consensus (only with mRMR)
        if fs_cfg["rfe_consensus"].get("enabled", True):
            X_mrmr = X_clean[mrmr_features]
            rfe_result = rfe_consensus(X_mrmr, y, groups, config)
            selected_features = rfe_result["consensus_features"]
        else:
            logger.info("RFE Consensus disabled, using mRMR features directly")
            selected_features = mrmr_features

    logger.info(
        "Feature selection complete: %d → %d features",
        n_input, len(selected_features),
    )

    return {
        "selected_features": selected_features,
        "mrmr_result": mrmr_result,
        "boruta_result": boruta_result,
        "pls_vip_result": pls_vip_result,
        "rfe_result": rfe_result,
        "prescreening_result": prescreening_result,
    }
