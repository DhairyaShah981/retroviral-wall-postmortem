"""Active learning candidate selector for the wet-lab loop.

Given a fitted model + a pool of candidate RTs (with features), return the top-K
candidates most worth sending to the wet lab. The objective matches Mandrake's
real cost structure: missing a high-efficiency RT is far worse than misranking
a low-efficiency one.

Strategy: efficiency × uncertainty.
  - efficiency  = predicted PE efficiency (or active-class probability)
  - uncertainty = std of predictions across an ensemble (RF tree std, or
                  k-fold bagging std)
  - score       = (efficiency ** alpha) * (uncertainty ** beta)

α=2, β=1 is a sensible default: prioritize high-mean candidates with
non-trivial uncertainty (exploration-exploitation balance, weighted toward
exploitation since we don't get to do many rounds).
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def rf_predict_with_uncertainty(X_train, y_train, X_candidates, n_estimators=400, random_state=42):
    """Fit RF regressor; return (mean prediction, prediction std) for candidates."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_features="sqrt", min_samples_leaf=2,
        random_state=random_state, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    per_tree = np.stack([t.predict(X_candidates) for t in rf.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def next_batch(X_train, y_train, X_candidates, candidate_names=None, k=10,
               alpha=2.0, beta=1.0, model_fn=None):
    """Return top-K candidate indices for wet-lab testing.

    Parameters
    ----------
    X_train, y_train : training features + targets (efficiency works well)
    X_candidates     : candidate features
    candidate_names  : optional names (same order as X_candidates rows)
    k                : how many to select
    alpha, beta      : weights for efficiency vs uncertainty

    Returns
    -------
    list of dicts with: rank, name (if names given), predicted_efficiency,
    uncertainty, score, exploration_share
    """
    if model_fn is None:
        mean_pred, std_pred = rf_predict_with_uncertainty(X_train, y_train, X_candidates)
    else:
        mean_pred, std_pred = model_fn(X_train, y_train, X_candidates)

    # Clip to avoid negatives / zeros killing the geometric mean
    mp = np.clip(mean_pred, 1e-3, None)
    sp = np.clip(std_pred, 1e-3, None)
    score = (mp ** alpha) * (sp ** beta)

    order = np.argsort(-score)[:k]
    results = []
    for rank, idx in enumerate(order, start=1):
        results.append({
            "rank": rank,
            "name": candidate_names[idx] if candidate_names is not None else None,
            "index": int(idx),
            "predicted_efficiency": float(mean_pred[idx]),
            "uncertainty": float(std_pred[idx]),
            "score": float(score[idx]),
            "exploration_share": float((sp[idx] ** beta) / (mp[idx] ** alpha + sp[idx] ** beta + 1e-9)),
        })
    return results


def regret_simulation(y_true_pool, mean_pred, std_pred, k=10, alpha=2.0, beta=1.0):
    """If we sent the top-K candidates to wet-lab, what fraction of the total
    *realised* efficiency would we have captured? Useful for evaluating the
    selection function offline using held-out folds.
    """
    mp = np.clip(mean_pred, 1e-3, None)
    sp = np.clip(std_pred, 1e-3, None)
    score = (mp ** alpha) * (sp ** beta)
    order = np.argsort(-score)[:k]
    captured = np.asarray(y_true_pool)[order].sum()
    total = np.asarray(y_true_pool).sum()
    return {
        "captured": float(captured),
        "total": float(total),
        "capture_rate": float(captured / max(total, 1e-9)),
        "top_k_indices": order.tolist(),
    }
