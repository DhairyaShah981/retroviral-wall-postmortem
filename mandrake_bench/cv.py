"""Leave-One-Family-Out cross-validation."""
from __future__ import annotations
import numpy as np
import pandas as pd


def lofo_predict(X, y, families, fit_predict_fn, pe_efficiency=None, return_per_family=False):
    """Run LOFO and return out-of-fold predictions aligned to original index.

    fit_predict_fn(X_train, y_train, X_test, **kwargs) -> y_score_test
    kwargs may include 'pe_train' if pe_efficiency is provided.

    Returns
    -------
    np.ndarray of shape (n,) — out-of-fold scores
    (optional) dict per-family with held-out indices
    """
    families = np.asarray(families)
    y = np.asarray(y)
    oof = np.zeros(len(y), dtype=float)
    per_fam_idx = {}
    for fam in sorted(set(families)):
        test_mask = families == fam
        train_mask = ~test_mask
        kwargs = {}
        if pe_efficiency is not None:
            kwargs["pe_train"] = np.asarray(pe_efficiency)[train_mask]
        scores = fit_predict_fn(
            X[train_mask] if hasattr(X, "__getitem__") else None,
            y[train_mask],
            X[test_mask] if hasattr(X, "__getitem__") else None,
            **kwargs,
        )
        oof[test_mask] = np.asarray(scores).ravel()
        per_fam_idx[fam] = np.where(test_mask)[0].tolist()
    if return_per_family:
        return oof, per_fam_idx
    return oof
