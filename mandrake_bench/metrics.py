"""CLS (Cross-Lineage Score) metric — exact match to Mandrake's official evaluator.

CLS = harmonic_mean(PR-AUC, Weighted Spearman)

Note: the challenge page says weights = pe_efficiency + 0.1, but the official
evaluator code uses EPSILON = 0.01. We follow the code (it's what gets scored)
and expose EPSILON as a kwarg so the page-spec can be tested too.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

EPSILON_CODE = 0.01   # what evaluation/evaluate.py actually uses
EPSILON_PAGE = 0.1    # what the challenge page text says


def weighted_spearman(y_score, true_efficiency, weights):
    pred_ranks = np.argsort(np.argsort(y_score)).astype(float)
    true_ranks = np.argsort(np.argsort(true_efficiency)).astype(float)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    mu_p = np.dot(w, pred_ranks)
    mu_t = np.dot(w, true_ranks)
    dp = pred_ranks - mu_p
    dt = true_ranks - mu_t
    cov = np.sum(w * dp * dt)
    std_p = np.sqrt(np.sum(w * dp ** 2))
    std_t = np.sqrt(np.sum(w * dt ** 2))
    if std_p < 1e-12 or std_t < 1e-12:
        return 0.0
    return cov / (std_p * std_t)


def cls(y_true, y_score, pe_efficiency, epsilon=EPSILON_CODE):
    """Returns dict with cls, pr_auc, w_spearman. Floors w_spearman at 0."""
    pr_auc = average_precision_score(y_true, y_score)
    weights = np.asarray(pe_efficiency, dtype=float) + epsilon
    ws = weighted_spearman(y_score, pe_efficiency, weights)
    ws = max(ws, 0.0)
    if pr_auc <= 0 or ws <= 0:
        score = 0.0
    else:
        score = 2.0 * pr_auc * ws / (pr_auc + ws)
    return {"cls": score, "pr_auc": pr_auc, "w_spearman": ws, "epsilon": epsilon}


def per_family_pr_auc(y_true, y_score, families):
    out = {}
    families = np.asarray(families)
    for fam in sorted(set(families)):
        mask = families == fam
        n, na = int(mask.sum()), int(np.asarray(y_true)[mask].sum())
        if 0 < na < n:
            out[fam] = {
                "n": n,
                "active": na,
                "pr_auc": float(average_precision_score(np.asarray(y_true)[mask], np.asarray(y_score)[mask])),
            }
        else:
            out[fam] = {"n": n, "active": na, "pr_auc": None}
    return out
