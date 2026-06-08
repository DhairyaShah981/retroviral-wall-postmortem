"""Memorization / leakage diagnostics for participant submissions.

The motivation: in the public Kaggle leaderboard, top 6 entries are all tied at
1.00000 on a 57-sample LOFO problem. That's a methodology-tell, not a signal.
These functions catch common ways that happens:

  - degeneracy: predictions are mostly constant or binary, which can game PR-AUC
    on imbalanced data
  - family-leakage: predictions on a held-out family match family-prior much
    better than they should
  - rank vs class mismatch: model classifies well but ranks badly (or vice versa)
  - shuffle-null: how much CLS comes from genuine signal vs class imbalance
"""
from __future__ import annotations
import numpy as np
from .metrics import cls as cls_metric, weighted_spearman


def degeneracy(y_score):
    """Return diagnostics on score distribution. Flags if predictions are degenerate."""
    s = np.asarray(y_score, dtype=float)
    unique = np.unique(np.round(s, 4))
    return {
        "n_unique_values": int(len(unique)),
        "min": float(s.min()),
        "max": float(s.max()),
        "std": float(s.std()),
        "fraction_at_extremes": float(((s == s.min()) | (s == s.max())).mean()),
        "is_degenerate": bool(len(unique) <= 2 or s.std() < 1e-6),
    }


def shuffle_null_cls(y_true, y_score, pe_efficiency, n_shuffles=2000, seed=0):
    """How likely is your CLS under random label permutation? p-value < 0.05 = signal."""
    rng = np.random.default_rng(seed)
    real = cls_metric(y_true, y_score, pe_efficiency)["cls"]
    nulls = []
    y_true = np.asarray(y_true)
    pe = np.asarray(pe_efficiency)
    perm = np.arange(len(y_true))
    for _ in range(n_shuffles):
        rng.shuffle(perm)
        nulls.append(cls_metric(y_true[perm], y_score, pe[perm])["cls"])
    nulls = np.array(nulls)
    p = float((nulls >= real).mean())
    return {
        "observed_cls": real,
        "null_mean": float(nulls.mean()),
        "null_std": float(nulls.std()),
        "null_max": float(nulls.max()),
        "p_value": p,
        "is_significant": bool(p < 0.05),
    }


def family_leakage(y_score, families, y_true):
    """Detect predictions that are family-constant — i.e. memorising family→label.

    For each family, compute the std of predictions within the family. If std is
    near zero across all families, the model has effectively memorised family ID.
    """
    families = np.asarray(families)
    out = {}
    for fam in sorted(set(families)):
        mask = families == fam
        s = np.asarray(y_score)[mask]
        out[fam] = {
            "n": int(mask.sum()),
            "active_in_fam": int(np.asarray(y_true)[mask].sum()),
            "pred_mean": float(s.mean()) if len(s) else None,
            "pred_std": float(s.std()) if len(s) else None,
        }
    stds = np.array([v["pred_std"] for v in out.values() if v["pred_std"] is not None])
    return {
        "per_family": out,
        "max_within_family_std": float(stds.max()),
        "min_within_family_std": float(stds.min()),
        "is_family_constant": bool(stds.max() < 1e-3),
    }


def class_rank_consistency(y_true, y_score, pe_efficiency):
    """Compare PR-AUC and Weighted Spearman directly.

    A model that nails PR-AUC but tanks WSpearman is a binary classifier
    that doesn't rank within the active set — i.e. won't help wet-lab triage.
    """
    from sklearn.metrics import average_precision_score
    pr_auc = float(average_precision_score(y_true, y_score))
    weights = np.asarray(pe_efficiency, dtype=float) + 0.01
    ws = max(weighted_spearman(y_score, pe_efficiency, weights), 0.0)
    gap = abs(pr_auc - ws)
    return {
        "pr_auc": pr_auc,
        "w_spearman": ws,
        "gap": gap,
        "is_classifier_only": bool(pr_auc > 0.6 and ws < 0.15),
        "is_ranker_only": bool(ws > 0.6 and pr_auc < 0.5),
    }


def run_full_audit(y_true, y_score, pe_efficiency, families):
    """Run all audit checks and return a single dict."""
    return {
        "degeneracy": degeneracy(y_score),
        "family_leakage": family_leakage(y_score, families, y_true),
        "class_rank_consistency": class_rank_consistency(y_true, y_score, pe_efficiency),
        "shuffle_null": shuffle_null_cls(y_true, y_score, pe_efficiency, n_shuffles=500),
    }
