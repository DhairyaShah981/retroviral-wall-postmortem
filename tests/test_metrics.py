"""Tests for mandrake_bench.metrics vs the official evaluator.

These tests confirm the CLS implementation matches mandrake-repo/evaluation/evaluate.py
on adversarial and corner-case inputs. Without these tests the "0.006 gap" claim
in WRITEUP.md is just a vibe — these tests make it auditable.

Run:
    cd ~/Documents/retroviral-wall-postmortem
    source .venv/bin/activate
    pip install pytest
    pytest tests/ -v
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mandrake_bench import metrics, audit

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
EVAL_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "mandrake-repo", "evaluation", "evaluate.py")


@pytest.fixture(scope="module")
def gt():
    return pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)


def _official_cls(predictions_df, tmp_path=None):
    """Run the official evaluator subprocess on a predictions DataFrame and parse CLS."""
    import subprocess, tempfile
    if tmp_path is None:
        tmp_path = tempfile.mkdtemp()
    p = os.path.join(tmp_path, "test_preds.csv")
    predictions_df.to_csv(p, index=False)
    out = subprocess.run([sys.executable, EVAL_SCRIPT, "--predictions", p],
                          capture_output=True, text=True, check=True)
    for line in out.stdout.split("\n"):
        if line.strip().startswith("CLS:"):
            return float(line.split(":")[1].strip())
    raise RuntimeError(f"Couldn't parse CLS from:\n{out.stdout}")


def test_oracle_ranker_is_1(gt):
    """Predicting pe_efficiency exactly should give CLS = 1.0."""
    res = metrics.cls(gt["active"].values, gt["pe_efficiency_pct"].values, gt["pe_efficiency_pct"].values)
    assert res["cls"] == pytest.approx(1.0, abs=1e-6)
    assert res["pr_auc"] == pytest.approx(1.0, abs=1e-6)
    assert res["w_spearman"] == pytest.approx(1.0, abs=1e-6)


def test_all_inactive_is_0(gt):
    """Predicting zeros for everything: PR-AUC = base rate, but WSpearman = 0 → CLS = 0."""
    scores = np.zeros(len(gt))
    res = metrics.cls(gt["active"].values, scores, gt["pe_efficiency_pct"].values)
    assert res["cls"] == 0.0
    assert res["w_spearman"] == 0.0


def test_random_seeded_matches_official(gt, tmp_path):
    """Random predictions should give the same CLS in our code as the official evaluator
    (modulo sklearn argsort tie-breaking < 0.01)."""
    rng = np.random.default_rng(42)
    scores = rng.uniform(size=len(gt))
    res = metrics.cls(gt["active"].values, scores, gt["pe_efficiency_pct"].values)
    pred_df = pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores})
    official = _official_cls(pred_df, tmp_path=str(tmp_path))
    assert res["cls"] == pytest.approx(official, abs=0.01), \
        f"Our CLS {res['cls']:.4f} differs from official {official:.4f} by >0.01"


def test_isotonic_stack_predictions_match_official(gt, tmp_path):
    """The actual best model's predictions must match the official evaluator."""
    pred_path = os.path.join(os.path.dirname(__file__), "..", "results",
                              "04_isotonic_stack_handonly_predictions.csv")
    if not os.path.exists(pred_path):
        pytest.skip("Run notebook 04 first to generate predictions")
    preds = pd.read_csv(pred_path)
    merged = gt.merge(preds, on="rt_name")
    res = metrics.cls(merged["active"].values, merged["predicted_score"].values,
                      merged["pe_efficiency_pct"].values)
    official = _official_cls(preds, tmp_path=str(tmp_path))
    assert res["cls"] == pytest.approx(official, abs=0.01)


def test_epsilon_makes_a_difference(gt):
    """ε=0.01 vs ε=0.1 should give materially different scores on non-trivial inputs."""
    rng = np.random.default_rng(0)
    scores = rng.uniform(size=len(gt))
    r1 = metrics.cls(gt["active"].values, scores, gt["pe_efficiency_pct"].values, epsilon=metrics.EPSILON_CODE)
    r2 = metrics.cls(gt["active"].values, scores, gt["pe_efficiency_pct"].values, epsilon=metrics.EPSILON_PAGE)
    # They should differ — confirming the page-vs-code discrepancy actually matters
    assert abs(r1["cls"] - r2["cls"]) > 0.001, \
        "ε=0.01 and ε=0.1 produce same CLS; epsilon doesn't matter (unexpected)"


def test_audit_flags_oracle_binary(gt):
    """Audit should flag a binary 0/1 oracle as classifier-only."""
    scores = gt["active"].values.astype(float)
    a = audit.run_full_audit(gt["active"].values, scores, gt["pe_efficiency_pct"].values, gt["rt_family"].values)
    assert a["degeneracy"]["is_degenerate"]
    assert a["class_rank_consistency"]["is_classifier_only"]


def test_audit_flags_family_constant(gt):
    """Audit should flag per-family-constant predictions."""
    fam_map = {"Retroviral": 1.0, "Retron": 0.5, "Group_II_Intron": 0.5,
                "LTR_Retrotransposon": 0.1, "CRISPR-associated": 0.0,
                "Other": 0.0, "Unclassified": 0.0}
    scores = np.array([fam_map[f] for f in gt["rt_family"]])
    a = audit.run_full_audit(gt["active"].values, scores, gt["pe_efficiency_pct"].values, gt["rt_family"].values)
    assert a["family_leakage"]["is_family_constant"]


def test_audit_passes_oracle_ranker(gt):
    """Audit should NOT flag a legitimate oracle ranker — no false positives."""
    scores = gt["pe_efficiency_pct"].values
    a = audit.run_full_audit(gt["active"].values, scores, gt["pe_efficiency_pct"].values, gt["rt_family"].values)
    assert not a["degeneracy"]["is_degenerate"]
    assert not a["family_leakage"]["is_family_constant"]
    assert not a["class_rank_consistency"]["is_classifier_only"]
    assert a["shuffle_null"]["is_significant"]


def test_weighted_spearman_floor(gt):
    """WSpearman is floored at 0 — negative correlations don't get negative."""
    scores = -gt["pe_efficiency_pct"].values  # perfectly anti-correlated → would give -1 unfloored
    res = metrics.cls(gt["active"].values, scores, gt["pe_efficiency_pct"].values)
    assert res["w_spearman"] == 0.0
    assert res["cls"] == 0.0


def test_lofo_loop_basic():
    """LOFO loop produces a 57-element array with one prediction per RT."""
    from mandrake_bench import cv
    X = np.random.rand(57, 10)
    y = np.random.randint(0, 2, 57)
    fams = np.array(["A"] * 20 + ["B"] * 20 + ["C"] * 17)

    def dummy(X_train, y_train, X_test, **kw):
        return np.full(len(X_test), 0.5)

    oof = cv.lofo_predict(X, y, fams, dummy)
    assert oof.shape == (57,)
    assert (oof == 0.5).all()
