"""06 — Sequence-identity proximity probe.

Per reviewer: for each held-out RT, plot prediction error vs max %identity to
any training-family RT. If error collapses below 40-50% identity, models are
doing nearest-neighbour lookup, not learning biophysics. Standard CASP rigour
check.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import identity_probe

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
sequences = dict(zip(gt["rt_name"], gt["sequence"]))
families = dict(zip(gt["rt_name"], gt["rt_family"]))
pe_eff = dict(zip(gt["rt_name"], gt["pe_efficiency_pct"]))
active = dict(zip(gt["rt_name"], gt["active"]))

# Cache the identity matrix (57*56/2 = 1596 pairwise alignments ~ 1-2 min)
cache = os.path.join(RESULTS, "identity_matrix.csv")
print("Building pairwise identity matrix (cached after first run)...")
M = identity_probe.pairwise_identity_matrix(sequences, cache_path=cache)
print(f"  identity matrix shape: {M.shape}")
print(f"  cached at: {cache}\n")

# Nearest cross-family identity per RT (simulates LOFO train pool)
nearest = identity_probe.nearest_train_identity(M, families)
print("=== Cross-family nearest neighbour identity (top 10 highest) ===")
print(nearest.sort_values("max_identity_to_other_family", ascending=False).head(10).to_string(index=False))
print()
print("=== Bottom 10 (most isolated RTs in their family) ===")
print(nearest.sort_values("max_identity_to_other_family", ascending=True).head(10).to_string(index=False))
print()

# Per-family identity stats
print("=== Per-family max-other-family-identity stats ===")
fam_stats = nearest.groupby("family")["max_identity_to_other_family"].agg(["mean", "min", "max", "count"]).round(2)
print(fam_stats.to_string())
print()

# Cross with our best predictions
preds = pd.read_csv(os.path.join(RESULTS, "04_isotonic_stack_handonly_predictions.csv"))
merged = identity_probe.identity_vs_error_diagnostic(preds, nearest, pe_eff, active)
print("=== Prediction error vs nearest cross-family identity (top 10 by identity) ===")
print(merged[["rt_name", "family", "true_pe", "active", "max_identity_to_other_family",
              "predicted_score", "pred_error"]].head(10).to_string(index=False))

# Correlation
from scipy.stats import spearmanr, pearsonr
r_pe, _ = spearmanr(merged["max_identity_to_other_family"], merged["pred_error"])
print(f"\nSpearman r(identity, pred_error): {r_pe:+.4f}")
print(f"  (negative = high-identity RTs have lower error = model is doing nearest-neighbour lookup)")

# Save
nearest.to_csv(os.path.join(RESULTS, "06_nearest_train_identity.csv"), index=False)
merged.to_csv(os.path.join(RESULTS, "06_identity_vs_error.csv"), index=False)
print(f"\nSaved to {RESULTS}/06_*.csv")
