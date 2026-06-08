"""Sequence-identity proximity probe (per reviewer recommendation).

The standard rigour check in CASP-style enzyme evaluation: for each held-out
RT, compute max %identity to the nearest training RT. If model error collapses
when nearest-neighbour identity rises, the model is doing nearest-neighbour
lookup, not learning biophysics.

Implementation note: real CASP uses BLAST or mmseqs2 easy-search. For this
57-RT dataset we use Biopython pairwise2 global alignment with BLOSUM62 — slow
but exact and dependency-light (no mmseqs2 install required). Cached after
first run.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from Bio import pairwise2
from Bio.Align import substitution_matrices

BLOSUM62 = substitution_matrices.load("BLOSUM62")


def percent_identity(seq_a: str, seq_b: str) -> float:
    """Global alignment percent identity using BLOSUM62. Returns 0-100."""
    # Use a fast affine-gap global alignment
    aln = pairwise2.align.globalds(seq_a, seq_b, BLOSUM62, -10, -0.5, one_alignment_only=True)
    if not aln:
        return 0.0
    a, b, score, start, end = aln[0]
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    aligned_len = sum(1 for x, y in zip(a, b) if x != "-" and y != "-")
    if aligned_len == 0:
        return 0.0
    return 100.0 * matches / aligned_len


def pairwise_identity_matrix(sequences: dict, cache_path: str | None = None) -> pd.DataFrame:
    """Compute NxN percent-identity matrix. Cached if cache_path given."""
    names = list(sequences.keys())
    n = len(names)
    if cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, index_col=0)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            if i == j:
                M[i, j] = 100.0
            else:
                M[i, j] = M[j, i] = percent_identity(sequences[names[i]], sequences[names[j]])
    df = pd.DataFrame(M, index=names, columns=names)
    if cache_path:
        df.to_csv(cache_path)
    return df


def nearest_train_identity(identity_matrix: pd.DataFrame, families: dict) -> pd.DataFrame:
    """For each RT, compute its max %identity to any RT in a *different* family
    (i.e. simulating the LOFO held-out setting). Returns DataFrame with columns:
    rt_name, family, max_identity_to_other_family.
    """
    rows = []
    rt_names = list(identity_matrix.index)
    for rt in rt_names:
        fam = families[rt]
        # Mask out same-family RTs (LOFO would not see them in train)
        other_fam = [n for n in rt_names if families[n] != fam and n != rt]
        if not other_fam:
            rows.append({"rt_name": rt, "family": fam, "max_identity_to_other_family": 0.0})
        else:
            mx = float(identity_matrix.loc[rt, other_fam].max())
            rows.append({"rt_name": rt, "family": fam, "max_identity_to_other_family": mx})
    return pd.DataFrame(rows)


def identity_vs_error_diagnostic(predictions: pd.DataFrame, identity_df: pd.DataFrame,
                                  pe_efficiency: dict, active: dict):
    """Return a DataFrame combining each RT's prediction, true activity,
    max train-family identity, and prediction error. Sort by identity DESC to
    eyeball whether high-identity RTs have lower error (= nearest-neighbour lookup).
    """
    merged = predictions.merge(identity_df, on="rt_name", how="left")
    merged["true_pe"] = merged["rt_name"].map(pe_efficiency)
    merged["active"] = merged["rt_name"].map(active)
    # Compute a unit-scaled prediction error
    s = merged["predicted_score"]
    s_norm = (s - s.min()) / max((s.max() - s.min()), 1e-9)
    pe_norm = merged["true_pe"] / max(merged["true_pe"].max(), 1e-9)
    merged["pred_error"] = (s_norm - pe_norm).abs()
    return merged.sort_values("max_identity_to_other_family", ascending=False)
