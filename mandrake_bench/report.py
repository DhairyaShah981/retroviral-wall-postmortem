"""Markdown report renderer — match Mandrake's challenge-page table format."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import cls as cls_metric, per_family_pr_auc


def markdown(name, y_true, y_score, pe_efficiency, families, audit_results=None, baseline_cls=None):
    out = [f"# {name}\n"]
    res = cls_metric(y_true, y_score, pe_efficiency)
    out.append(f"**Pooled OOF CLS:** `{res['cls']:.4f}` (PR-AUC `{res['pr_auc']:.4f}`, WSpearman `{res['w_spearman']:.4f}`)\n")
    if baseline_cls is not None:
        delta = res["cls"] - baseline_cls
        sign = "+" if delta >= 0 else ""
        out.append(f"vs Mandrake baseline `0.318`: **{sign}{delta:.4f}**\n")
    out.append("\n## Per-family PR-AUC\n")
    pf = per_family_pr_auc(y_true, y_score, families)
    out.append("| Family | n | Active | PR-AUC |")
    out.append("|---|---|---|---|")
    for fam, v in pf.items():
        pra = f"{v['pr_auc']:.4f}" if v["pr_auc"] is not None else "N/A"
        out.append(f"| {fam} | {v['n']} | {v['active']} | {pra} |")
    if audit_results:
        out.append("\n## Audit\n")
        deg = audit_results.get("degeneracy", {})
        out.append(f"- **Score degeneracy:** {deg.get('n_unique_values','?')} unique values; std `{deg.get('std',0):.4f}`; degenerate: `{deg.get('is_degenerate',False)}`")
        cr = audit_results.get("class_rank_consistency", {})
        out.append(f"- **Class vs Rank gap:** {cr.get('gap',0):.4f} (classifier-only: `{cr.get('is_classifier_only',False)}`)")
        fl = audit_results.get("family_leakage", {})
        out.append(f"- **Family-constant predictions:** `{fl.get('is_family_constant',False)}` (min within-family std `{fl.get('min_within_family_std',0):.4f}`)")
        sn = audit_results.get("shuffle_null", {})
        out.append(f"- **Permutation p-value:** `{sn.get('p_value',1):.4f}` ({'SIGNIFICANT' if sn.get('is_significant',False) else 'NOT significant'} vs label-shuffle null; null mean `{sn.get('null_mean',0):.4f}`)")
    return "\n".join(out)
