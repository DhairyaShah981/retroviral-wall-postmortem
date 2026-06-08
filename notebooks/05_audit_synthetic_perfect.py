"""05 — Audit harness vs synthetic 'tied at 1.0' submissions.

The Kaggle public leaderboard has 6 entries tied at exactly 1.00000. On a
57-sample LOFO problem with the page-spec CLS, that's not a winning signal —
it's a methodology-tell. This notebook constructs three synthetic submissions
that all hit ~1.0 on a simpler metric, then shows the audit harness flags
each one's specific failure mode that Mandrake would care about.

Three synthetic attacks on the metric:

  A) "Oracle binary"  : predict 1 for every active, 0 for every inactive.
                        Achieves PR-AUC 1.0 trivially, but WSpearman ~0
                        within the active set → CLS collapses (audit catches
                        via class_rank_consistency).

  B) "Family-constant": predict 1.0 for all Retroviral, 0.5 for Retron+G2I,
                        0.0 for others. Memorizes family, achieves high
                        PR-AUC on the public score, but family_leakage flags
                        within-family std = 0.

  C) "Oracle ranker"  : predict pe_efficiency_pct as the score (perfect ranking).
                        CLS = 1.0 exactly. This is the upper bound — what a
                        true oracle would produce. shuffle_null finds it
                        significant; nothing wrong with it methodologically.
                        Demonstrates the audit's *acceptance* case.

If the audit cleanly distinguishes A and B as flagged and C as legitimate,
the harness is doing its job.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import metrics, audit, report

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values

submissions = {}

# A) Oracle binary
submissions["A_oracle_binary"] = y.astype(float)

# B) Family-constant
fam_score_map = {
    "Retroviral": 1.0,
    "Retron": 0.5,
    "Group_II_Intron": 0.5,
    "LTR_Retrotransposon": 0.1,
    "CRISPR-associated": 0.0,
    "Other": 0.0,
    "Unclassified": 0.0,
}
submissions["B_family_constant"] = np.array([fam_score_map[f] for f in fams])

# C) Oracle ranker
submissions["C_oracle_ranker"] = pe.copy()

# --- Score + audit
summary = []
print("=== Audit synthetic submissions ===\n")
for name, scores in submissions.items():
    m = metrics.cls(y, scores, pe)
    a = audit.run_full_audit(y, scores, pe, fams)
    print(f"--- {name} ---")
    print(f"  CLS:           {m['cls']:.4f}  (PR-AUC {m['pr_auc']:.4f}, WSpearman {m['w_spearman']:.4f})")
    print(f"  Degenerate scores:        {a['degeneracy']['is_degenerate']} ({a['degeneracy']['n_unique_values']} unique)")
    print(f"  Family-constant:          {a['family_leakage']['is_family_constant']} (min within-fam std {a['family_leakage']['min_within_family_std']:.4f})")
    print(f"  Classifier-only:          {a['class_rank_consistency']['is_classifier_only']} (gap {a['class_rank_consistency']['gap']:.4f})")
    print(f"  Shuffle-null p-value:     {a['shuffle_null']['p_value']:.4f}  ({'SIGNIFICANT' if a['shuffle_null']['is_significant'] else 'NOT significant'})")
    print()
    md = report.markdown(name, y, scores, pe, fams, audit_results=a, baseline_cls=0.318)
    with open(os.path.join(RESULTS, f"05_{name}.md"), "w") as f:
        f.write(md)
    summary.append({
        "submission": name,
        "cls": m["cls"],
        "pr_auc": m["pr_auc"],
        "w_spearman": m["w_spearman"],
        "is_degenerate": a["degeneracy"]["is_degenerate"],
        "is_family_constant": a["family_leakage"]["is_family_constant"],
        "is_classifier_only": a["class_rank_consistency"]["is_classifier_only"],
        "shuffle_p_value": a["shuffle_null"]["p_value"],
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "05_audit_synthetic_summary.csv"), index=False)

# Quick narrative interpretation
print("\n=== Interpretation ===")
print("A (oracle binary):     CLS collapses despite PR-AUC=1.0 — caught by classifier-only flag.")
print("B (family-constant):   family_leakage flags within-family std = 0 — perfect detection.")
print("C (oracle ranker):     CLS = 1.0 legitimately; passes all audit checks — the harness")
print("                       does not false-positive on real signal.")
