"""01 — Baselines: reproduce Mandrake's CLS 0.318 reference + the family-leakage canary.

Run:
    cd ~/Documents/retroviral-wall-postmortem && source .venv/bin/activate
    python notebooks/01_baselines.py
"""
from __future__ import annotations
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mandrake_bench import metrics, cv, baselines, audit, report

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# --- Load data
gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv"))
feats = pd.read_csv(os.path.join(DATA, "handcrafted_features.csv"))
esm = np.load(os.path.join(DATA, "esm2_embeddings.npz"), allow_pickle=True)
esm_names = esm["names"].tolist()
esm_X = esm["embeddings"]  # (57, 1280)

# Align ordering: use gt order as canonical
gt = gt.sort_values("rt_name").reset_index(drop=True)
feats = feats.set_index("rt_name").loc[gt["rt_name"]].reset_index()
esm_idx = [esm_names.index(n) for n in gt["rt_name"]]
esm_X = esm_X[esm_idx]

y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values

# Handcrafted features: drop non-numeric metadata, keep numeric
drop_cols = {"rt_name", "rt_family", "yxdd_motif_seq"}  # motif_seq is a string
feat_cols = [c for c in feats.columns if c not in drop_cols]
# Coerce remaining to numeric (some thermostability class is encoded; coerce)
X_hand = feats[feat_cols].apply(pd.to_numeric, errors="coerce").values

print(f"Loaded: {len(gt)} RTs · {len(feat_cols)} handcrafted features · ESM-2 shape {esm_X.shape}")
print(f"Families: {sorted(set(fams))}")
print(f"Active rate: {y.mean():.3f} ({int(y.sum())}/{len(y)})\n")

# --- Run baselines
runs = {}

# 1. predict-all-inactive
runs["all_inactive"] = cv.lofo_predict(X_hand, y, fams, baselines.predict_all_inactive)
# 2. random
runs["random"] = cv.lofo_predict(X_hand, y, fams, baselines.predict_random)
# 3. RF classifier on handcrafted (Mandrake's reference)
runs["rf_handcrafted_clf"] = cv.lofo_predict(X_hand, y, fams, baselines.predict_rf_classifier)
# 4. RF regressor on handcrafted (regress pe_efficiency directly)
runs["rf_handcrafted_reg"] = cv.lofo_predict(X_hand, y, fams, baselines.predict_rf_regressor, pe_efficiency=pe)
# 5. ESM-only logreg (the family-leakage canary)
runs["esm_logreg"] = cv.lofo_predict(esm_X, y, fams, baselines.predict_esm_logreg)

# --- Score + audit + report
summary = []
for name, scores in runs.items():
    res = metrics.cls(y, scores, pe)
    summary.append({"model": name, "cls": res["cls"], "pr_auc": res["pr_auc"], "w_spearman": res["w_spearman"]})
    audit_res = audit.run_full_audit(y, scores, pe, fams)
    md = report.markdown(name, y, scores, pe, fams, audit_results=audit_res, baseline_cls=0.318)
    with open(os.path.join(RESULTS, f"01_{name}.md"), "w") as f:
        f.write(md)
    # save predictions
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"01_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("\n=== Baseline summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "01_summary.csv"), index=False)
print(f"\nPer-baseline markdown reports written to {RESULTS}/01_*.md")
