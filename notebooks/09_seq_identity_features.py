"""09 — Add sequence-identity-to-known-actives as features.

The identity matrix from notebook 06 is N x N pairwise %identity. We can extract
per-RT features:
  - max identity to any MMLV-family (Retroviral) active RT
  - max identity to any Retron active RT
  - mean identity to top-3 nearest training RTs
  - identity to MMLV specifically (the gold standard)

These are alignment-derived signals the existing handcrafted features don't include.
Critical: they encode 'similarity to a known active' without encoding family ID
directly. In LOFO, when a family is held out, the model still sees
'how similar are you to the actives we *do* know about' across other families.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import metrics, cv, audit, report

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
feats = pd.read_csv(os.path.join(DATA, "handcrafted_features.csv")).set_index("rt_name").loc[gt["rt_name"]].reset_index()
M = pd.read_csv(os.path.join(RESULTS, "identity_matrix.csv"), index_col=0)
M = M.loc[gt["rt_name"], gt["rt_name"]].values  # align with gt order

y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values
DROP_COLS = {"rt_name", "rt_family", "yxdd_seq"}
hand_cols = [c for c in feats.columns if c not in DROP_COLS]
X_hand = feats[hand_cols].apply(pd.to_numeric, errors="coerce").values
rt_names = gt["rt_name"].tolist()

# Find MMLV index (gold-standard active)
mmlv_idx = rt_names.index("MMLV-RT")
print(f"MMLV-RT is RT #{mmlv_idx}, pe={pe[mmlv_idx]}%, active={y[mmlv_idx]}")


def build_identity_features_for_lofo(test_idx_list, train_idx_list):
    """For each test RT, compute identity-based features using only train RTs
    (matching the LOFO setting). Returns (test_features, train_features).
    """
    def feats_for(idxs, train_idxs):
        rows = []
        train_active_mask = y[train_idxs] == 1
        train_active_idxs = np.array(train_idxs)[train_active_mask]
        train_inactive_idxs = np.array(train_idxs)[~train_active_mask]
        for i in idxs:
            id_to_train = M[i, train_idxs]
            id_to_train_actives = M[i, train_active_idxs] if len(train_active_idxs) > 0 else np.array([0])
            id_to_train_inactives = M[i, train_inactive_idxs] if len(train_inactive_idxs) > 0 else np.array([0])
            row = {
                "max_id_to_any_train": id_to_train.max() if len(id_to_train) else 0,
                "max_id_to_train_active": id_to_train_actives.max() if len(id_to_train_actives) else 0,
                "max_id_to_train_inactive": id_to_train_inactives.max() if len(id_to_train_inactives) else 0,
                "mean_top3_id_to_train_active": np.sort(id_to_train_actives)[-3:].mean() if len(id_to_train_actives) else 0,
                "id_to_mmlv": M[i, mmlv_idx] if mmlv_idx in train_idxs else 0.0,  # only if MMLV in train
                "active_minus_inactive_max_id":
                    (id_to_train_actives.max() if len(id_to_train_actives) else 0) -
                    (id_to_train_inactives.max() if len(id_to_train_inactives) else 0),
            }
            rows.append(row)
        return pd.DataFrame(rows).values
    return feats_for(test_idx_list, train_idx_list), feats_for(train_idx_list, train_idx_list)


def isotonic_stack_with_id(X_train, y_train, X_test, pe_train=None, **kw):
    clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                                   min_samples_leaf=2, random_state=42, n_jobs=-1))])
    reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                                  min_samples_leaf=2, random_state=42, n_jobs=-1))])
    clf.fit(X_train, y_train); reg.fit(X_train, pe_train)
    iso = IsotonicRegression(out_of_bounds="clip").fit(reg.predict(X_train), y_train)
    return np.sqrt(np.clip(clf.predict_proba(X_test)[:, 1], 1e-6, 1.0) *
                    np.clip(iso.transform(reg.predict(X_test)), 1e-6, 1.0))


def run_lofo_with_id_feats(X_base):
    """Custom LOFO loop that adds per-fold identity features."""
    families_arr = np.asarray(fams)
    oof = np.zeros(len(y), dtype=float)
    for fam in sorted(set(families_arr)):
        test_mask = families_arr == fam
        train_mask = ~test_mask
        test_idxs = np.where(test_mask)[0].tolist()
        train_idxs = np.where(train_mask)[0].tolist()
        Xtest_id, Xtrain_id = build_identity_features_for_lofo(test_idxs, train_idxs)
        # Concat identity features to base features
        Xtr = np.concatenate([X_base[train_mask], Xtrain_id], axis=1)
        Xte = np.concatenate([X_base[test_mask], Xtest_id], axis=1)
        oof[test_mask] = isotonic_stack_with_id(Xtr, y[train_mask], Xte, pe_train=pe[train_mask])
    return oof


# Run
runs = {}
runs["isotonic_handonly_plus_id"] = run_lofo_with_id_feats(X_hand)

# Also: identity features ALONE
runs["isotonic_id_only"] = run_lofo_with_id_feats(np.empty((57, 0)))  # empty base

summary = []
for name, scores in runs.items():
    res = metrics.cls(y, scores, pe)
    pf = metrics.per_family_pr_auc(y, scores, fams)
    rec = {"model": name, "cls": res["cls"], "pr_auc": res["pr_auc"], "w_spearman": res["w_spearman"]}
    for fam in ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]:
        rec[f"PRAUC_{fam}"] = pf.get(fam, {}).get("pr_auc")
    summary.append(rec)
    a = audit.run_full_audit(y, scores, pe, fams)
    md = report.markdown(name, y, scores, pe, fams, audit_results=a, baseline_cls=0.318)
    with open(os.path.join(RESULTS, f"09_{name}.md"), "w") as f:
        f.write(md)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"09_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("=== Day 3 round 3: identity-derived features ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "09_summary.csv"), index=False)
