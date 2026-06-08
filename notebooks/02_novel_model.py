"""02 — Novel angle: family-leakage-aware features + LightGBM + ensemble.

Hypothesis 1: foldseek_TM_{family} columns are explicit family fingerprints.
              Drop them and we should generalize better on held-out families,
              even if pooled CLS doesn't move.

Hypothesis 2: ESM-2 embeddings encode family. Projecting them onto the
              subspace orthogonal to family-mean directions removes that leak
              while preserving sequence-level structure.

Hypothesis 3: a small ensemble of (LGBM-classifier on biophysics) +
              (LGBM-regressor on PE efficiency) gives both PR-AUC and
              WSpearman gains, since CLS is harmonic-meaned.

Run:
    python notebooks/02_novel_model.py
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import metrics, cv, audit, report
import lightgbm as lgb

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# --- Load
gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
feats = pd.read_csv(os.path.join(DATA, "handcrafted_features.csv")).set_index("rt_name").loc[gt["rt_name"]].reset_index()
esm_data = np.load(os.path.join(DATA, "esm2_embeddings.npz"), allow_pickle=True)
esm_names = esm_data["names"].tolist()
esm_idx = [esm_names.index(n) for n in gt["rt_name"]]
esm_X = esm_data["embeddings"][esm_idx]

y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values

# Feature groups
FOLDSEEK_COLS = [c for c in feats.columns if "foldseek" in c.lower()]
DROP_COLS = {"rt_name", "rt_family", "yxdd_seq"}
ALL_NUMERIC = [c for c in feats.columns if c not in DROP_COLS]
NO_FOLDSEEK = [c for c in ALL_NUMERIC if c not in FOLDSEEK_COLS]

def to_X(cols):
    return feats[cols].apply(pd.to_numeric, errors="coerce").values

X_all = to_X(ALL_NUMERIC)
X_clean = to_X(NO_FOLDSEEK)  # family-leakage-aware

print(f"All features: {X_all.shape}; foldseek-dropped: {X_clean.shape}")
print(f"Foldseek (dropped from clean): {FOLDSEEK_COLS}\n")


def lgbm_clf(X_train, y_train, X_test, **kw):
    model = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_data_in_leaf=3, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def lgbm_reg(X_train, y_train, X_test, pe_train=None, **kw):
    if pe_train is None:
        raise ValueError("pass pe_train")
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
        min_data_in_leaf=3, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    # Train on PE efficiency directly (zero for inactive, 0.5-41% for active)
    model.fit(X_train, pe_train)
    return model.predict(X_test)


def family_orthogonal_esm(X_train_emb, train_families):
    """Compute family-mean vectors from training; return a projector matrix that
    removes those directions from any input. Used to de-bias ESM-2."""
    means = []
    for fam in sorted(set(train_families)):
        means.append(X_train_emb[train_families == fam].mean(axis=0))
    M = np.array(means)
    # SVD to orthonormalize the family-mean subspace
    u, s, vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    # Keep top-k singular vectors (one per family)
    keep = (s > 1e-6).sum()
    fam_basis = vt[:keep]  # (k, 1280) — basis of family-confound subspace
    # Projector onto orthogonal complement: I - fam_basis^T @ fam_basis
    P = np.eye(X_train_emb.shape[1]) - fam_basis.T @ fam_basis
    return P


def esm_orth_logreg(X_train, y_train, X_test, **kw):
    """ESM with family-orthogonal projection learned from train."""
    # Need train family info — passed via closure (see runs below)
    raise NotImplementedError("use the closure version")


def make_esm_orth_runner(train_fam_lookup_idx):
    """Returns a fit_predict_fn that knows the per-fold train families.
    We supply the train family list per call via cv's logic — but cv doesn't
    pass it, so we recompute here from train indices.
    """
    def runner(X_train, y_train, X_test, **kw):
        # In LOFO, X_train is all rows where families != held-out. We don't
        # know train_fams in the closure directly, so reconstruct: cv's design
        # is to give us X already split. We can use kw if extended; for now,
        # learn family-mean from clustering on training embeddings (a proxy).
        # Simpler: just remove top-k principal components — those soak up
        # the largest-variance directions, which empirically map to family.
        from sklearn.decomposition import PCA
        pca = PCA(n_components=6, random_state=42)  # 6 families in train (one held out)
        pca.fit(X_train)
        P = np.eye(X_train.shape[1]) - pca.components_.T @ pca.components_
        Xtr_orth = X_train @ P
        Xte_orth = X_test @ P
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        clf = Pipeline([("s", StandardScaler()), ("lr", LogisticRegression(C=0.3, max_iter=2000, random_state=42))])
        clf.fit(Xtr_orth, y_train)
        return clf.predict_proba(Xte_orth)[:, 1]
    return runner


# --- Runs
runs = {}

# Baseline anchors
runs["lgbm_clf_allfeat"] = cv.lofo_predict(X_all, y, fams, lgbm_clf)
runs["lgbm_reg_allfeat"] = cv.lofo_predict(X_all, y, fams, lgbm_reg, pe_efficiency=pe)

# Novel: drop family-leak features
runs["lgbm_clf_nofoldseek"] = cv.lofo_predict(X_clean, y, fams, lgbm_clf)
runs["lgbm_reg_nofoldseek"] = cv.lofo_predict(X_clean, y, fams, lgbm_reg, pe_efficiency=pe)

# Novel: ESM with PCA-orthogonal-projection (family-leak mitigated)
runs["esm_orth_logreg"] = cv.lofo_predict(esm_X, y, fams, make_esm_orth_runner(None))

# Ensemble: harmonic-mean alignment — sum of clf rank + reg rank
def rank01(x):
    r = pd.Series(x).rank(method="average").values
    return (r - 1) / (len(r) - 1)


runs["ensemble_clf+reg_nofoldseek"] = rank01(runs["lgbm_clf_nofoldseek"]) + rank01(runs["lgbm_reg_nofoldseek"])

# Triple ensemble
runs["ensemble_triple"] = (
    rank01(runs["lgbm_clf_nofoldseek"]) +
    rank01(runs["lgbm_reg_nofoldseek"]) +
    rank01(runs["esm_orth_logreg"])
)

# --- Score
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
    with open(os.path.join(RESULTS, f"02_{name}.md"), "w") as f:
        f.write(md)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"02_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("\n=== Novel-model summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "02_summary.csv"), index=False)
