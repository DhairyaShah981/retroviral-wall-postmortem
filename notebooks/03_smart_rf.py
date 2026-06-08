"""03 — Smart RF: ablations + biophysics+ESM combined + calibration.

LightGBM overfits at N=57. RF wins. Goal here: push RF beyond 0.534 by
combining handcrafted biophysics with family-confound-projected ESM-2,
and stack regressor + classifier outputs intelligently.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import metrics, cv, audit, report

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

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

FOLDSEEK_COLS = [c for c in feats.columns if "foldseek" in c.lower()]
DROP_COLS = {"rt_name", "rt_family", "yxdd_seq"}
ALL_NUMERIC = [c for c in feats.columns if c not in DROP_COLS]
NO_FOLDSEEK = [c for c in ALL_NUMERIC if c not in FOLDSEEK_COLS]

def to_X(cols): return feats[cols].apply(pd.to_numeric, errors="coerce").values
X_all = to_X(ALL_NUMERIC)
X_clean = to_X(NO_FOLDSEEK)


def rf_clf_fp(X_train, y_train, X_test, **kw):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                       max_features="sqrt", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def rf_reg_fp(X_train, y_train, X_test, pe_train=None, **kw):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                      max_features="sqrt", random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_train, pe_train)
    return pipe.predict(X_test)


def make_esm_pca_runner(k_remove=3):
    """Remove top-k PCA components (likely family-confound) before classify."""
    def runner(X_train, y_train, X_test, **kw):
        scaler = StandardScaler().fit(X_train)
        Xtr = scaler.transform(X_train)
        Xte = scaler.transform(X_test)
        pca = PCA(n_components=min(k_remove, Xtr.shape[0]-1), random_state=42).fit(Xtr)
        P = np.eye(Xtr.shape[1]) - pca.components_.T @ pca.components_
        Xtr2 = Xtr @ P
        Xte2 = Xte @ P
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=0.5, max_iter=2000, random_state=42).fit(Xtr2, y_train)
        return clf.predict_proba(Xte2)[:, 1]
    return runner


def combo_rf_runner(X_train, y_train, X_test, pe_train=None, **kw):
    """RF classifier + RF regressor (on PE eff) → averaged after rank-normalizing.
    This jointly optimizes PR-AUC (clf) and WSpearman (reg)."""
    # Classifier
    p_clf = rf_clf_fp(X_train, y_train, X_test)
    # Regressor
    p_reg = rf_reg_fp(X_train, y_train, X_test, pe_train=pe_train)
    # Combine: geometric mean of clf prob + sigmoid-mapped reg score
    # Keep both contributions roughly equal in scale
    p_reg_scaled = (p_reg - p_reg.min()) / (max(p_reg.max() - p_reg.min(), 1e-9))
    return np.sqrt(np.clip(p_clf, 1e-6, 1.0) * np.clip(p_reg_scaled, 1e-6, 1.0))


def combo_with_esm(X_train, y_train, X_test, X_train_esm=None, X_test_esm=None, pe_train=None, **kw):
    """Combo biophysics (RF) + ESM-PCA-clean (LR), then ensemble."""
    p1 = combo_rf_runner(X_train, y_train, X_test, pe_train=pe_train)
    p2 = make_esm_pca_runner(k_remove=3)(X_train_esm, y_train, X_test_esm)
    return 0.7 * p1 + 0.3 * p2


# --- Runs
runs = {}
runs["rf_clf_allfeat"] = cv.lofo_predict(X_all, y, fams, rf_clf_fp)
runs["rf_reg_allfeat"] = cv.lofo_predict(X_all, y, fams, rf_reg_fp, pe_efficiency=pe)
runs["rf_clf_nofoldseek"] = cv.lofo_predict(X_clean, y, fams, rf_clf_fp)
runs["rf_reg_nofoldseek"] = cv.lofo_predict(X_clean, y, fams, rf_reg_fp, pe_efficiency=pe)
runs["rf_combo_allfeat"] = cv.lofo_predict(X_all, y, fams, combo_rf_runner, pe_efficiency=pe)
runs["rf_combo_nofoldseek"] = cv.lofo_predict(X_clean, y, fams, combo_rf_runner, pe_efficiency=pe)

# ESM alone with PCA cleaning
for k in [1, 3, 5, 7]:
    runs[f"esm_pca{k}_logreg"] = cv.lofo_predict(esm_X, y, fams, make_esm_pca_runner(k_remove=k))

# Combo: RF on biophysics + ESM-PCA-cleaned
def run_combo_with_esm():
    families_arr = np.asarray(fams)
    oof = np.zeros(len(y), dtype=float)
    for fam in sorted(set(families_arr)):
        test_mask = families_arr == fam
        train_mask = ~test_mask
        scores = combo_with_esm(
            X_all[train_mask], y[train_mask], X_all[test_mask],
            X_train_esm=esm_X[train_mask], X_test_esm=esm_X[test_mask],
            pe_train=pe[train_mask],
        )
        oof[test_mask] = scores
    return oof
runs["combo_RFbio_ESMpca"] = run_combo_with_esm()

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
    with open(os.path.join(RESULTS, f"03_{name}.md"), "w") as f:
        f.write(md)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"03_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("\n=== Smart RF + ESM-PCA summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "03_summary.csv"), index=False)
