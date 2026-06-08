"""07 — Day 3: push past CLS 0.55.

Things to try, ranked by expected payoff at N=57:
  A) Family-balanced bootstrap: oversample minority families when fitting RF.
     Direct attack on the "RT family confounded with activity" structure.
  B) Stacked meta-learner: LogisticRegression on (RF-clf-prob, RF-reg-pred,
     ESM-PCA-clean-prob) features. Three-channel ensemble.
  C) Quantile RF regression: predict the median PE efficiency via quantile RF
     and use it as the ranking score. Robust to outliers (MMLV at 41% etc.).
  D) Sample weighting by inverse family size: each training sample weighted by
     1 / n_in_family. The standard small-N debiasing trick from computational
     enzymology (cited by reviewer).
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
esm_data = np.load(os.path.join(DATA, "esm2_embeddings.npz"), allow_pickle=True)
esm_X = esm_data["embeddings"][[esm_data["names"].tolist().index(n) for n in gt["rt_name"]]]

y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values

DROP_COLS = {"rt_name", "rt_family", "yxdd_seq"}
hand_cols = [c for c in feats.columns if c not in DROP_COLS]
X = feats[hand_cols].apply(pd.to_numeric, errors="coerce").values
print(f"Data: {X.shape}, ESM: {esm_X.shape}\n")


# --- A) Family-balanced bootstrap RF
def family_balanced_rf(X_train, y_train, X_test, train_fams=None, pe_train=None, **kw):
    """Oversample to balance families before fitting RF."""
    # cv.lofo_predict doesn't pass train_fams; we need a closure
    raise NotImplementedError("use the closure variant")


def make_fambal_runner():
    """Closure that recomputes train families from a global cache."""
    fam_cache = {}
    def runner(X_train, y_train, X_test, **kw):
        # We don't have train family info per-call from cv.lofo_predict.
        # Workaround: detect via length matching against the global `fams`.
        # The training mask is len(X_train) rows; find which slice of `fams` matches.
        # Simpler: identify the held-out family by elimination.
        n_train = len(X_train)
        for fam in sorted(set(fams)):
            mask = fams != fam
            if mask.sum() == n_train:
                train_fams = fams[mask]
                break
        else:
            train_fams = fams[:n_train]  # fallback

        # Compute per-sample weight = 1 / n_in_family
        from collections import Counter
        c = Counter(train_fams)
        w = np.array([1.0 / c[f] for f in train_fams])
        w = w / w.mean()  # normalize to unit mean

        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                           min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(X_train, y_train, rf__sample_weight=w)
        return pipe.predict_proba(X_test)[:, 1]
    return runner


# --- B) Stacked meta-learner
def stacked_meta_runner(X_train, y_train, X_test, pe_train=None, X_train_esm=None, X_test_esm=None, **kw):
    """Three channels meta-stacked with LogisticRegression on out-of-fold-like features.
    Uses k-fold within the train set to generate level-1 predictions for the LR stacker.
    """
    from sklearn.model_selection import KFold
    n = len(X_train)
    z = np.zeros((n, 3), dtype=float)
    kf = KFold(n_splits=min(5, n // 2), shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(X_train):
        rf_c = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))])
        rf_r = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1))])
        rf_c.fit(X_train[tr_idx], y_train[tr_idx])
        rf_r.fit(X_train[tr_idx], pe_train[tr_idx])
        z[va_idx, 0] = rf_c.predict_proba(X_train[va_idx])[:, 1]
        z[va_idx, 1] = rf_r.predict(X_train[va_idx])
        # ESM logistic regression as third channel
        if X_train_esm is not None:
            esm_lr = Pipeline([("s", StandardScaler()),
                               ("lr", LogisticRegression(C=0.3, max_iter=2000, random_state=42))])
            esm_lr.fit(X_train_esm[tr_idx], y_train[tr_idx])
            z[va_idx, 2] = esm_lr.predict_proba(X_train_esm[va_idx])[:, 1]

    # Train final RF + reg on full train
    rf_c_full = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))]).fit(X_train, y_train)
    rf_r_full = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))]).fit(X_train, pe_train)
    z_test = np.zeros((len(X_test), 3), dtype=float)
    z_test[:, 0] = rf_c_full.predict_proba(X_test)[:, 1]
    z_test[:, 1] = rf_r_full.predict(X_test)
    if X_train_esm is not None:
        esm_full = Pipeline([("s", StandardScaler()),
                             ("lr", LogisticRegression(C=0.3, max_iter=2000, random_state=42))]).fit(X_train_esm, y_train)
        z_test[:, 2] = esm_full.predict_proba(X_test_esm)[:, 1]

    # Train LR meta-stacker on z, predict on z_test
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=42).fit(z, y_train)
    return meta.predict_proba(z_test)[:, 1]


def run_stacked_with_esm():
    """Custom LOFO loop that passes ESM features in."""
    families_arr = np.asarray(fams)
    oof = np.zeros(len(y), dtype=float)
    for fam in sorted(set(families_arr)):
        test_mask = families_arr == fam
        train_mask = ~test_mask
        oof[test_mask] = stacked_meta_runner(
            X[train_mask], y[train_mask], X[test_mask],
            pe_train=pe[train_mask],
            X_train_esm=esm_X[train_mask], X_test_esm=esm_X[test_mask],
        )
    return oof


# --- C) Isotonic-stack with family-balanced sample weights
def make_isofambal_runner():
    def runner(X_train, y_train, X_test, pe_train=None, **kw):
        n_train = len(X_train)
        for fam in sorted(set(fams)):
            if (fams != fam).sum() == n_train:
                train_fams = fams[fams != fam]
                break
        from collections import Counter
        c = Counter(train_fams)
        w = np.array([1.0 / c[f] for f in train_fams])
        w = w / w.mean()

        pipe_clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                                            min_samples_leaf=2, random_state=42, n_jobs=-1))])
        pipe_reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("rf", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                                           min_samples_leaf=2, random_state=42, n_jobs=-1))])
        pipe_clf.fit(X_train, y_train, rf__sample_weight=w)
        pipe_reg.fit(X_train, pe_train, rf__sample_weight=w)

        # Isotonic calibrate reg → indicator
        iso = IsotonicRegression(out_of_bounds="clip").fit(pipe_reg.predict(X_train), y_train)
        clf_test = pipe_clf.predict_proba(X_test)[:, 1]
        reg_test = iso.transform(pipe_reg.predict(X_test))
        return np.sqrt(np.clip(clf_test, 1e-6, 1.0) * np.clip(reg_test, 1e-6, 1.0))
    return runner


# --- Runs
runs = {}
runs["fambal_rf_clf"] = cv.lofo_predict(X, y, fams, make_fambal_runner())
runs["isofambal_stack"] = cv.lofo_predict(X, y, fams, make_isofambal_runner(), pe_efficiency=pe)
runs["stacked_meta"] = run_stacked_with_esm()

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
    with open(os.path.join(RESULTS, f"07_{name}.md"), "w") as f:
        f.write(md)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"07_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("=== Day 3 push-past-0.55 summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "07_summary.csv"), index=False)
