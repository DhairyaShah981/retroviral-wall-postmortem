"""08 — Day 3 part 2: more variants. Try GP regression + log-PE + multi-seed RF + concat with ESM."""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
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
X_hand = feats[hand_cols].apply(pd.to_numeric, errors="coerce").values
# Concat: handcrafted + ESM (no orthogonal projection)
X_concat = np.concatenate([X_hand, esm_X], axis=1)
log_pe = np.log1p(pe)


def rf_multi_seed_stack(X_train, y_train, X_test, pe_train=None, seeds=(42, 7, 13, 99, 256), **kw):
    """Average isotonic-calibrated RF stack predictions across multiple random seeds.
    Reduces variance on small N."""
    test_preds = []
    for seed in seeds:
        clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                                       min_samples_leaf=2, random_state=seed, n_jobs=-1))])
        reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                        ("rf", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                                      min_samples_leaf=2, random_state=seed, n_jobs=-1))])
        clf.fit(X_train, y_train)
        reg.fit(X_train, pe_train)
        iso = IsotonicRegression(out_of_bounds="clip").fit(reg.predict(X_train), y_train)
        clf_test = clf.predict_proba(X_test)[:, 1]
        reg_test = iso.transform(reg.predict(X_test))
        test_preds.append(np.sqrt(np.clip(clf_test, 1e-6, 1.0) * np.clip(reg_test, 1e-6, 1.0)))
    return np.mean(test_preds, axis=0)


def extra_trees_stack(X_train, y_train, X_test, pe_train=None, **kw):
    clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1))])
    clf.fit(X_train, y_train); reg.fit(X_train, pe_train)
    iso = IsotonicRegression(out_of_bounds="clip").fit(reg.predict(X_train), y_train)
    return np.sqrt(np.clip(clf.predict_proba(X_test)[:, 1], 1e-6, 1.0) *
                    np.clip(iso.transform(reg.predict(X_test)), 1e-6, 1.0))


def log_pe_stack(X_train, y_train, X_test, pe_train=None, **kw):
    """Regress log(1+pe) instead of raw pe — compresses MMLV's 41% outlier."""
    clf = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1))])
    reg = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1))])
    clf.fit(X_train, y_train); reg.fit(X_train, np.log1p(pe_train))
    iso = IsotonicRegression(out_of_bounds="clip").fit(reg.predict(X_train), y_train)
    return np.sqrt(np.clip(clf.predict_proba(X_test)[:, 1], 1e-6, 1.0) *
                    np.clip(iso.transform(reg.predict(X_test)), 1e-6, 1.0))


def gp_reg_pe(X_train, y_train, X_test, pe_train=None, **kw):
    """Gaussian Process regression — built for small N. Predicts PE efficiency directly."""
    imp = SimpleImputer(strategy="median").fit(X_train)
    scl = StandardScaler().fit(imp.transform(X_train))
    Xtr = scl.transform(imp.transform(X_train))
    Xte = scl.transform(imp.transform(X_test))
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=42, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(Xtr, pe_train)
    return gp.predict(Xte)


def isotonic_stack_concat(X_train, y_train, X_test, pe_train=None, **kw):
    """Isotonic stack on concat(handcrafted, ESM)."""
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


runs = {}
runs["rf_multiseed_handonly"] = cv.lofo_predict(X_hand, y, fams, rf_multi_seed_stack, pe_efficiency=pe)
runs["extra_trees_handonly"] = cv.lofo_predict(X_hand, y, fams, extra_trees_stack, pe_efficiency=pe)
runs["log_pe_handonly"] = cv.lofo_predict(X_hand, y, fams, log_pe_stack, pe_efficiency=pe)
runs["gp_handonly"] = cv.lofo_predict(X_hand, y, fams, gp_reg_pe, pe_efficiency=pe)
runs["isotonic_concat"] = cv.lofo_predict(X_concat, y, fams, isotonic_stack_concat, pe_efficiency=pe)
runs["rf_multiseed_concat"] = cv.lofo_predict(X_concat, y, fams, rf_multi_seed_stack, pe_efficiency=pe)

summary = []
for name, scores in runs.items():
    res = metrics.cls(y, scores, pe)
    pf = metrics.per_family_pr_auc(y, scores, fams)
    rec = {"model": name, "cls": res["cls"], "pr_auc": res["pr_auc"], "w_spearman": res["w_spearman"]}
    for fam in ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]:
        rec[f"PRAUC_{fam}"] = pf.get(fam, {}).get("pr_auc")
    summary.append(rec)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"08_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("=== Day 3 round 2 (push harder) ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "08_summary.csv"), index=False)
