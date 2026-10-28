"""Step 3: test-set evaluation (with bootstrap CIs) and explainability (EBM, SHAP, LIME)."""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from scipy.stats import spearmanr
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
                             brier_score_loss, roc_curve, confusion_matrix)

import features as F
from xai_utils import LimeHH, shap_by_feature, ebm_local
from config import DATA, MODELS, RESULTS, FIGS, SEED, BOOT

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
L = F.LABELS
tr, va, te = (pd.read_parquet(DATA / f"{n}.parquet") for n in ("train", "val", "test"))
Xtr, ytr, Xte, yte = tr[F.FEATURES], tr.high_user, te[F.FEATURES], te.high_user
NAMES = ["EBM", "Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "LightGBM"]
M = {n: joblib.load(MODELS / f"{n.lower().replace(' ', '_')}.pkl") for n in NAMES}
P = {n: m.predict_proba(Xte)[:, 1] for n, m in M.items()}
T = json.loads((RESULTS / "tuning.json").read_text())

# ------------------------------------------------------------------ evaluation
rng = np.random.default_rng(SEED)
B = [rng.integers(0, len(yte), len(yte)) for _ in range(BOOT)]
rows, pairs = [], []
for n in NAMES:
    p = P[n]; yh = (p >= 0.5).astype(int)
    boots = [roc_auc_score(yte.iloc[b], p[b]) for b in B]
    rows.append({"model": n, "ROC-AUC": roc_auc_score(yte, p), "AUC CI low": np.percentile(boots, 2.5), "AUC CI high": np.percentile(boots, 97.5),
                 "Accuracy": accuracy_score(yte, yh), "Precision": precision_score(yte, yh), "Recall": recall_score(yte, yh),
                 "F1": f1_score(yte, yh), "Brier": brier_score_loss(yte, p), "CV AUC": T[n]["cv_auc"],
                 "Train AUC": roc_auc_score(ytr, M[n].predict_proba(Xtr)[:, 1])})
    if n != "EBM":
        d = [roc_auc_score(yte.iloc[b], P["EBM"][b]) - roc_auc_score(yte.iloc[b], p[b]) for b in B]
        pairs.append({"model": n, "dAUC (EBM - model)": float(np.mean(d)), "CI low": float(np.percentile(d, 2.5)),
                      "CI high": float(np.percentile(d, 97.5)), "significant": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)})
met = pd.DataFrame(rows); met["Overfit gap"] = met["Train AUC"] - met["ROC-AUC"]
print(met.round(4).to_string(index=False)); print(pd.DataFrame(pairs).round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for n in NAMES:
    f, t, _ = roc_curve(yte, P[n]); ax[0].plot(f, t, lw=2.2 if n == "EBM" else 1, label=f"{n} ({roc_auc_score(yte, P[n]):.3f})")
ax[0].plot([0, 1], [0, 1], "k:", lw=.8); ax[0].set(title="ROC curves (test set)", xlabel="false-positive rate", ylabel="true-positive rate"); ax[0].legend(fontsize=7)
cm = confusion_matrix(yte, (P["EBM"] >= .5).astype(int))
ax[1].imshow(cm, cmap="Blues"); [ax[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=12) for i in range(2) for j in range(2)]
ax[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["≤90", ">90"], yticklabels=["≤90", ">90"], xlabel="predicted", ylabel="actual", title="EBM confusion matrix (threshold 0.5)")
plt.tight_layout(); plt.savefig(FIGS / "eval.png"); plt.close()

# ------------------------------------------------------------------ EBM global
ebm = M["EBM"]
imp = pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)
lab = lambda t: " × ".join(L.get(p.strip(), p.strip()) for p in t.split("&"))
ax = imp.head(15)[::-1].rename(lab).plot.barh(figsize=(7, 4.2), color=["#e69f00" if "&" in t else "#4f81bd" for t in imp.head(15).index[::-1]])
ax.set_title("EBM global importance (mean |contribution|, log-odds)\norange = pairwise interaction", fontsize=9); ax.set_xlabel(""); plt.tight_layout(); plt.savefig(FIGS / "ebm_importance.png"); plt.close()

G = ebm.explain_global()
def shape(term):
    i = ebm.term_names_.index(term); d = G.data(i)
    return d["names"], np.asarray(d["scores"]), d.get("lower_bounds"), d.get("upper_bounds")

mains = [t for t in imp.index if "&" not in t][:6]
fig, axs = plt.subplots(2, 3, figsize=(12, 5.6))
for a, t in zip(axs.ravel(), mains):
    names, sc, lo, hi = shape(t)
    if t in F.CATEG:
        a.bar(range(len(sc)), sc, color="#4f81bd"); a.set_xticks(range(len(sc)), names, rotation=30, ha="right", fontsize=7)
    else:
        e = np.asarray(names, float); x = (e[:-1] + e[1:]) / 2
        hi_x = np.nanpercentile(Xtr[t].dropna(), 99) if Xtr[t].notna().any() else x.max()
        k = x <= max(hi_x, x.min()); a.step(x[k], sc[k], where="mid", color="#4f81bd")
        if lo is not None: a.fill_between(x[k], np.asarray(lo)[k], np.asarray(hi)[k], step="mid", alpha=.2)
    a.axhline(0, c="grey", lw=.6); a.set(title=L.get(t, t), ylabel="log-odds")
plt.suptitle("EBM shape functions for the six most important features (99th-percentile range shown)"); plt.tight_layout(); plt.savefig(FIGS / "ebm_shapes.png"); plt.close()

def score_at(term, value):
    names, sc, *_ = shape(term)
    if term in F.CATEG:
        return float(sc[list(names).index(value)]) if value in names else np.nan
    e = np.asarray(names, float); j = np.clip(np.searchsorted(e, value, side="right") - 1, 0, len(sc) - 1); return float(sc[j])

checks = {
    "More AC hours raise the odds (0 vs 40 h/week)": score_at("ac_hours_week", 40) > score_at("ac_hours_week", 0),
    "Larger households use more (2 vs 6 members)": score_at("household_members", 6) > score_at("household_members", 2),
    "Grid-powered water heating raises use (vs none)": score_at("hot_water", "electric (grid)") > score_at("hot_water", "none"),
    "On-grid solar lowers metered use (expected; NOT observed)": bool(score_at("solar_on_grid", 1) < score_at("solar_on_grid", 0)),
    "Higher socio-economic class uses more (A vs E)": score_at("socio_economic_class", "A") > score_at("socio_economic_class", "E"),
    "Bigger floor area uses more (600 vs 2000 sq ft)": score_at("floor_area_sqft", 2000) > score_at("floor_area_sqft", 600),
}
inter = [t for t in imp.index if "&" in t][:5]

# ------------------------------------------------------------------ local: two real test households
pe = P["EBM"]
cases = {"likely high user": int(np.argmax(np.where(yte == 1, pe, -1))),
         "borderline": int(np.argmin(np.abs(pe - 0.5)))}
lime = LimeHH(Xtr, seed=SEED)
lgb = M["LightGBM"]; Xt_lgb = lgb[0].transform(Xte); cols = lgb[0].get_feature_names_out()
sv = shap.TreeExplainer(lgb[-1]).shap_values(Xt_lgb); sv = sv[1] if isinstance(sv, list) else sv
S_feat = shap_by_feature(sv, cols)

fig, axs = plt.subplots(len(cases), 3, figsize=(14, 3.4 * len(cases)))
local = {}
for r, (cname, i) in enumerate(cases.items()):
    row = Xte.iloc[[i]]
    e_sc, _ = ebm_local(ebm, row); e_sc = e_sc.reindex(e_sc.abs().sort_values(ascending=False).index)[:8]
    l_list, _ = lime.explain(ebm, row); l_feat = LimeHH.by_feature(l_list)
    s_row = S_feat.iloc[i]; s_top = s_row.reindex(s_row.abs().sort_values(ascending=False).index)[:8]
    for a, ser, ttl in [(axs[r, 0], e_sc.rename(lab), "EBM (exact)"), (axs[r, 1], pd.Series(dict(l_list)), "LIME on EBM"),
                        (axs[r, 2], s_top.rename(lab), "SHAP on LightGBM")]:
        ser = ser[::-1]; a.barh(range(len(ser)), ser.values, color=["#c0504d" if v < 0 else "#4f81bd" for v in ser.values])
        a.set_yticks(range(len(ser)), [str(s)[:38] for s in ser.index], fontsize=7); a.axvline(0, c="grey", lw=.6)
        a.set_title(f"{ttl}: {cname} (EBM p = {pe[i]:.2f}, actual {'>90' if yte.iloc[i] else '≤90'})", fontsize=8)
    e_all, _ = ebm_local(ebm, row)
    common = [f for f in F.FEATURES if f in e_all.index]
    local[cname] = {"p_ebm": float(pe[i]), "actual": int(yte.iloc[i]), "median_kwh": float(te.median_kwh.iloc[i]),
                    "top_ebm": {lab(k): round(float(v), 3) for k, v in e_sc.items()},
                    "rho_ebm_shap": float(spearmanr(e_all[common], S_feat.iloc[i][common]).correlation)}
plt.tight_layout(); plt.savefig(FIGS / "local.png"); plt.close()

# ------------------------------------------------------------------ agreement across explainers (global)
shap_glob = S_feat.abs().mean()
ebm_main = pd.Series({f: imp.get(f, 0) for f in F.FEATURES})
rho_global = float(spearmanr(ebm_main, shap_glob.reindex(F.FEATURES).fillna(0)).correlation)
top10_overlap = len(set(ebm_main.nlargest(10).index) & set(shap_glob.nlargest(10).index))
rng2 = np.random.default_rng(1); sample = rng2.choice(len(Xte), 30, replace=False)
rhos, stab = [], []
for i in sample:
    row = Xte.iloc[[i]]; e_all, _ = ebm_local(ebm, row)
    l_feat = LimeHH.by_feature(lime.explain(ebm, row, num_features=len(F.FEATURES))[0])
    common = [f for f in F.FEATURES if f in e_all.index]
    rhos.append(spearmanr(e_all[common], pd.Series(l_feat).reindex(common).fillna(0)).correlation)
for s in range(5):  # LIME stability on one case
    l = LimeHH(Xtr, seed=s).explain(ebm, Xte.iloc[[cases['borderline']]])[0]; stab.append(list(LimeHH.by_feature(l))[0])
fig, ax = plt.subplots(figsize=(7, 3.6))
top = ebm_main.nlargest(12).index
z = pd.DataFrame({"EBM": ebm_main[top] / ebm_main[top].max(), "SHAP (LightGBM)": shap_glob[top] / shap_glob[top].max()})
z.rename(index=lab)[::-1].plot.barh(ax=ax); ax.set(title=f"Global importance: EBM vs SHAP (Spearman ρ = {rho_global:.2f}, all features)", xlabel="scaled to max = 1")
plt.tight_layout(); plt.savefig(FIGS / "agreement.png"); plt.close()

# ------------------------------------------------------------------ fairness by socio-economic class
yh = (pe >= .5).astype(int); fair = []
for g, idx in Xte.groupby("socio_economic_class").groups.items():
    k = Xte.index.get_indexer(idx); yt = yte.iloc[k]
    if len(k) < 20: continue
    fair.append({"SEC": g, "n": len(k), "actual high-user rate": float(yt.mean()), "predicted rate": float(yh[k].mean()),
                 "recall": float(recall_score(yt, yh[k], zero_division=0)), "precision": float(precision_score(yt, yh[k], zero_division=0)),
                 "AUC": float(roc_auc_score(yt, pe[k])) if yt.nunique() > 1 else None})
fair = pd.DataFrame(fair).sort_values("SEC")
print(fair.round(3).to_string(index=False))

out = {"metrics": met.round(4).to_dict(orient="records"), "pairs_vs_ebm": pd.DataFrame(pairs).round(4).to_dict(orient="records"),
       "ebm_importance": imp.round(4).to_dict(), "interactions": inter, "domain_checks": checks, "local": local,
       "agreement": {"rho_ebm_shap_global": rho_global, "top10_overlap": top10_overlap,
                     "rho_ebm_lime_local_median": float(np.nanmedian(rhos)), "lime_top_feature_stability": f"{stab.count(stab[0])}/5 ({stab[0]})"},
       "fairness": fair.round(3).to_dict(orient="records"), "ebm_params": T["EBM"]["best_params"]}
(RESULTS / "results.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: out[k] for k in ["domain_checks", "agreement", "interactions"]}, indent=1))
