"""Step 1: build household features + target, audit, EDA figures, 70/10/20 stratified split."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import features as F
from config import DATA, RESULTS, FIGS, SEED, SPLIT, THRESHOLD_UNITS

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})
t = F.load_raw()
X = F.build_features(t)
info, keep = F.build_target(t)
S = {"households_surveyed": int(len(X)), "monthly_readings": int(len(t["cons"])),
     "dropped_too_few_months_or_mostly_zero": int((~keep).sum())}
ids = keep[keep].index.intersection(X.index)
X, info = X.loc[ids], info.loc[ids]
y = info["high_user"]
S.update(households_used=int(len(X)), high_user_rate=float(y.mean()), median_kwh=float(info.median_kwh.median()),
         kwh_quartiles=info.median_kwh.quantile([.25, .5, .75]).round(1).tolist(),
         missing={c: int(v) for c, v in X.isna().sum().items() if v},
         survey_period=[str(pd.to_datetime(t["dates"].wave1).min().date()), str(pd.to_datetime(t["dates"].wave1).max().date())])

# --- EDA figure: target distribution + three strongest raw relationships
fig, ax = plt.subplots(1, 4, figsize=(14, 3.1))
ax[0].hist(info.median_kwh.clip(upper=500), bins=50, color="#4f81bd")
ax[0].axvline(THRESHOLD_UNITS, c="#c0504d", ls="--"); ax[0].set(title=f"Median monthly use ({y.mean():.0%} > {THRESHOLD_UNITS} kWh)", xlabel="kWh / month (clipped at 500)")
E = X.assign(y=y)
E.groupby(E.n_acs.clip(upper=3))["y"].mean().plot.bar(ax=ax[1], color="#4f81bd"); ax[1].set(title="High-user rate by number of ACs", xlabel="ACs (3 = 3+)", ylabel="rate")
E.groupby(E.household_members.clip(upper=8))["y"].mean().plot(ax=ax[2], marker="o"); ax[2].set(title="…by household size", xlabel="members (8 = 8+)")
E.groupby("socio_economic_class")["y"].mean().plot.bar(ax=ax[3], color="#4f81bd"); ax[3].set(title="…by socio-economic class", xlabel="")
for a in ax[1:]: a.tick_params(axis="x", rotation=0)
plt.tight_layout(); plt.savefig(FIGS / "eda.png"); plt.close()

# --- split
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=SPLIT["val"] + SPLIT["test"], stratify=y, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=SPLIT["test"] / (SPLIT["val"] + SPLIT["test"]), stratify=y_tmp, random_state=SEED)
for n, a, b in [("train", X_tr, y_tr), ("val", X_va, y_va), ("test", X_te, y_te)]:
    a.assign(high_user=b.values, median_kwh=info.loc[a.index, "median_kwh"].values).reset_index(drop=True).to_parquet(DATA / f"{n}.parquet")
S["split"] = {n: {"rows": int(len(b)), "high_user_rate": round(float(b.mean()), 3)} for n, b in [("train", y_tr), ("val", y_va), ("test", y_te)]}
(RESULTS / "data_summary.json").write_text(json.dumps(S, indent=1))
print(json.dumps(S, indent=1))
