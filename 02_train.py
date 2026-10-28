"""Step 2: tune (5-fold CV on train, ROC-AUC) and fit EBM + the taught models."""
import json, os, time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from interpret.glassbox import ExplainableBoostingClassifier

import features as F
from config import DATA, MODELS, RESULTS, SEED

NJ = max(1, (os.cpu_count() or 2))
tr = pd.read_parquet(DATA / "train.parquet")
X, y = tr[F.FEATURES], tr["high_user"]
num = F.NUMERIC + F.FLAGS


def prep(scale):
    n = [("imp", SimpleImputer(strategy="median"))] + ([("sc", StandardScaler())] if scale else [])
    return ColumnTransformer([("num", Pipeline(n), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), F.CATEG)])


cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
SPACE = {
    "Logistic Regression": (Pipeline([("p", prep(True)), ("m", LogisticRegression(max_iter=3000))]),
                            {"m__C": loguniform(1e-3, 10)}, 20),
    "Decision Tree": (Pipeline([("p", prep(False)), ("m", DecisionTreeClassifier(random_state=SEED))]),
                      {"m__max_depth": randint(2, 12), "m__min_samples_leaf": randint(5, 80)}, 40),
    "Random Forest": (Pipeline([("p", prep(False)), ("m", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=NJ))]),
                      {"m__max_depth": [None, 6, 10, 16], "m__min_samples_leaf": randint(1, 20), "m__max_features": ["sqrt", 0.3, 0.5]}, 25),
    "XGBoost": (Pipeline([("p", prep(False)), ("m", XGBClassifier(n_estimators=400, random_state=SEED, n_jobs=NJ, eval_metric="logloss"))]),
                {"m__max_depth": randint(2, 7), "m__learning_rate": loguniform(0.01, 0.2), "m__subsample": uniform(0.6, 0.4),
                 "m__colsample_bytree": uniform(0.5, 0.5), "m__min_child_weight": randint(1, 15), "m__reg_lambda": loguniform(0.1, 20)}, 40),
    "LightGBM": (Pipeline([("p", prep(False)), ("m", LGBMClassifier(n_estimators=400, random_state=SEED, n_jobs=NJ, verbose=-1))]),
                 {"m__num_leaves": randint(4, 48), "m__learning_rate": loguniform(0.01, 0.2), "m__min_child_samples": randint(10, 80),
                  "m__subsample": uniform(0.6, 0.4), "m__subsample_freq": [1], "m__colsample_bytree": uniform(0.5, 0.5),
                  "m__reg_lambda": loguniform(0.1, 20)}, 40),
    "EBM": (ExplainableBoostingClassifier(random_state=SEED, n_jobs=1, outer_bags=8, feature_types=None),
            {"interactions": [0, 5, 10, 15], "learning_rate": loguniform(0.005, 0.05), "max_bins": [64, 128, 256],
             "min_samples_leaf": [2, 4, 10, 20], "max_leaves": [2, 3]}, 10),
}
TF = RESULTS / "tuning.json"
T = json.loads(TF.read_text()) if TF.exists() else {}
for name, (est, space, n_iter) in SPACE.items():
    if name in T and (MODELS / f"{name.lower().replace(' ', '_')}.pkl").exists():
        continue
    t0 = time.time()
    s = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="roc_auc", cv=cv, random_state=SEED,
                           n_jobs=1 if name in ("Random Forest", "XGBoost", "LightGBM") else NJ, refit=True)
    s.fit(X, y)
    T[name] = {"cv_auc": round(float(s.best_score_), 4), "cv_auc_sd": round(float(s.cv_results_["std_test_score"][s.best_index_]), 4),
               "best_params": {k: (v if isinstance(v, (int, str, type(None))) else float(v)) for k, v in s.best_params_.items()},
               "seconds": round(time.time() - t0)}
    joblib.dump(s.best_estimator_, MODELS / f"{name.lower().replace(' ', '_')}.pkl", compress=3)
    print(name, T[name], flush=True)
    TF.write_text(json.dumps(T, indent=1))
