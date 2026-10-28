"""Shared explanation helpers (used by 03_evaluate_explain.py and app.py)."""
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

import features as F


class LimeHH:
    """LIME over the raw household features; categorical columns encoded as integer codes."""

    def __init__(self, background: pd.DataFrame, seed=0):
        self.cats = {c: sorted(background[c].astype(str).unique()) for c in F.CATEG}
        self.med = background[F.NUMERIC + F.FLAGS].median()
        enc = self.encode(background)
        self.exp = LimeTabularExplainer(enc, feature_names=F.FEATURES, class_names=["≤90", ">90"],
                                        categorical_features=[F.FEATURES.index(c) for c in F.CATEG],
                                        categorical_names={F.FEATURES.index(c): v for c, v in self.cats.items()},
                                        discretize_continuous=True, random_state=seed, mode="classification")

    def encode(self, X):
        Z = X[F.FEATURES].copy()
        Z[F.NUMERIC + F.FLAGS] = Z[F.NUMERIC + F.FLAGS].fillna(self.med)
        for c, v in self.cats.items():
            Z[c] = Z[c].astype(str).map({k: i for i, k in enumerate(v)}).fillna(0)
        return Z.to_numpy(float)

    def decode(self, A):
        Z = pd.DataFrame(A, columns=F.FEATURES)
        for c, v in self.cats.items():
            Z[c] = [v[int(round(i))] for i in Z[c].clip(0, len(v) - 1)]
        return Z

    def explain(self, model, row: pd.DataFrame, num_features=10, num_samples=3000):
        fn = lambda A: model.predict_proba(self.decode(A))
        e = self.exp.explain_instance(self.encode(row)[0], fn, num_features=num_features, num_samples=num_samples, labels=(1,))
        return e.as_list(label=1), e

    @staticmethod
    def by_feature(lst):
        """Map LIME condition strings back to feature names."""
        out = {}
        for cond, w in lst:
            name = next((f for f in sorted(F.FEATURES, key=len, reverse=True) if f in cond), cond)
            out[name] = out.get(name, 0) + w
        return out


def base_feature(col):
    """'num__n_acs' -> 'n_acs'; 'cat__house_type_flat' -> 'house_type'."""
    c = col.split("__", 1)[-1]
    for f in sorted(F.FEATURES, key=len, reverse=True):
        if c == f or c.startswith(f + "_"):
            return f
    return c


def shap_by_feature(values, cols):
    s = pd.DataFrame(values, columns=[base_feature(c) for c in cols])
    return s.T.groupby(level=0).sum().T


def ebm_local(ebm, row):
    d = ebm.explain_local(row[F.FEATURES]).data(0)
    return pd.Series(d["scores"], index=d["names"]), d.get("extra", {}).get("scores", [0])[0]
