"""Streamlit front end: will this Sri Lankan household use more than 90 kWh a month?"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import features as F
from xai_utils import LimeHH, shap_by_feature, ebm_local
from config import DATA, MODELS, RESULTS, FIGS, THRESHOLD_UNITS

st.set_page_config(page_title="Household Electricity Use Predictor", page_icon="⚡", layout="wide")
L = F.LABELS
NAMES = ["EBM", "LightGBM", "XGBoost", "Random Forest", "Logistic Regression", "Decision Tree"]


@st.cache_resource
def load():
    models = {n: joblib.load(MODELS / f"{n.lower().replace(' ', '_')}.pkl") for n in NAMES}
    tr = pd.read_parquet(DATA / "train.parquet"); te = pd.read_parquet(DATA / "test.parquet")
    res = json.loads((RESULTS / "results.json").read_text())
    return models, tr, te, res, LimeHH(tr[F.FEATURES], seed=0)


models, tr, te, res, lime = load()
met = pd.DataFrame(res["metrics"]).set_index("model")
med = tr[F.FEATURES].median(numeric_only=True)

with st.sidebar:
    st.markdown("## ⚡ Household Electricity Use")
    st.caption(f"Predicts whether a Sri Lankan household's typical monthly use is above **{THRESHOLD_UNITS} kWh**, where the "
               "domestic tariff moves to a much higher price block, using only answers from a household survey (no bills).")
    model_name = st.selectbox("Model", NAMES, index=0)
    c1, c2 = st.columns(2)
    c1.metric("Test ROC-AUC", f"{met.loc[model_name, 'ROC-AUC']:.3f}"); c2.metric("Accuracy", f"{met.loc[model_name, 'Accuracy']:.3f}")
    st.caption("Data: ReCon SL (IEEE DataPort), 3,983 real households from Moratuwa, Panadura, Galle, Kalutara and nearby areas, 2023–24.")
model = models[model_name]

t1, t2, t3, t4 = st.tabs(["🏠 Check a household", "📊 Model comparison", "🔍 What the model learned", "⚖️ Ethics & limits"])

with t1:
    src = st.radio("Start from", ["a real household from the test set", "a blank form (typical values)"], horizontal=True)
    if src.startswith("a real"):
        i = st.number_input(f"Test household # (0–{len(te) - 1})", 0, len(te) - 1, 0)
        base = te.iloc[i][F.FEATURES].copy(); actual = te.iloc[i]
    else:
        base = pd.Series({**med.to_dict(), **{c: tr[c].mode()[0] for c in F.CATEG}}); actual = None
    st.caption("Edit any answer; everything else stays as in the starting household.")
    a, b, c, d = st.columns(4)
    v = base.to_dict()
    v["household_members"] = a.number_input(L["household_members"], 1, 15, int(v["household_members"]))
    v["floor_area_sqft"] = a.number_input(L["floor_area_sqft"], 100, 9000, int(v["floor_area_sqft"] if pd.notna(v["floor_area_sqft"]) else 1000), step=100)
    v["house_type"] = a.selectbox(L["house_type"], sorted(tr.house_type.unique()), index=sorted(tr.house_type.unique()).index(v["house_type"]))
    v["socio_economic_class"] = a.selectbox(L["socio_economic_class"], sorted(tr.socio_economic_class.unique()),
                                            index=sorted(tr.socio_economic_class.unique()).index(v["socio_economic_class"]))
    v["n_acs"] = b.number_input(L["n_acs"], 0, 30, int(v["n_acs"]))
    v["ac_hours_week"] = b.number_input(L["ac_hours_week"], 0, 1200, int(v["ac_hours_week"]))
    v["n_fans"] = b.number_input(L["n_fans"], 0, 80, int(v["n_fans"]))
    v["n_bulbs"] = b.number_input(L["n_bulbs"], 0, 250, int(v["n_bulbs"]))
    v["n_fridges"] = c.number_input(L["n_fridges"], 0, 5, int(v["n_fridges"]))
    v["n_washing_machines"] = c.number_input(L["n_washing_machines"], 0, 3, int(v["n_washing_machines"]))
    v["n_water_heaters"] = c.number_input(L["n_water_heaters"], 0, 4, int(v["n_water_heaters"]))
    v["n_appliances"] = c.number_input(L["n_appliances"], 0, 70, int(v["n_appliances"]))
    v["hot_water"] = d.selectbox(L["hot_water"], sorted(tr.hot_water.unique()), index=sorted(tr.hot_water.unique()).index(v["hot_water"]))
    v["electric_cooking"] = int(d.checkbox(L["electric_cooking"], bool(v["electric_cooking"])))
    v["solar"] = int(d.checkbox(L["solar"], bool(v["solar"])))
    v["solar_on_grid"] = int(d.checkbox(L["solar_on_grid"], bool(v["solar_on_grid"])))
    row = pd.DataFrame([v])[F.FEATURES]
    for col in F.NUMERIC + F.FLAGS:
        row[col] = pd.to_numeric(row[col])

    p = float(model.predict_proba(row)[0, 1])
    m1, m2, m3 = st.columns(3)
    m1.metric(f"P(> {THRESHOLD_UNITS} kWh / month), {model_name}", f"{p:.0%}")
    m2.metric("Prediction", "HIGH USER" if p >= .5 else "≤ 90 kWh")
    if actual is not None:
        m3.metric("Actual (median of 12 months)", f"{actual.median_kwh:.0f} kWh")
    st.progress(p)

    left, right = st.columns([1, 1.3])
    with left:
        st.caption("Every model on the same household")
        st.dataframe(pd.DataFrame({"P(>90)": {n: f"{m.predict_proba(row)[0, 1]:.0%}" for n, m in models.items()}}), use_container_width=True)
    with right:
        opts = ["EBM (exact)", "LIME"] if model_name == "EBM" else (["SHAP", "LIME"] if model_name in ("LightGBM", "XGBoost", "Random Forest") else ["LIME"])
        how = st.radio("Explanation", opts, horizontal=True, help="EBM's own contributions are exact. SHAP and LIME approximate black-box models.")
        if how.startswith("EBM"):
            s, _ = ebm_local(model, row)
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:10]
            s.index = [" × ".join(L.get(x.strip(), x.strip()) for x in k.split("&")) for k in s.index]
        elif how == "SHAP":
            import shap
            Xt = model[0].transform(row); sv = shap.TreeExplainer(model[-1]).shap_values(Xt)
            sv = sv[1] if isinstance(sv, list) else sv
            if np.ndim(sv) == 3: sv = sv[:, :, 1]
            s = shap_by_feature(sv, model[0].get_feature_names_out()).iloc[0]
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:10].rename(lambda k: L.get(k, k))
        else:
            with st.spinner("Running LIME (3,000 perturbed households)…"):
                s = pd.Series(dict(lime.explain(model, row)[0]))
        s = s[::-1]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.barh(range(len(s)), s.values, color=["#c0504d" if x < 0 else "#4f81bd" for x in s.values])
        ax.set_yticks(range(len(s)), [str(k)[:42] for k in s.index], fontsize=8); ax.axvline(0, c="grey", lw=.6)
        ax.set_title(f"Why? {how}" + ("" if how.startswith("EBM") else f" on {model_name}"), fontsize=9); ax.set_xlabel("← lowers P(high user)    raises →", fontsize=8)
        st.pyplot(fig, clear_figure=True)

with t2:
    show = met[["ROC-AUC", "AUC CI low", "AUC CI high", "Accuracy", "Precision", "Recall", "F1", "Brier", "CV AUC", "Overfit gap"]]
    st.dataframe(show.style.format("{:.3f}"), use_container_width=True)
    st.caption("Test set: 797 held-out households. 95% CIs from 500 bootstrap resamples.")
    st.dataframe(pd.DataFrame(res["pairs_vs_ebm"]).set_index("model").style.format({c: "{:.3f}" for c in ["dAUC (EBM - model)", "CI low", "CI high"]}), use_container_width=True)
    st.image(str(FIGS / "eval.png"))

with t3:
    st.image(str(FIGS / "ebm_importance.png")); st.image(str(FIGS / "ebm_shapes.png")); st.image(str(FIGS / "agreement.png"))
    st.write("**Domain checks**", {k: ("✅" if v else "❌") for k, v in res["domain_checks"].items()})

with t4:
    st.markdown("""
- **Use:** screening and planning (e.g. targeting energy-efficiency advice or solar schemes), not billing or denying service.
- **Data:** surveyed households from a research sample in Western and Southern Province service areas, so results may not transfer to rural or other provinces.
- **Timing:** only survey answers are used; bills, expenditure and solar-generation figures are excluded so the model never sees the outcome.
- **Fairness:** socio-economic class is a strong predictor. Using it to set prices would penalise households for who they are; the Ethics section of the report shows error rates by class.
- **Privacy:** no names, addresses or account numbers; the app holds only de-identified survey features of test households.
""")
    st.dataframe(pd.DataFrame(res["fairness"]).set_index("SEC"), use_container_width=True)
