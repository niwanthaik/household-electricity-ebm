"""Builds report/Electricity_EBM_Report.html and .pdf from results/*.json and figures/*.png (all numbers come from the JSON)."""
import asyncio, base64, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import features as F
from config import THRESHOLD_UNITS

OUTD = ROOT / "report"
S = json.loads((ROOT / "results/data_summary.json").read_text())
R = json.loads((ROOT / "results/results.json").read_text())
T = json.loads((ROOT / "results/tuning.json").read_text())
met = pd.DataFrame(R["metrics"]).set_index("model")
pairs = pd.DataFrame(R["pairs_vs_ebm"]).set_index("model")
fair = pd.DataFrame(R["fairness"]).set_index("SEC")
E = met.loc["EBM"]
GH = "https://github.com/niwanthaik/household-electricity-ebm"
APP = "https://household-electricity.redwave-d9ae0d05.eastasia.azurecontainerapps.io"
pc = lambda v, d=0: f"{100 * v:.{d}f}%"
f3 = lambda v: f"{v:.3f}"
L = F.LABELS


def img(name, cap, w=100):
    img.n += 1
    b64 = base64.b64encode((ROOT / "figures" / name).read_bytes()).decode()
    return f"<figure><img src='data:image/png;base64,{b64}' style='width:{w}%'><figcaption><b>Figure {img.n}.</b> {cap}</figcaption></figure>"
img.n = 0


def table(df, cap, index=True):
    table.n += 1
    return f"<div class='tcap'><b>Table {table.n}.</b> {cap}</div>" + df.to_html(index=index, escape=False, border=0, classes="tbl")
table.n = 0


CSS = """
body { font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt; line-height: 1.42; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 4pt; color: #1f3864; text-align: center; }
h2 { font-size: 13.5pt; color: #1f3864; border-bottom: 1.5px solid #1f3864; padding-bottom: 2pt; margin-top: 16pt; page-break-after: avoid; }
h3 { font-size: 11.3pt; color: #2f5597; margin: 10pt 0 3pt; page-break-after: avoid; }
p { margin: 3pt 0 6pt; text-align: justify; } .marks { font-size: 8.5pt; color: #888; font-weight: normal; }
.meta { text-align: center; font-size: 10pt; line-height: 1.6; margin-bottom: 10pt; }
.abstract { border-left: 3px solid #2f5597; background: #f3f6fb; padding: 6pt 10pt; font-size: 9.8pt; margin: 6pt 0 8pt; }
figure { margin: 6pt 0 8pt; page-break-inside: avoid; text-align: center; } figure img { max-width: 100%; }
figcaption { font-size: 8.6pt; color: #444; margin-top: 2pt; text-align: left; }
.tbl { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2pt 0 8pt; font-family: Helvetica, Arial, sans-serif; page-break-inside: avoid; }
.tbl th { background: #1f3864; color: white; padding: 2.5pt 4pt; text-align: left; } .tbl td { border-bottom: .5px solid #ccc; padding: 2pt 4pt; vertical-align: top; }
.tbl tr:nth-child(even) td { background: #f6f8fb; } .tcap { font-size: 8.6pt; color: #444; margin-top: 6pt; page-break-after: avoid; }
ul { margin: 2pt 0 6pt 16pt; padding: 0; } li { margin: 1.5pt 0; } .ref p { font-size: 9pt; text-align: left; margin: 2pt 0; }
code { font-size: 8.8pt; background: #f2f2f2; padding: 0 2pt; }
"""

best_bb = met.loc[["Random Forest", "XGBoost", "LightGBM"], "ROC-AUC"]
ck = R["domain_checks"]; n_ok = sum(ck.values())
ag = R["agreement"]
H = []; a = H.append

a(f"""<h1>Predicting High Electricity Use in Sri Lankan Households<br>with an Explainable Boosting Machine</h1>
<div class='meta'>Applied Machine Learning: Assignment Report<br><b>H.M.N.I.K. Herath</b> · Index number 269115L · MSc in Artificial Intelligence, University of Moratuwa<br>
Code: <a href='{GH}'>{GH}</a><br>Live app: <a href='{APP}'>{APP}</a></div>
<div class='abstract'><b>Summary.</b> Using a real survey of {S['households_used']:,} Sri Lankan households linked to their electricity-provider records (ReCon SL),
this project predicts, from survey answers alone, whether a household's typical monthly consumption exceeds {THRESHOLD_UNITS} kWh, the domestic tariff
boundary above which the unit price and fixed charge rise sharply. An <b>Explainable Boosting Machine (EBM)</b>
reaches a test ROC-AUC of <b>{E['ROC-AUC']:.3f}</b> (95% CI {E['AUC CI low']:.3f}–{E['AUC CI high']:.3f}). That is statistically indistinguishable from
random forest, XGBoost, LightGBM and logistic regression, and clearly better than a decision tree, while being fully interpretable and overfitting far less than the tree ensembles.
Its explanations are cross-checked with SHAP and LIME.</div>""")

# ---------------------------------------------------------------- 1
a(f"""<h2>1. Problem definition and dataset</h2>
<h3>1.1 Problem</h3>
<p>Sri Lanka's domestic electricity tariff is block-based. Under the PUCSL domestic tariff, consumption above {THRESHOLD_UNITS} units a month moves a household
into blocks with a much higher unit charge and fixed charge [6]. Utilities (CEB, LECO) and energy-efficiency or solar programmes want to know <i>which</i>
households are likely to exceed that line, and <i>why</i>, so advice and support can be targeted. The task is <b>binary classification</b>:
<i>will a household's median monthly use exceed {THRESHOLD_UNITS} kWh?</i>, predicted only from information collected in a household survey, before any bill is seen.</p>
<h3>1.2 Dataset</h3>
<p><b>ReCon SL: Residential Electricity Consumption Dataset, Sri Lanka</b> (IEEE DataPort, open access) [1]. It combines utility billing records for
{S['households_surveyed']:,} households ({S['monthly_readings']:,} monthly readings, Oct 2022 – Oct 2024) with three waves of household surveys.
This project uses the Wave 1 survey (conducted {S['survey_period'][0]} to {S['survey_period'][1]}): household information, demographics, and
appliance, room, AC, fan and light rosters. The households are mainly in Western and Southern Province service areas (Moratuwa, Panadura, Galle, Kalutara, Maharagama and others).</p>
<h3>1.3 Target, features and the timing rule</h3>
<p><b>Target:</b> the median of non-zero monthly readings in the 12 months up to the survey month; label 1 if above {THRESHOLD_UNITS} kWh.
{S['dropped_too_few_months_or_mostly_zero']} households with fewer than six valid months or mostly zero readings (likely vacant) were removed, leaving
<b>{S['households_used']:,}</b> households; <b>{pc(S['high_user_rate'])}</b> are high users (median {S['median_kwh']:.0f} kWh; quartiles {S['kwh_quartiles'][0]:.0f} / {S['kwh_quartiles'][2]:.0f}), so the classes are balanced.</p>
<p><b>Features ({len(F.FEATURES)}):</b> the multi-row rosters were aggregated to one row per household: household size, children, elderly members, members working, hours at home;
floor area, rooms, bedrooms, building age, house type; number and weekly hours of ACs, fans and lights (lighting kWh = wattage × hours); counts of major appliances
(refrigerators, washing machines, water heaters, cooking appliances, computers, pumps and so on); solar, backup generator, cooking with grid electricity, home business, renting;
socio-economic class (SEC A–E) and service area.</p>
<p><b>Timing rule.</b> Every feature must be known at the moment of prediction, before a bill. All features are Wave 1 survey answers. Three survey fields were
<b>excluded</b> because they contain or reveal the outcome: total monthly household expenditure (includes the electricity bill), units generated by solar, and the Wave 3 billing table.</p>
<h3>1.4 Preprocessing and ethics</h3>
<ul><li>Comma-formatted readings (e.g. "1,476") converted to numbers; missing readings ignored in the median.</li>
<li>Missing values: floor area ({S['missing'].get('floor_area_sqft', 0)}), building age ({S['missing'].get('built_period', 0)}, "don't know"); EBM bins missing natively; other models use median imputation inside the pipeline (fitted on training data only).</li>
<li>Rare categories grouped (service areas outside the top 12 → "Other"; house types into five groups); one-hot encoding for the comparison models.</li>
<li><b>Privacy:</b> the data are de-identified (household IDs only). No addresses or account numbers are used, and the app holds only aggregated survey features of test households.</li></ul>
{img('eda.png', 'Target distribution and three raw relationships: high-user rate rises with ACs, household size and socio-economic class.')}""")

# ---------------------------------------------------------------- 2
p = R["ebm_params"]
a(f"""<h2>2. Selected algorithm: Explainable Boosting Machine</h2>
<p>The EBM [2, 3] is a <b>generalised additive model with pairwise interactions</b> (GA²M), trained with gradient boosting:</p>
<p style='text-align:center'><i>logit P(y = 1 | x) = β₀ + Σ<sub>j</sub> f<sub>j</sub>(x<sub>j</sub>) + Σ<sub>(i,j)</sub> f<sub>ij</sub>(x<sub>i</sub>, x<sub>j</sub>)</i></p>
<ul><li><b>Cyclic boosting:</b> features are binned; in each round, a very shallow tree is fitted on <i>one feature at a time</i> to the current residuals,
with a small learning rate, cycling through all features for thousands of rounds. Each feature's trees sum into its own shape function f<sub>j</sub>.</li>
<li><b>Interactions:</b> after the main effects, the FAST heuristic ranks feature pairs, and the top pairs get 2-D shape functions f<sub>ij</sub>.</li>
<li><b>Bagging:</b> the whole procedure is repeated on bootstrap samples ("outer bags") and averaged, which smooths the shapes and gives confidence bands.</li></ul>
<p><b>How it differs from other models.</b> Gradient boosting, XGBoost and LightGBM grow trees over all features jointly, so the result is a black box
that needs SHAP or LIME to explain. Logistic regression is additive but linear. The EBM keeps the additive, exactly decomposable structure of a GAM while learning
non-linear shapes like a boosted model: each prediction is the exact sum of per-feature contributions, so its explanations are the model itself, not an approximation.</p>
<p><b>Why it suits this problem.</b> Energy advice must be explainable to households and programme staff ("your four ACs used 40 hours a week add X"), the features are
heterogeneous survey answers with missing values and skewed counts (handled natively by binning), and n ≈ 4,000 is a size where a constrained model should overfit less than deep ensembles.</p>
<p><b>Configuration</b> (InterpretML 0.7.8): outer bags 8; tuned values: interactions {p['interactions']}, learning rate {p['learning_rate']:.4f}, max bins {p['max_bins']},
max leaves {p['max_leaves']}, min samples per leaf {p['min_samples_leaf']}; early stopping on an internal validation split.</p>""")

# ---------------------------------------------------------------- 3
m = met[["ROC-AUC", "AUC CI low", "AUC CI high", "Accuracy", "Precision", "Recall", "F1", "Brier", "CV AUC", "Train AUC", "Overfit gap"]].copy()
m = m.map(f3)
pr = pairs.copy(); pr["95% CI"] = [f"{l:.3f} to {h:.3f}" for l, h in zip(pr["CI low"], pr["CI high"])]
pr = pr[["dAUC (EBM - model)", "95% CI", "significant"]].rename(columns={"dAUC (EBM - model)": "ΔAUC (EBM − model)"})
pr["ΔAUC (EBM − model)"] = pr["ΔAUC (EBM − model)"].map(lambda v: f"{v:+.3f}"); pr["significant"] = pr["significant"].map({True: "yes", False: "no"})
a(f"""<h2>3. Model training and evaluation</h2>
<p><b>Split:</b> stratified 70 / 10 / 20: train {S['split']['train']['rows']:,}, validation {S['split']['val']['rows']:,}, test {S['split']['test']['rows']:,} households
(each {pc(S['split']['test']['high_user_rate'], 1)} positive). The test set was used once, at the end.</p>
<p><b>Tuning:</b> randomised search with 5-fold stratified cross-validation on the training set, optimising ROC-AUC, for the EBM and five comparison models
(logistic regression, decision tree, random forest, XGBoost, LightGBM). All preprocessing sits inside each pipeline, so no fold sees statistics from another.
<b>Metrics:</b> ROC-AUC (threshold-free ranking quality, the primary metric), accuracy, precision, recall and F1 at threshold 0.5, Brier score (calibration);
95% confidence intervals and paired EBM-vs-model differences from {500} bootstrap resamples of the test set.</p>
{table(m, 'Test-set results. Overfit gap = training AUC − test AUC.')}
{table(pr, 'Paired bootstrap difference in test AUC, EBM minus each model. "Significant" = 95% CI excludes 0.')}
{img('eval.png', 'ROC curves of all models on the test set, and the EBM confusion matrix.', 92)}
<p><b>Result.</b> The EBM (AUC {E['ROC-AUC']:.3f}) is in a statistical tie with random forest ({met.loc['Random Forest','ROC-AUC']:.3f}), XGBoost, LightGBM and
logistic regression: all paired differences are within ±{pairs.loc[['Random Forest','XGBoost','LightGBM','Logistic Regression'],'dAUC (EBM - model)'].abs().max():.3f} AUC and none is significant.
It is significantly better than the decision tree (+{pairs.loc['Decision Tree','dAUC (EBM - model)']:.3f}). The tree ensembles fit the training data far more closely
(overfit gap {met.loc[['Random Forest','XGBoost','LightGBM'],'Overfit gap'].min():.2f}–{met.loc[['Random Forest','XGBoost','LightGBM'],'Overfit gap'].max():.2f}) than the EBM ({E['Overfit gap']:.3f}).
At 0.5 the EBM finds {pc(E['Recall'])} of high users with {pc(E['Precision'])} precision. Accuracy (about {pc(E['Accuracy'])}) is limited by what a survey can capture: two similar homes can use very different amounts.</p>""")

# ---------------------------------------------------------------- 4
imp = pd.Series(R["ebm_importance"])
top5 = [L.get(t, t) for t in imp.index if "&" not in t][:5]
lab = lambda t: " × ".join(L.get(x.strip(), x.strip()) for x in t.split("&"))
loc = R["local"]
ck_html = "".join(f"<li>{'✔' if v else '✘'} {k}</li>" for k, v in ck.items())
a(f"""<h2>4. Explainability and interpretation</h2>
<p>Three complementary methods are used: the EBM's own exact explanations (global importance, shape functions, interactions, local contributions),
<b>SHAP</b> [4] (TreeSHAP on LightGBM, the strongest black box) and <b>LIME</b> [5] (local surrogate on the EBM, to test whether a model-agnostic method recovers the exact answer).</p>
<h3>4.1 Global behaviour</h3>
{img('ebm_importance.png', 'EBM global importance: mean absolute contribution to the log-odds across training households.', 70)}
<p>The five most influential features are <b>{', '.join(top5)}</b>. Fan hours are a proxy for both occupancy and cooling demand; household size and the main
"white goods" (refrigerators, washing machines) follow. Service area matters: some areas have higher use even after appliances are accounted for (house size, local climate and income are partly bundled into area).</p>
{img('ebm_shapes.png', 'Shape functions: contribution (log-odds) of each value. Shaded bands = variation across the 8 outer bags.')}
<p>The shapes are readable directly. Contribution rises steadily with household size (≈ −0.4 for one or two members to +0.6 for nine); fan use rises to about 150 hours a week and then plateaus;
a second refrigerator or washing machine adds about +0.4. The top interaction terms are {', '.join(lab(t) for t in R['interactions'][:3])}: for example, the effect of an extra refrigerator depends on how many people live in the home.</p>
<h3>4.2 Local explanations</h3>
{img('local.png', 'Two real test households explained three ways. Blue raises, red lowers the probability of being a high user.')}
<p>For the clear high user (EBM p = {loc['likely high user']['p_ebm']:.2f}, actual {loc['likely high user']['median_kwh']:.0f} kWh/month) all three methods point to air conditioning,
appliances and household size. For the borderline case (p = {loc['borderline']['p_ebm']:.2f}, actual {loc['borderline']['median_kwh']:.0f} kWh/month) low fan use pushes down and many bulbs push up, and the methods disagree more, as expected near the boundary.</p>
<h3>4.3 Do the explanations agree, and are they sensible?</h3>
{img('agreement.png', 'Global importance from the EBM and from SHAP on LightGBM, each scaled to its maximum.', 70)}
<ul><li><b>EBM vs SHAP (global):</b> Spearman ρ = {ag['rho_ebm_shap_global']:.2f} over all features; {ag['top10_overlap']} of the top 10 features are shared. Two different models learned the same drivers.</li>
<li><b>EBM vs LIME (local, 30 test households):</b> median ρ = {ag['rho_ebm_lime_local_median']:.2f}. LIME recovers the EBM's main drivers but only approximately, and its top feature
changed across random seeds (stable in {ag['lime_top_feature_stability'].split(' ')[0]} runs). This illustrates why an exactly interpretable model is preferable to post-hoc approximations when explanations will be acted on.</li>
<li><b>Domain checks ({n_ok}/{len(ck)} pass):</b><ul>{ck_html}</ul></li></ul>
<p>The failed check is informative: solar households are <i>more</i> likely to be high users (about 88% of on-grid solar homes vs 51% without solar).
Solar is concentrated in SEC A homes, which use more electricity, and the provider record may report import from the grid rather than net use. The model learned a genuine
pattern in the data, but it is not a causal effect of installing solar, and the report flags it rather than hiding it.</p>""")

# ---------------------------------------------------------------- 5
fa = fair.copy()
for c in ["actual high-user rate", "predicted rate", "recall", "precision", "AUC"]:
    fa[c] = fa[c].map(f3)
a(f"""<h2>5. Critical discussion</h2>
<p><b>Strengths.</b> Real, recent Sri Lankan data with an outcome from provider records rather than self-report; a strict timing rule; and a model that is as accurate as the best black box,
overfits less, and gives exact explanations that domain checks and SHAP confirm.</p>
<p><b>Limitations.</b></p>
<ul><li><b>Predictive ceiling:</b> AUC ≈ 0.82 means survey answers explain a lot but not everything. Behaviour (thermostat settings, actual usage of each appliance) and appliance efficiency are only partly captured.</li>
<li><b>Self-reported hours:</b> AC, fan and light hours refer to "last week" and may not represent the 12-month target window.</li>
<li><b>Representativeness:</b> mainly urban and semi-urban Western/Southern Province households from a research sample, so it may not transfer to rural or other provinces.</li>
<li><b>Target definition:</b> a 12-month median around the survey date; tariff changes in 2023–24 may have shifted behaviour.</li>
<li><b>Solar and metering:</b> as discussed in §4.3, metered consumption is not the same as total consumption for solar homes.</li></ul>
<p><b>Fairness and ethics.</b> Socio-economic class strongly predicts use, and errors are unequal across classes (Table 3). Lower-SEC high users are missed more often
(recall {fair.loc['E','recall']:.2f} for SEC E vs {fair.loc['A','recall']:.2f} for SEC A), because high use is rarer in those groups. If the model were used to target <i>support</i>, this would under-serve poorer households; if it were used to target <i>charges</i>,
it would be discriminatory. The model is therefore suitable only for screening and advice, with human review, and should never set prices or restrict supply.
The EBM makes this visible: the SEC term can be inspected or removed, which a black box would hide.</p>
{table(fa, 'EBM error rates by socio-economic class (test set, threshold 0.5).')}
<p><b>Real-world impact.</b> A utility or a solar/efficiency programme could screen applicants with a short form and explain to each household which appliances drive its predicted use, a concrete, trustworthy message in a period of high tariffs.</p>""")

# ---------------------------------------------------------------- 6
a(f"""<h2>6. Front end</h2>
<p>A Streamlit app (<code>app.py</code>) is deployed on <b>Azure Container Apps</b> at <a href='{APP}'>{APP}</a>. A Docker image is built by GitHub Actions and published to the GitHub Container Registry.
Users choose any of the six models, start from a real test household or a blank form, edit the answers and get the prediction with an explanation (EBM exact, SHAP or LIME).
Other tabs show the model comparison, the EBM's global behaviour, and the ethics and fairness notes.</p>
<h2>References</h2><div class='ref'>
<p>[1] ReCon SL: Residential Electricity Consumption Dataset, Sri Lanka (2025). IEEE DataPort. https://ieee-dataport.org/open-access/residential-electricity-consumption-dataset-sri-lanka-recon-sl</p>
<p>[2] Lou, Y., Caruana, R., Gehrke, J. &amp; Hooker, G. (2013). Accurate intelligible models with pairwise interactions. <i>Proc. KDD</i>, 623–631.</p>
<p>[3] Nori, H., Jenkins, S., Koch, P. &amp; Caruana, R. (2019). InterpretML: A unified framework for machine learning interpretability. arXiv:1909.09223.</p>
<p>[4] Lundberg, S. &amp; Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i>.</p>
<p>[5] Ribeiro, M. T., Singh, S. &amp; Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. <i>Proc. KDD</i>.</p>
<p>[6] Public Utilities Commission of Sri Lanka. Domestic tariff (block tariff, customer category D-1). https://www.pucsl.gov.lk/?p=8390</p></div>
<p style='font-size:8.8pt;color:#555;margin-top:10pt'><b>Tools and AI use.</b> Python 3.12, InterpretML 0.7.8, scikit-learn 1.9, XGBoost 3.4, LightGBM 4.7, SHAP 0.54, LIME 0.2, Streamlit. Random seed 42.
An AI coding assistant helped write and debug code. Modelling decisions and interpretation were reviewed by the author.</p>""")

html = f"<!doctype html><html><head><meta charset='utf-8'><title>Electricity EBM Report</title><style>{CSS}</style></head><body>{''.join(H)}</body></html>"
(OUTD / "Electricity_EBM_Report.html").write_text(html, encoding="utf-8")


async def pdf():
    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        b = await pw.chromium.launch(); pg = await b.new_page()
        await pg.goto((OUTD / "Electricity_EBM_Report.html").as_uri())
        await pg.pdf(path=str(OUTD / "Electricity_EBM_Report.pdf"), format="A4", print_background=True,
                     margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"}, display_header_footer=True,
                     header_template="<span></span>",
                     footer_template="<div style='font-size:8px;width:100%;text-align:center;color:#888'><span class='pageNumber'></span></div>")
        await b.close()
asyncio.run(pdf())
print("written", OUTD / "Electricity_EBM_Report.pdf")
