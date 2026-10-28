# Predicting High Electricity Use in Sri Lankan Households with an Explainable Boosting Machine

Applied Machine Learning assignment (MSc in AI, University of Moratuwa). H.M.N.I.K. Herath, 269115L.

**Live app:** https://household-electricity.redwave-d9ae0d05.eastasia.azurecontainerapps.io (cold start: up to 1 minute)
**Report:** `report/Electricity_EBM_Report.pdf`

## Task
Predict, from household survey answers only, whether a household's median monthly use exceeds **90 kWh**
(the PUCSL domestic tariff block boundary). Binary classification. Model: Explainable Boosting Machine (InterpretML),
compared with logistic regression, decision tree, random forest, XGBoost and LightGBM. Explained with EBM, SHAP and LIME.

## Dataset
**ReCon SL: Residential Electricity Consumption Dataset, Sri Lanka** (IEEE DataPort, open access):
https://ieee-dataport.org/open-access/residential-electricity-consumption-dataset-sri-lanka-recon-sl

- 4,063 households, monthly provider readings Oct 2022 – Oct 2024, plus three survey waves.
- Used here: `monthly_consumption.csv` and the eight Wave 1 survey tables, plus `survey_dates.csv`.
- Raw files are not redistributed. Download them and place these files in `data/raw/`:
  `monthly_consumption.csv`, `survey_dates.csv`, `w1_household_information_and_history.csv`,
  `w1_electricity_generation_water_heating_cooking.csv`, `w1_demographics.csv`, `w1_appliances.csv`,
  `w1_ac_roster.csv`, `w1_fan_roster.csv`, `w1_light_roster.csv`, `w1_room_roster.csv`.
- `data/train|val|test.parquet` are the de-identified household-level features used by the models and the app.

## Run
```bash
pip install -r requirements.txt
python 01_prepare.py           # features, target, EDA, 70/10/20 split
python 02_train.py             # 5-fold CV tuning of all models (~10 min on 2 CPUs)
python 03_evaluate_explain.py  # test metrics, bootstrap CIs, EBM/SHAP/LIME, fairness
python report/build_report.py  # report PDF
streamlit run app.py
```

## Deploy
Every push to `main` builds `ghcr.io/<owner>/<repo>:latest` with GitHub Actions (`.github/workflows/docker-publish.yml`).
Azure Container Apps (East Asia, 1 vCPU / 2 GiB, scale to zero) runs that image:
```bash
az containerapp update -n household-electricity -g rg-al-early-warning --image ghcr.io/<owner>/<repo>:latest --revision-suffix v$(date +%m%d%H%M)
```
