# Medical Cost Forecasting Accelerator Dashboard

A **two-day, portfolio-scale actuarial AI/ML proof of concept**: transparent synthetic 2023–2025 claims, member-year features, GLM baseline, XGBoost challenger, actuarial validation, SHAP, top-decile risk segmentation, and a Streamlit dashboard ready for GCP Cloud Run.

> **IMPORTANT DATA PROVENANCE:** The CMS DE-SynPUF was created for **2008–2010**. The modern 2023–2025 dataset generated here is **entirely simulated by this repository**, NOT actual CMS claims for 2023–2025, NOT an official CMS forward projection, and NOT based on real contemporary utilization. CMS files may optionally be downloaded to `data/raw/` for separate **schema and null-rate profiling**. The default synthetic generator uses illustrative assumptions rather than fitting to CMS distributions. All outputs are educational only and must not be used for clinical, financial, pricing, or insurance decisions.

CMS: https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf

## Local quickstart (Python 3.11)

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m src.data --members 5000 --seed 42
python -m src.features
python -m src.train
streamlit run app.py
pytest -q
```

Generating 5,000 members produces multiple years of simulated claims. For a quicker smoke test, use `--members 500`. 

## CMS source profiling (optional, independent of demo dataset)

1. Visit the CMS DE-SynPUF page linked above. Select a small sample (e.g. sample 1), then download beneficiary summary and claims ZIPs into `data/raw/`.
2. Run `python -m src.profile_cms --input data/raw`.
3. Review `artifacts/cms_profile.json` for headers, sample null proportions, and file-level errors. Profiling uses only the first 10,000 CSV rows/file and is **not population prevalence estimation**.
4. Do **not** change the official CMS data dates to 2023–2025; doing so would falsely represent the dataset's provenance.

No automatic CMS download is attempted: CMS release links and accessibility can change. Generated data never contains PHI.

## Architecture

```text
Official CMS 2008–2010 ZIPs (optional) -> sample schema profile
Synthetic 2023–2025 generator -> Parquet claims and enrollment
   -> member-year features at Dec 31 of feature year
   -> 2024 labels (train) / 2025 labels (out-of-time test)
   -> Tweedie GLM baseline and XGBoost regression
   -> actuarial evaluation + SHAP importance
   -> full-cohort ranked risk deciles and top-decile capture
   -> Streamlit dashboard / Cloud Run
```

## Features and target

- Demographics: age, sex, region; five simulated chronic-condition flags and chronic count
- Prior annual paid amount and PMPM, prior claim count
- Prior utilization and paid dollars by inpatient, outpatient, professional, and pharmacy categories
- Member-months exposure (12/months per year in MVP)
- Target: **next calendar year's total paid amount**; target PMPM calculated with next year's member-months

All prior-year utilization features are lagged; the 2025 target cannot be used when predicting it. Time-based validation trains against **2024 costs** and tests against **2025 costs**. XGBoost is fit to `log1p(cost)` and predictions are inverse-transformed. GLM is a Tweedie(log) baseline. The model is **not** rate setting or comprehensive actuarial reserving.

## Actuarial evaluation

- WAPE = `sum(abs(actual-predicted)) / sum(actual)`
- Aggregate forecast bias % = `100*(sum(predicted)-sum(actual))/sum(actual)`
- PMPM = `sum(total paid)/sum(member_months)`
- Top-decile precision = fraction of actual top-decile spenders in predicted top 10% of members
- Top-decile recall = share of actual top-decile members captured in predicted top 10%
- Top-decile spend capture = realized spending among predicted top 10% / total realized spending

Model selection uses held-out WAPE for demonstration only; choosing and reporting on the same test set introduces selection bias. A production implementation needs a separate validation set, untouched final test set, multiple rolling-origin years, and statistical uncertainty bounds.

**Explainability:** `artifacts/shap_importance.csv` describes mean absolute Tree SHAP contributions to XGBoost's **log1p(cost)** prediction. SHAP describes model behavior, not causality. Sensitive traits and subgroup model errors warrant explicit fairness and governance reviews before real deployment.

## GCP deployment (only after generating artifacts locally)

The Streamlit dashboard only reads generated demo artifacts; predictions are not computed inside Cloud Run.

```bash
# Login, choose a billing-enabled project, and select a region
# gcloud auth login
# gcloud config set project YOUR_PROJECT_ID
# gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
# gcloud run deploy medical-cost-forecast-dashboard --source . --region us-central1 --allow-unauthenticated
```

`artifacts/` is ignored by Git. For deployment, build from a **local folder with generated artifacts** (Dockerfile COPY will include them unless ignored), or implement secure Google Cloud Storage loading. Public Cloud Run access exposes the synthetic demo data and is **only suitable for non-sensitive data**. Cloud Run, builds, storage and networking may incur charges; verify budgets and stop unused services.

For GCP-native storage, upload generated Parquet to Cloud Storage and load it into BigQuery; this is an optional extension, not required to run the dashboard. No service-account JSON keys should be committed.

## Known limitations

- Simulated distributions are invented; not calibrated to CMS or contemporary medical costs.
- Three generated years enable only one 2024 training target and one 2025 test target. Generalization cannot be established.
- All members have full-year enrollment, a simplification that should be replaced with enrollment spells.
- No inflation adjustment beyond the generator's assumed trend; no clinical accuracy or causal claims.
- The ranked deciles are created on the complete test population before dashboard filters, so the filtered top 10% may not represent 10% of filtered members.
- The dashboard uses a selected best-WAPE model, but SHAP always explains the XGBoost challenger explicitly.

## Repo contents

- `src/profile_cms.py`: optional CMS ZIP/CSV schema profiling
- `src/data.py`: reproducible transparent synthetic claim generator
- `src/features.py`: member-year, leakage-safe feature table
- `src/train.py`: baseline and XGBoost, actuarial metrics, SHAP and rankings
- `app.py`: Streamlit dashboard
- `tests/`: fundamental temporal leakage and metric checks
- `Dockerfile`: Cloud Run deployment
