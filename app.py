"""Interactive actuarial dashboard. All figures derived from SIMULATED data."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title='Medical Cost Forecasting Accelerator',layout='wide')
st.title('Medical Cost Forecasting Accelerator')
st.warning('DEMONSTRATION — wholly simulated 2023–2025 member and claims data; not actual CMS 2023–2025 data. Predictions are not validated for insurance pricing or patient decisions.')
folder=Path('artifacts')
if not (folder/'predictions.parquet').exists():
    st.info('Run `python -m src.data`, `python -m src.features`, and `python -m src.train` to create demo artifacts.')
    st.stop()

@st.cache_data
def load():
    return pd.read_parquet(folder/'predictions.parquet'),json.loads((folder/'metrics.json').read_text()),pd.read_csv(folder/'shap_importance.csv')

df,report,shap_importance=load()
st.sidebar.header('Population filters')
regions=st.sidebar.multiselect('Region',sorted(df.region.unique()),default=sorted(df.region.unique()))
minage,maxage=st.sidebar.slider('Age',int(df.age.min()),int(df.age.max()),(int(df.age.min()),int(df.age.max())))
risk=st.sidebar.multiselect('Risk segment',sorted(df.risk_tier.unique()),default=sorted(df.risk_tier.unique()))
f=df[df.region.isin(regions)&df.age.between(minage,maxage)&df.risk_tier.isin(risk)]
if f.empty:st.warning('No members match the filters.');st.stop()
selected=report['selected_by_wape']
st.caption(f'Model selected by 2025 out-of-time WAPE: {selected}. Explanations below are for XGBoost, regardless of selected model.')
a,b,c,d=st.columns(4)
a.metric('Members',f'{len(f):,}')
b.metric('Predicted PMPM',f'${f.predicted_paid.sum()/f.next_year_months.sum():,.0f}')
c.metric('Actual PMPM (simulated)',f'${f.next_year_paid.sum()/f.next_year_months.sum():,.0f}')
d.metric('Forecast bias',f'{(f.predicted_paid.sum()/max(f.next_year_paid.sum(),1)-1)*100:+.1f}%')
st.subheader('Risk segmentation and predicted spend')
col1,col2=st.columns(2)
seg=f.groupby('risk_tier',as_index=False).agg(members=('member_id','size'),predicted_paid=('predicted_paid','sum'),actual_paid=('next_year_paid','sum'))
order=['Lower 50%','Elevated 40%','Top 10%']
with col1:st.plotly_chart(px.bar(seg,x='risk_tier',y='predicted_paid',category_orders={'risk_tier':order},title='Predicted spend by risk segment'),use_container_width=True)
with col2:st.plotly_chart(px.scatter(f,x='predicted_paid',y='next_year_paid',color='risk_tier',opacity=.45,title='Predicted vs actual annual paid amount'),use_container_width=True)
st.subheader('Model performance — held-out 2025 target')
st.dataframe(pd.DataFrame(report['model_metrics']).T.style.format(precision=3),use_container_width=True)
st.caption('WAPE and aggregate bias are computed on annual paid amounts. Top-decile capture uses true spending available only for retrospective evaluation. Top decile is defined on the entire test cohort before interactive filtering.')
st.subheader('Explainability — XGBoost')
st.plotly_chart(px.bar(shap_importance.head(14).sort_values('mean_abs_shap_log1p_paid'),x='mean_abs_shap_log1p_paid',y='feature',orientation='h',title='Mean absolute SHAP in log1p(annual paid amount) model output'),use_container_width=True)
st.caption('Global SHAP values are predictive associations, not causal cost drivers. Because XGBoost was trained against log1p(spending), SHAP values use that transformed target scale.')
st.subheader('Population drill-down')
st.dataframe(f[['member_id','region','age','chronic_count','risk_tier','predicted_paid','next_year_paid']].sort_values('predicted_paid',ascending=False).head(200),use_container_width=True,hide_index=True)
st.download_button('Download filtered predictions (CSV)',f.to_csv(index=False),'filtered_predictions.csv','text/csv')
