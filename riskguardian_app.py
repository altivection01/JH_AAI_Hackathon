import streamlit as st, pandas as pd, numpy as np, joblib
st.set_page_config(page_title='RiskGuardian', layout='wide')
st.title('RiskGuardian — Cyber Risk Projection')
art = joblib.load('riskguardian_model.joblib'); df = pd.read_parquet('riskguardian_scored.parquet')
ind = st.sidebar.multiselect('Industry', sorted(df.industry.unique()), default=sorted(df.industry.unique()))
view = df[df.industry.isin(ind)]
c1, c2, c3 = st.columns(3)
c1.metric('Assets', len(view))
c2.metric('Mean RES', round(view.RES.mean(), 3))
c3.metric('Critical-tier assets', int((view.RES_tier=='Critical').sum()))
st.subheader('Risk mix by industry')
st.bar_chart(pd.crosstab(view.industry, view.risk_class))
st.subheader('Top assets to action')
st.dataframe(view.sort_values('RES', ascending=False).head(20))