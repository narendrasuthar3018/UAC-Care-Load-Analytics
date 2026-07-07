import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="UAC Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better look
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1f77b4;}
    .stMetric {background-color: #e6f3ff; padding: 10px; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("🛡️ UAC System Capacity & Care Load Analytics")
st.markdown("**Developed by: Narendra Suthar | Unified Mentor**")
st.caption("Professional Analytics Dashboard | Real-time Insights & Forecasting")

# df = pd.read_csv('../data/cleaned_uac_data.csv')
df = pd.read_csv('data/cleaned_uac_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

st.sidebar.header("🎛️ Controls")
year_filter = st.sidebar.multiselect("Select Years", [2023,2024,2025], default=[2023,2024,2025])
show_forecast = st.sidebar.checkbox("Show 90-Day Forecast", value=True)

filtered_df = df[df['Year'].isin(year_filter)]

# KPI Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Total Load", f"{filtered_df['Total_System_Load'].mean():,.0f}")
with col2:
    st.metric("Avg HHS Care", f"{filtered_df['Children in HHS Care'].mean():,.0f}")
with col3:
    st.metric("Avg Net Intake", f"{filtered_df['Net_Intake'].mean():.1f}")
with col4:
    st.metric("Anomalies", len(df[df['Anomaly']]) if 'Anomaly' in df.columns else 0)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📊 Comparison", "🔮 Forecasting", "🔴 Anomalies", "📋 Summary"])

with tab1:
    st.subheader("Total System Load Trend")
    fig = px.line(filtered_df, x='Date', y='Total_System_Load', color='Year', title="Total System Load Over Time", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Year-wise Performance")
    yearly = df.groupby('Year').agg({
        'Total_System_Load':'mean',
        'Net_Intake':'mean',
        'Children in HHS Care':'mean'
    }).round(1)
    st.dataframe(yearly, use_container_width=True)

with tab3:
    if show_forecast:
        st.subheader("90-Day Future Load Forecast")
        # (forecast code same as before)
        df_f = df.copy()
        df_f['Days'] = (df_f['Date'] - df_f['Date'].min()).dt.days
        X = df_f[['Days']]
        y = df_f['Total_System_Load']
        model = LinearRegression().fit(X, y)
        future_days = np.array(range(df_f['Days'].max() + 1, df_f['Days'].max() + 91)).reshape(-1, 1)
        future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, 91)]
        future_load = model.predict(future_days)
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(x=df['Date'], y=df['Total_System_Load'], name="Historical"))
        fig_f.add_trace(go.Scatter(x=future_dates, y=future_load, name="Forecast", line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_f, use_container_width=True)

with tab4:
    st.subheader("Anomaly Detection")
    df['Rolling_Mean'] = df['Total_System_Load'].rolling(7).mean()
    df['Rolling_Std'] = df['Total_System_Load'].rolling(7).std()
    df['Anomaly'] = abs(df['Total_System_Load'] - df['Rolling_Mean']) > 2 * df['Rolling_Std']
    st.plotly_chart(px.scatter(df, x='Date', y='Total_System_Load', color='Anomaly', title="Anomalies in System Load"), use_container_width=True)

with tab5:
    st.subheader("Executive Summary")
    st.write("The UAC care system has shown significant improvement in 2025 with reduced load and better stability.")
    st.write("Recommendation: Implement this dashboard for continuous monitoring.")

st.success("Professional Dashboard by Narendra Suthar | Unified Mentor")