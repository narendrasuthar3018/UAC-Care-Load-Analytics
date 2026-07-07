import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

import numpy as np

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import warnings


MONTH_ORDER = [
    "January","February","March",
    "April","May","June",
    "July","August","September",
    "October","November","December"
]

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="UAC Analytics Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""

<style>

html, body, [class*="css"]{
    background:#0f172a;
    color:white;
}

.block-container{
    padding-top:1rem;
}

.main-title{

    font-size:40px;
    font-weight:bold;
    text-align:center;
    color:#00E5FF;

}

.subtitle{

    text-align:center;
    color:#A5B4FC;
    font-size:18px;
    margin-bottom:25px;

}

.metric-card{

    background:#1e293b;
    padding:18px;
    border-radius:14px;
    border:1px solid #334155;

}

div[data-testid="metric-container"]{

    background:#172033;
    border:1px solid #334155;
    border-radius:12px;
    padding:10px;

}

footer{

    visibility:hidden;

}

header{

    visibility:hidden;

}

</style>

""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<div class='main-title'>🛡️ UAC CARE SYSTEM ANALYTICS</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Advanced Capacity Monitoring & Intelligence Platform</div>",
    unsafe_allow_html=True
)

st.caption("Developed by Narendra Suthar | Unified Mentor")


# --------------------------------------------------
# LOAD DATA
# (File Path Same)
# --------------------------------------------------

@st.cache_data

@st.cache_data
def load_data():

    BASE_DIR = Path(__file__).resolve().parent.parent

    DATA_FILE = BASE_DIR / "data" / "cleaned_uac_data.csv"

    df = pd.read_csv(DATA_FILE)

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year

    return df

df = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙ Dashboard Controls")

years = sorted(df["Year"].unique())

selected_year = st.sidebar.multiselect(
    "Select Year",
    years,
    default=years
)

months = list(df["Month"].unique())

selected_month = st.sidebar.multiselect(
    "Select Month",
    months,
    default=months
)

quarters = sorted(df["Quarter"].unique())

selected_quarter = st.sidebar.multiselect(
    "Select Quarter",
    quarters,
    default=quarters
)

metric_view = st.sidebar.selectbox(

    "Primary Metric",

    [
        "Total_System_Load",
        "Children in HHS Care",
        "Net_Intake",
        "7d_Rolling_HHS"
    ]

)

show_forecast = st.sidebar.checkbox(
    "Enable Forecast",
    value=True
)

show_anomaly = st.sidebar.checkbox(
    "Enable Risk Detection",
    value=True
)

show_rolling = st.sidebar.checkbox(
    "Show Rolling Average",
    value=True
)


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered_df = df[
    (df["Year"].isin(selected_year))
    &
    (df["Month"].isin(selected_month))
    &
    (df["Quarter"].isin(selected_quarter))
].copy()


# --------------------------------------------------
# LAST UPDATE
# --------------------------------------------------

colA,colB=st.columns([4,1])

with colA:

    st.info(
        f"Last Updated : {filtered_df['Date'].max().strftime('%d %b %Y')}"
    )

with colB:

    st.success(datetime.now().strftime("%H:%M"))


# --------------------------------------------------
# KPI SECTION
# --------------------------------------------------
if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

if len(filtered_df) < 2:
    st.warning("Please select at least 2 records.")
    st.stop()
latest = filtered_df.iloc[-1]

previous = filtered_df.iloc[-2]


col1,col2,col3,col4,col5=st.columns(5)

with col1:

    st.metric(

        "Current Load",

        f"{latest['Total_System_Load']:,.0f}",

        f"{latest['Total_System_Load']-previous['Total_System_Load']:.0f}"

    )

with col2:

    st.metric(

        "Children in HHS Care",

        f"{latest['Children in HHS Care']:,.0f}",

        f"{latest['Children in HHS Care']-previous['Children in HHS Care']:.0f}"

    )

with col3:

    st.metric(

        "Net Intake",

        f"{latest['Net_Intake']:,.0f}",

        f"{latest['Net_Intake']-previous['Net_Intake']:.0f}"

    )

with col4:

    st.metric(

        "Peak Load",

        f"{filtered_df['Total_System_Load'].max():,.0f}"

    )

with col5:

    stability = round(
        filtered_df["Total_System_Load"].pct_change().std()*100,
        2
    )

    st.metric(

        "Volatility",

        f"{stability}%"

    )


# --------------------------------------------------
# EXTRA KPI
# --------------------------------------------------

k1,k2,k3,k4=st.columns(4)

k1.info(f"Average Load : {filtered_df['Total_System_Load'].mean():,.0f}")

k2.success(f"Highest Intake : {filtered_df['Net_Intake'].max():,.0f}")

k3.warning(f"Lowest Intake : {filtered_df['Net_Intake'].min():,.0f}")

k4.error(f"Observations : {len(filtered_df)}")


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs(

[
"📈 Main Trends",

"📊 Comparison",

"📉 Advanced Charts",

"🔥 Heatmaps",

"🔮 Forecast",

"🚨 Risk Analysis",

"📋 Executive Insights",

"📥 Export"

]
)
# ======================================================
# TAB 1 : MAIN TRENDS
# ======================================================

with tab1:

    st.subheader("📈 Real-Time System Trend Dashboard")

    chart1, chart2 = st.columns([3, 2])

    with chart1:

        fig = px.line(
            filtered_df,
            x="Date",
            y=metric_view,
            color="Year",
            template="plotly_dark",
            markers=True,
            title=f"{metric_view} Trend"
        )

        fig.update_layout(
            hovermode="x unified",
            height=500,
            legend_title="Year"
        )

        fig.update_xaxes(rangeslider_visible=True)

        if show_rolling:

            rolling = filtered_df[metric_view].rolling(7).mean()

            fig.add_trace(

                go.Scatter(

                    x=filtered_df["Date"],

                    y=rolling,

                    name="7-Day Rolling Avg",

                    line=dict(color="yellow", width=3)

                )

            )

        st.plotly_chart(fig, use_container_width=True)

    with chart2:

        st.subheader("📌 Summary")

        st.metric(
            "Highest Value",
            f"{filtered_df[metric_view].max():,.0f}"
        )

        st.metric(
            "Lowest Value",
            f"{filtered_df[metric_view].min():,.0f}"
        )

        st.metric(
            "Average",
            f"{filtered_df[metric_view].mean():,.2f}"
        )

        st.metric(
            "Median",
            f"{filtered_df[metric_view].median():,.2f}"
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("📊 Area Trend")

        area = px.area(

            filtered_df,

            x="Date",

            y=metric_view,

            color="Year",

            template="plotly_dark"

        )

        area.update_layout(height=450)

        st.plotly_chart(area, use_container_width=True)

    with right:

        st.subheader("📊 Monthly Distribution")

        month_avg = (

            filtered_df

            .groupby("Month")[metric_view]

            .mean()

            .reset_index()

        )
        month_avg["Month"] = pd.Categorical(
            month_avg["Month"],
            categories=MONTH_ORDER,
            ordered=True
)

        month_avg = month_avg.sort_values("Month")

        bar = px.bar(

            month_avg,

            x="Month",

            y=metric_view,

            color=metric_view,

            template="plotly_dark"

        )

        bar.update_layout(height=450)

        st.plotly_chart(bar, use_container_width=True)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("📅 Quarterly Trend")

        quarter_df = (

            filtered_df

            .groupby("Quarter")[metric_view]

            .mean()

            .reset_index()

        )

        qfig = px.line(

            quarter_df,

            x="Quarter",

            y=metric_view,

            markers=True,

            template="plotly_dark"

        )

        st.plotly_chart(qfig, use_container_width=True)

    with c2:

        st.subheader("📈 Daily Change")

        daily = filtered_df.copy()

        daily["Change"] = daily[metric_view].diff()

        dfig = px.bar(

            daily,

            x="Date",

            y="Change",

            color="Change",

            template="plotly_dark"

        )

        st.plotly_chart(dfig, use_container_width=True)
        # ======================================================
# TAB 2 : COMPARISON DASHBOARD
# ======================================================

with tab2:

    st.subheader("📊 Multi-Year Performance Comparison")

    yearly = (
        filtered_df
        .groupby("Year")
        .mean(numeric_only=True)
        .reset_index()
    )

    col1, col2 = st.columns(2)

    with col1:

        fig_year = px.bar(
            yearly,
            x="Year",
            y="Total_System_Load",
            color="Year",
            text_auto=".2s",
            template="plotly_dark",
            title="Average Total System Load"
        )

        fig_year.update_layout(height=450)

        st.plotly_chart(fig_year, use_container_width=True)

    with col2:

        fig_hhs = px.bar(
            yearly,
            x="Year",
            y="Children in HHS Care",
            color="Year",
            text_auto=".2s",
            template="plotly_dark",
            title="Average Children in HHS Care"
        )

        fig_hhs.update_layout(height=450)

        st.plotly_chart(fig_hhs, use_container_width=True)


    st.divider()


    left, right = st.columns(2)

    with left:

        monthly = (
            filtered_df
            .groupby(["Year","Month"])[metric_view]
            .mean()
            .reset_index()
        )

        month_avg["Month"] = pd.Categorical(
             month_avg["Month"],
            categories=MONTH_ORDER,
            ordered=True
        )

        monthly = monthly.sort_values("Month")

        fig_month = px.line(

            monthly,

            x="Month",

            y=metric_view,

            color="Year",

            markers=True,

            template="plotly_dark",

            title="Monthly Comparison"

        )

        fig_month.update_layout(height=500)

        st.plotly_chart(fig_month, use_container_width=True)


    with right:

        quarter = (

            filtered_df

            .groupby(["Year","Quarter"])[metric_view]

            .mean()

            .reset_index()

        )

        fig_quarter = px.bar(

            quarter,

            x="Quarter",

            y=metric_view,

            color="Year",

            barmode="group",

            template="plotly_dark",

            title="Quarter Comparison"

        )

        fig_quarter.update_layout(height=500)

        st.plotly_chart(fig_quarter, use_container_width=True)


    st.divider()


    st.subheader("📈 Growth Analysis")

    growth = yearly.copy()

    growth["Growth %"] = growth["Total_System_Load"].pct_change()*100

    growth["Growth %"] = growth["Growth %"].fillna(0)

    fig_growth = px.bar(

        growth,

        x="Year",

        y="Growth %",

        color="Growth %",

        text_auto=".2f",

        template="plotly_dark",

        title="Year-over-Year Growth (%)"

    )

    fig_growth.update_layout(height=450)

    st.plotly_chart(fig_growth, use_container_width=True)


    st.divider()


    st.subheader("📋 Comparison Summary")

    summary = yearly.round(2)

    st.dataframe(

        summary.style.highlight_max(axis=0),

        use_container_width=True

    )


    csv = summary.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Comparison CSV",

        csv,

        file_name="comparison_summary.csv",

        mime="text/csv"

    )
    use_container_width=True
    # ======================================================
# TAB 3 : ADVANCED CHARTS
# ======================================================

with tab3:

    st.subheader("📉 Advanced Data Visualization")

    row1_col1, row1_col2 = st.columns(2)

    # --------------------------------------------------
    # Histogram
    # --------------------------------------------------

    with row1_col1:

        st.markdown("### 📊 Histogram")

        fig_hist = px.histogram(

            filtered_df,

            x=metric_view,

            nbins=30,

            color_discrete_sequence=["#00D9FF"],

            template="plotly_dark"

        )

        fig_hist.update_layout(height=450)

        st.plotly_chart(fig_hist,use_container_width=True)


    # --------------------------------------------------
    # Box Plot
    # --------------------------------------------------

    with row1_col2:

        st.markdown("### 📦 Box Plot")

        fig_box = px.box(

            filtered_df,

            y=metric_view,

            color="Year",

            template="plotly_dark"

        )

        fig_box.update_layout(height=450)

        st.plotly_chart(fig_box,use_container_width=True)


    st.divider()

    row2_col1,row2_col2=st.columns(2)

    # --------------------------------------------------
    # Pie Chart
    # --------------------------------------------------

    with row2_col1:

        st.markdown("### 🥧 Year Contribution")

        pie_df=(

            filtered_df

            .groupby("Year")[metric_view]

            .sum()

            .reset_index()

        )

        fig_pie=px.pie(

            pie_df,

            names="Year",

            values=metric_view,

            hole=.45,

            template="plotly_dark"

        )

        fig_pie.update_traces(textposition="inside")

        fig_pie.update_layout(height=450)

        st.plotly_chart(fig_pie,use_container_width=True)


    # --------------------------------------------------
    # Violin Plot
    # --------------------------------------------------

    with row2_col2:

        st.markdown("### 🎻 Distribution")

        fig_violin=px.violin(

            filtered_df,

            y=metric_view,

            color="Year",

            box=True,

            points="all",

            template="plotly_dark"

        )

        fig_violin.update_layout(height=450)

        st.plotly_chart(fig_violin,use_container_width=True)


    st.divider()

    # --------------------------------------------------
    # Scatter Plot
    # --------------------------------------------------

    st.markdown("### 🔵 Relationship Analysis")

    scatter=px.scatter(

        filtered_df,

        x="Net_Intake",

        y="Total_System_Load",

        color="Year",

        size="Children in HHS Care",

        hover_data=["Date"],

        template="plotly_dark"

    )

    scatter.update_layout(height=550)

    st.plotly_chart(scatter,use_container_width=True)


    st.divider()

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    st.markdown("### 📋 Statistical Summary")

    stats=filtered_df[[

        "Total_System_Load",

        "Children in HHS Care",

        "Net_Intake"

    ]].describe().round(2)

    st.dataframe(stats,use_container_width=True)


    st.divider()

    # --------------------------------------------------
    # Outlier Detection
    # --------------------------------------------------

    st.markdown("### 🚨 Outlier Detection")

    q1=filtered_df[metric_view].quantile(.25)

    q3=filtered_df[metric_view].quantile(.75)

    iqr=q3-q1

    lower=q1-1.5*iqr

    upper=q3+1.5*iqr

    outliers=filtered_df[

        (filtered_df[metric_view]<lower)

        |

        (filtered_df[metric_view]>upper)

    ]

    st.metric(

        "Outliers Found",

        len(outliers)

    )

    st.dataframe(

        outliers,

        use_container_width=True

    )
    # ======================================================
# TAB 4 : HEATMAPS & DATA QUALITY
# ======================================================

with tab4:

    st.subheader("🔥 Heatmaps & Data Quality Analysis")

    # ----------------------------------------
    # Correlation Heatmap
    # ----------------------------------------

    st.markdown("### 📊 Correlation Heatmap")

    corr_cols = [
        "Total_System_Load",
        "Children in HHS Care",
        "Net_Intake"
    ]

    corr = filtered_df[corr_cols].corr()

    fig_corr = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        template="plotly_dark",

        aspect="auto"

    )

    fig_corr.update_layout(height=500)

    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ----------------------------------------
    # Monthly Heatmap
    # ----------------------------------------

    st.markdown("### 📅 Monthly Heatmap")

    heat = (

        filtered_df

        .pivot_table(

            values=metric_view,

            index="Month",

            columns="Year",

            aggfunc="mean"

        )

    )

    month_order = [

        "January","February","March",

        "April","May","June",

        "July","August","September",

        "October","November","December"

    ]

    heat = heat.reindex(month_order)

    fig_heat = px.imshow(

        heat,

        color_continuous_scale="Viridis",

        text_auto=".1f",

        aspect="auto",

        template="plotly_dark"

    )

    fig_heat.update_layout(height=600)

    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ----------------------------------------
    # Missing Values
    # ----------------------------------------

    st.markdown("### 📋 Missing Value Analysis")

    missing = filtered_df.isnull().sum()

    missing_df = pd.DataFrame({

        "Column": missing.index,

        "Missing": missing.values

    })

    fig_missing = px.bar(

        missing_df,

        x="Column",

        y="Missing",

        color="Missing",

        template="plotly_dark"

    )

    st.plotly_chart(fig_missing, use_container_width=True)

    st.divider()

    # ----------------------------------------
    # Data Quality
    # ----------------------------------------

    st.markdown("### ✅ Data Quality Summary")

    total_cells = filtered_df.shape[0] * filtered_df.shape[1]

    missing_cells = filtered_df.isnull().sum().sum()

    quality = ((total_cells - missing_cells) / total_cells) * 100

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(

            "Data Quality",

            f"{quality:.2f}%"

        )

    with c2:

        st.metric(

            "Rows",

            len(filtered_df)

        )

    with c3:

        st.metric(

            "Columns",

            len(filtered_df.columns)

        )

    st.divider()

    # ----------------------------------------
    # Data Preview
    # ----------------------------------------

    st.markdown("### 👀 Data Preview")

    st.dataframe(

        filtered_df.tail(20),

        use_container_width=True

    )
    # ======================================================
# TAB 5 : ADVANCED FORECAST
# ======================================================

with tab5:

    st.subheader("🔮 90-Day Predictive Forecast")

    if show_forecast:

        forecast_df = filtered_df.copy()

        forecast_df = forecast_df.sort_values("Date")

        forecast_df["Days"] = (
            forecast_df["Date"] - forecast_df["Date"].min()
        ).dt.days

        X = forecast_df[["Days"]]

        y = forecast_df["Total_System_Load"]

        # Polynomial Regression (Better than Linear)

        model = make_pipeline(

            PolynomialFeatures(degree=2),

            LinearRegression()

        )
        if len(forecast_df) < 10:
          st.warning("Not enough data for forecasting.")
          st.stop()

        model.fit(X, y)

        future_days = np.arange(

            forecast_df["Days"].max() + 1,

            forecast_df["Days"].max() + 91

        )

        future_dates = pd.date_range(

            forecast_df["Date"].max() + timedelta(days=1),

            periods=90

        )

        prediction = model.predict(

            future_days.reshape(-1,1)

        )

        future = pd.DataFrame({

            "Date":future_dates,

            "Prediction":prediction

        })

        # Confidence Interval

        std = np.std(y)

        future["Upper"] = future["Prediction"] + std

        future["Lower"] = future["Prediction"] - std

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=forecast_df["Date"],

                y=forecast_df["Total_System_Load"],

                name="Historical",

                line=dict(color="cyan",width=3)

            )

        )

        fig.add_trace(

            go.Scatter(

                x=future["Date"],

                y=future["Prediction"],

                name="Forecast",

                line=dict(color="red",dash="dash")

            )

        )

        fig.add_trace(

            go.Scatter(

                x=future["Date"],

                y=future["Upper"],

                line=dict(width=0),

                showlegend=False

            )

        )

        fig.add_trace(

            go.Scatter(

                x=future["Date"],

                y=future["Lower"],

                fill="tonexty",

                fillcolor="rgba(255,0,0,0.15)",

                line=dict(width=0),

                name="Confidence Band"

            )

        )

        fig.update_layout(

            template="plotly_dark",

            height=600,

            hovermode="x unified"

        )

        st.plotly_chart(fig,use_container_width=True)

        st.divider()

        c1,c2,c3,c4=st.columns(4)

        with c1:

            st.metric(

                "Current Load",

                f"{forecast_df['Total_System_Load'].iloc[-1]:,.0f}"

            )

        with c2:

            st.metric(

                "Forecast Average",

                f"{future['Prediction'].mean():,.0f}"

            )

        with c3:

            st.metric(

                "Forecast Max",

                f"{future['Prediction'].max():,.0f}"

            )

        with c4:

            growth=(

                (future["Prediction"].iloc[-1]-forecast_df["Total_System_Load"].iloc[-1])

                /

                forecast_df["Total_System_Load"].iloc[-1]

            )*100

            st.metric(

                "Expected Growth",

                f"{growth:.2f}%"

            )

        st.divider()

        st.subheader("📋 Next 90 Days Forecast")

        display = future.copy()

        display["Prediction"] = display["Prediction"].round(2)

        display["Upper"] = display["Upper"].round(2)

        display["Lower"] = display["Lower"].round(2)

        st.dataframe(

            display,

            use_container_width=True

        )

        csv = display.to_csv(index=False).encode()

        st.download_button(

            "⬇ Download Forecast CSV",

            csv,

            file_name="forecast.csv",

            mime="text/csv"

        )
        # ======================================================
# TAB 6 : RISK & ANOMALY ANALYSIS
# ======================================================

with tab6:

    st.subheader("🚨 Risk & Anomaly Detection Dashboard")

    if show_anomaly:

        risk_df = filtered_df.copy()

        # ----------------------------------------
        # Rolling Statistics
        # ----------------------------------------

        risk_df["Rolling_Mean"] = (
            risk_df["Total_System_Load"]
            .rolling(7)
            .mean()
        )

        risk_df["Rolling_STD"] = (
            risk_df["Total_System_Load"]
            .rolling(7)
            .std()
        )

        risk_df["Upper"] = (
            risk_df["Rolling_Mean"] +
            (2 * risk_df["Rolling_STD"])
        )

        risk_df["Lower"] = (
            risk_df["Rolling_Mean"] -
            (2 * risk_df["Rolling_STD"])
        )
        risk_df["Rolling_STD"] = risk_df["Rolling_STD"].replace(0, np.nan)

        risk_df["Anomaly"] = (

            (risk_df["Total_System_Load"] > risk_df["Upper"])

            |

            (risk_df["Total_System_Load"] < risk_df["Lower"])

        )

        # ----------------------------------------
        # Risk Score
        # ----------------------------------------

        risk_df["Risk Score"] = (

            abs(

                risk_df["Total_System_Load"]

                -

                risk_df["Rolling_Mean"]

            )

            /

            risk_df["Rolling_STD"]

        )

        risk_df["Risk Score"] = risk_df["Risk Score"].fillna(0)

        # ----------------------------------------
        # Severity
        # ----------------------------------------

        def severity(score):

            if score < 1:

                return "Low"

            elif score < 2:

                return "Medium"

            else:

                return "High"

        risk_df["Severity"] = risk_df["Risk Score"].apply(severity)

        # ----------------------------------------
        # KPI
        # ----------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(

            "Total Risk Days",

            int(risk_df["Anomaly"].sum())

        )

        c2.metric(

            "Highest Risk Score",

            f"{risk_df['Risk Score'].max():.2f}"

        )

        c3.metric(

            "Average Risk",

            f"{risk_df['Risk Score'].mean():.2f}"

        )

        c4.metric(

            "Maximum Load",

            f"{risk_df['Total_System_Load'].max():,.0f}"

        )

        st.divider()

        # ----------------------------------------
        # Risk Trend
        # ----------------------------------------

        fig = go.Figure()

        fig.add_trace(

            go.Scatter(

                x=risk_df["Date"],

                y=risk_df["Total_System_Load"],

                name="Load",

                line=dict(color="cyan")

            )

        )

        fig.add_trace(

            go.Scatter(

                x=risk_df["Date"],

                y=risk_df["Rolling_Mean"],

                name="Rolling Mean",

                line=dict(color="yellow")

            )

        )

        fig.add_trace(

            go.Scatter(

                x=risk_df["Date"],

                y=risk_df["Upper"],

                name="Upper Limit",

                line=dict(color="red",dash="dash")

            )

        )

        fig.add_trace(

            go.Scatter(

                x=risk_df["Date"],

                y=risk_df["Lower"],

                name="Lower Limit",

                line=dict(color="green",dash="dash")

            )

        )

        fig.update_layout(

            template="plotly_dark",

            height=550

        )

        st.plotly_chart(fig,use_container_width=True)

        st.divider()

        # ----------------------------------------
        # Scatter
        # ----------------------------------------

        scatter = px.scatter(

            risk_df,

            x="Date",

            y="Total_System_Load",

            color="Severity",

            size="Risk Score",

            hover_data=[

                "Risk Score",

                "Severity"

            ],

            template="plotly_dark"

        )

        scatter.update_layout(height=500)

        st.plotly_chart(scatter,use_container_width=True)

        st.divider()

        # ----------------------------------------
        # Top Risk Days
        # ----------------------------------------

        st.subheader("⚠ Top 10 Highest Risk Days")

        top = (

            risk_df

            .sort_values(

                "Risk Score",

                ascending=False

            )

            .head(10)

        )

        st.dataframe(

            top[

                [

                    "Date",

                    "Total_System_Load",

                    "Risk Score",

                    "Severity"

                ]

            ],

            use_container_width=True

        )

        st.divider()

        # ----------------------------------------
        # Risk Distribution
        # ----------------------------------------

        dist = (

            risk_df

            .groupby("Severity")

            .size()

            .reset_index(name="Count")

        )

        pie = px.pie(

            dist,

            names="Severity",

            values="Count",

            hole=.45,

            template="plotly_dark"

        )

        st.plotly_chart(

            pie,

            use_container_width=True

        )
        # ======================================================
# TAB 7 : EXECUTIVE INSIGHTS
# ======================================================

with tab7:

    st.subheader("📋 Executive Intelligence Dashboard")
    if filtered_df.empty:
        st.warning("No data available for selected filters.")
        st.stop()

if len(filtered_df) < 2:
    st.warning("Please select at least 2 records.")
    st.stop()

if len(filtered_df) < 2:
    st.warning("Please select at least 2 records.")
    st.stop()
    latest = filtered_df.iloc[-1]

    highest = filtered_df.loc[
        filtered_df["Total_System_Load"].idxmax()
    ]

    lowest = filtered_df.loc[
        filtered_df["Total_System_Load"].idxmin()
    ]

    monthly = (

        filtered_df

        .groupby("Month")["Total_System_Load"]

        .mean()

        .reset_index()

    )

    month_order = [

        "January","February","March",

        "April","May","June",

        "July","August","September",

        "October","November","December"

    ]

    monthly["Month"] = pd.Categorical(

        monthly["Month"],

        categories=month_order,

        ordered=True

    )

    monthly = monthly.sort_values("Month")

    best_month = monthly.loc[
        monthly["Total_System_Load"].idxmax()
    ]

    worst_month = monthly.loc[
        monthly["Total_System_Load"].idxmin()
    ]

    avg_load = filtered_df["Total_System_Load"].mean()

    avg_hhs = filtered_df["Children in HHS Care"].mean()

    avg_intake = filtered_df["Net_Intake"].mean()

    utilization = (

        avg_hhs /

        avg_load

    ) * 100

    growth = (

        (

            filtered_df["Total_System_Load"].iloc[-1]

            -

            filtered_df["Total_System_Load"].iloc[0]

        )

        /

        filtered_df["Total_System_Load"].iloc[0]

    ) * 100


    # -------------------------------------------------
    # KPI
    # -------------------------------------------------

    c1,c2,c3,c4=st.columns(4)

    c1.metric(

        "Average Load",

        f"{avg_load:,.0f}"

    )

    c2.metric(

        "Average HHS Care",

        f"{avg_hhs:,.0f}"

    )

    c3.metric(

        "Average Intake",

        f"{avg_intake:,.0f}"

    )

    c4.metric(

        "Growth",

        f"{growth:.2f}%"

    )

    st.divider()


    # -------------------------------------------------
    # Executive Summary
    # -------------------------------------------------

    st.success(f"""

### Executive Summary

• Current System Load : **{latest['Total_System_Load']:,.0f}**

• Highest Load Recorded : **{highest['Total_System_Load']:,.0f}**
on **{highest['Date'].strftime('%d-%b-%Y')}**

• Lowest Load Recorded : **{lowest['Total_System_Load']:,.0f}**
on **{lowest['Date'].strftime('%d-%b-%Y')}**

• Best Performing Month : **{best_month['Month']}**

• Highest Average Load :
**{best_month['Total_System_Load']:,.0f}**

• Lowest Performing Month :
**{worst_month['Month']}**

• Capacity Utilization :
**{utilization:.2f}%**

""")


    st.divider()


    # -------------------------------------------------
    # Recommendation Engine
    # -------------------------------------------------

    st.subheader("🤖 AI Recommendation Engine")

    recommendations=[]

    if growth>10:

        recommendations.append(
        "✔ System demand is increasing. Capacity planning is recommended."
        )

    else:

        recommendations.append(
        "✔ System growth is stable."
        )

    if utilization>80:

        recommendations.append(
        "✔ HHS capacity utilization is very high."
        )

    else:

        recommendations.append(
        "✔ Capacity utilization is under control."
        )

    if avg_intake>0:

        recommendations.append(
        "✔ Intake is positive."
        )

    else:

        recommendations.append(
        "✔ Intake trend should be monitored carefully."
        )

    if filtered_df["Total_System_Load"].std()>100:

        recommendations.append(
        "✔ High volatility detected."
        )

    else:

        recommendations.append(
        "✔ System is operating within expected range."
        )

    for item in recommendations:

        st.info(item)


    st.divider()


    # -------------------------------------------------
    # Performance Score
    # -------------------------------------------------

    st.subheader("🏆 Overall Performance Score")

    
    score = min(score,100)

    if growth>0:
        score+=25

    if utilization<90:
        score+=25

    if filtered_df["Total_System_Load"].std()<150:
        score+=25

    if avg_intake>0:
        score+=25

    st.progress(score/100)

    st.metric(

        "Overall Score",

        f"{score}/100"

    )


    st.divider()


    # -------------------------------------------------
    # Monthly Summary
    # -------------------------------------------------

    st.subheader("📅 Monthly Performance")

    fig=px.bar(

        monthly,

        x="Month",

        y="Total_System_Load",

        color="Total_System_Load",

        template="plotly_dark"

    )

    fig.update_layout(

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    # ======================================================
# DAY 2 : PERFORMANCE OPTIMIZATION
# ======================================================

st.divider()

st.subheader("⚙ Dashboard Performance Summary")

# --------------------------------------------
# Dataset Summary
# --------------------------------------------

c1,c2,c3,c4=st.columns(4)

c1.metric(
    "Rows",
    len(filtered_df)
)

c2.metric(
    "Columns",
    len(filtered_df.columns)
)

c3.metric(
    "Memory Usage",
    f"{filtered_df.memory_usage(deep=True).sum()/1024:.1f} KB"
)

c4.metric(
    "Missing Values",
    int(filtered_df.isna().sum().sum())
)

st.divider()

# --------------------------------------------
# Dataset Health
# --------------------------------------------

st.subheader("📋 Dataset Health Check")

health=[]

if filtered_df.isna().sum().sum()==0:
    health.append("✅ No Missing Values")
else:
    health.append("⚠ Missing values detected")

if filtered_df.duplicated().sum()==0:
    health.append("✅ No Duplicate Records")
else:
    health.append(f"⚠ {filtered_df.duplicated().sum()} Duplicate Rows")

if filtered_df["Date"].is_monotonic_increasing:
    health.append("✅ Date sequence is correct")
else:
    health.append("⚠ Date column is not sorted")

for item in health:
    st.success(item)

st.divider()

# --------------------------------------------
# Correlation Summary
# --------------------------------------------

st.subheader("📊 Quick Correlation")

corr=filtered_df[[
    "Total_System_Load",
    "Children in HHS Care",
    "Net_Intake"
]].corr().round(2)

st.dataframe(
    corr,
    use_container_width=True
)

st.divider()

# --------------------------------------------
# Metric Selector Preview
# --------------------------------------------

st.subheader("📈 Selected Metric Summary")

st.write(filtered_df[metric_view].describe().round(2))

st.divider()

# --------------------------------------------
# Last 10 Records
# --------------------------------------------

st.subheader("📅 Latest Records")

st.dataframe(

    filtered_df.tail(10),

    use_container_width=True

)

st.divider()
# ======================================================
# TAB 8 : EXPORT CENTER
# ======================================================

with tab8:

    st.subheader("📥 Export & Project Center")

    st.success("Download Reports & Project Data")

    st.divider()

    # ============================================
    # Dataset Summary
    # ============================================

    c1,c2,c3,c4=st.columns(4)

    c1.metric(
        "Rows",
        len(filtered_df)
    )

    c2.metric(
        "Columns",
        len(filtered_df.columns)
    )

    c3.metric(
        "Years",
        filtered_df["Year"].nunique()
    )

    c4.metric(
        "Records",
        f"{len(filtered_df):,}"
    )

    st.divider()

    # ============================================
    # Download Current Dataset
    # ============================================

    st.subheader("📄 Export Current Dataset")

    csv=filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(

        "⬇ Download Filtered Dataset",

        csv,

        file_name="Filtered_Dataset.csv",

        mime="text/csv",

        use_container_width=True

    )

    # ============================================
    # Export Summary
    # ============================================

    st.subheader("📊 Export Statistical Summary")

    summary=filtered_df.describe().round(2)

    csv_summary=summary.to_csv().encode("utf-8")

    st.download_button(

        "⬇ Download Summary",

        csv_summary,

        file_name="Summary_Report.csv",

        mime="text/csv",

        use_container_width=True

    )

    # ============================================
    # Export Correlation
    # ============================================

    st.subheader("📈 Export Correlation Matrix")

    corr=filtered_df.select_dtypes(include="number").corr().round(2)

    csv_corr=corr.to_csv().encode("utf-8")

    st.download_button(

        "⬇ Download Correlation",

        csv_corr,

        file_name="Correlation.csv",

        mime="text/csv",

        use_container_width=True

    )

    st.divider()

    # ============================================
    # Project Information
    # ============================================

    st.subheader("📘 Project Information")

    st.info("""

Project Name :
UAC Care System Analytics Dashboard

Developer :
Narendra Suthar

Internship :
Unified Mentor

Technology Stack :

• Python

• Streamlit

• Plotly

• Pandas

• NumPy

• Scikit-Learn

• Machine Learning

Dashboard Version :

Version 2.0

""")

    st.divider()

    # ============================================
    # Dashboard Statistics
    # ============================================

    st.subheader("📌 Dashboard Statistics")

    features=pd.DataFrame({

        "Feature":[

            "Interactive Charts",

            "Forecast",

            "Risk Analysis",

            "Heatmap",

            "Executive Dashboard",

            "Export Center",

            "ML Model"

        ],

        "Status":[

            "✅",

            "✅",

            "✅",

            "✅",

            "✅",

            "✅",

            "✅"

        ]

    })

    st.dataframe(

        features,

        use_container_width=True

    )

    st.divider()

    st.success("✅ Dashboard Ready For Internship Submission")

    st.caption(
        "Developed by Narendra Suthar | Unified Mentor Internship Project"
    )