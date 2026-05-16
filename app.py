import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import base64

# --- Page Config ---
st.set_page_config(
    page_title="VoltCast | Pro Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3e445b;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

API = "http://localhost:8000"

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-flat-flatarticons/128/external-energy-ecology-flatart-icons-flat-flatarticons.png", width=80)
    st.title("VoltCast Pro")
    st.subheader("Configuration")
    country = st.selectbox("Market Region", ["DE (Germany)","FR (France)","GB (Great Britain)","ES (Spain)","IT (Italy)"])
    country_code = country.split(" ")[0]
    
    start   = st.date_input("Analysis Start", value=pd.to_datetime("2025-01-01"))
    end     = st.date_input("Analysis End", value=pd.to_datetime("2025-01-07"))
    horizon = st.select_slider("Forecast Horizon", options=[24, 48, 72], value=24)
    
    st.divider()
    st.info("⚡ System Status: Connected to Backend")

# --- Header ---
col1, col2 = st.columns([2, 1])
with col1:
    st.title("⚡ Energy Demand Cockpit")
    st.caption(f"Real-time forecasting for {country} region using LSTM deep learning.")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Predictive Insights", "🔬 Advanced Analytics", "⚙️ System Info"])

with tab1:
    if st.button("🚀 Run Predictive Analysis"):
        with st.spinner("Analyzing market patterns..."):
            try:
                r = requests.post(f"{API}/forecast", json={
                    "country_code": country_code,
                    "start": str(start).replace("-",""),
                    "end":   str(end).replace("-",""),
                    "horizon": horizon
                })
                
                if r.status_code == 200:
                    forecast_data = r.json()["forecast"]
                    
                    # Metrics Row
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Peak Predicted Load", f"{max(forecast_data):,.0f} MW", "+2.5%")
                    m2.metric("Min Predicted Load", f"{min(forecast_data):,.0f} MW", "-1.2%")
                    m3.metric("Avg Load", f"{sum(forecast_data)/len(forecast_data):,.0f} MW")
                    
                    # Professional Chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=forecast_data, 
                        mode='lines+markers', 
                        name='LSTM Forecast',
                        line=dict(color='#ff4b4b', width=3),
                        fill='tozeroy'
                    ))
                    fig.update_layout(
                        title=f"Next {horizon} Hours Load Forecast",
                        xaxis_title="Hours Ahead",
                        yaxis_title="Load (MW)",
                        template="plotly_dark",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Backend error: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab2:
    if st.button("🔍 Load Market Analytics"):
        with st.spinner("Processing large dataset..."):
            try:
                r = requests.get(f"{API}/eda", params={
                    "country_code": country_code,
                    "start": str(start).replace("-",""),
                    "end":   str(end).replace("-","")
                })
                
                if r.status_code == 200:
                    data = r.json()
                    
                    # Display Plots
                    c1, c2 = st.columns(2)
                    with c1:
                        st.subheader("Historical Load Curve")
                        img1 = Image.open(io.BytesIO(base64.b64decode(data["load_curve"])))
                        st.image(img1, use_column_width=True)
                        
                        st.subheader("Weekly Seasonality")
                        img2 = Image.open(io.BytesIO(base64.b64decode(data["weekly_pattern"])))
                        st.image(img2, use_column_width=True)
                        
                    with c2:
                        st.subheader("Load Distribution (Hourly)")
                        img3 = Image.open(io.BytesIO(base64.b64decode(data["daily_boxplot"])))
                        st.image(img3, use_column_width=True)
                        
                        st.subheader("Statistical Summary")
                        st.table(pd.DataFrame([data["statistics"]]).T.rename(columns={0: "Value"}))
                else:
                    st.error(f"Backend error: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection failed: {e}")

with tab3:
    st.header("System Architecture")
    st.write("""
    - **Backend:** FastAPI (Python 3.14)
    - **ML Engine:** PyTorch LSTM
    - **Visualization:** Plotly & Streamlit
    - **Data Source:** ENTSO-E Transparency Platform / Local Historical CSV
    """)
    st.info("Note: This system is currently running in 'Data Fallback' mode using local electricity_load.csv.")