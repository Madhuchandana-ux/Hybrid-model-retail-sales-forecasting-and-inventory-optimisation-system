import streamlit as st
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(layout="wide", page_icon="🛒")
st.title("🛒 Retail Forecasting Dashboard")

# LOAD YOUR MODEL (created by your notebook)
@st.cache_data
def load_model():
    artifacts = joblib.load("retail_forecast_model.pkl")
    return artifacts

artifacts = load_model()
model = artifacts['model']
features = artifacts['features']
resid_std = artifacts['resid_std']

# YOUR FUNCTIONS (extracted automatically)
def croston_forecast(y, alpha=0.1, h=28):
    demand = y.values
    z = demand[demand>0]
    p = np.diff(np.r_[0, np.where(demand>0)[0]])
    if len(z)==0: return np.full(h, 0.1)
    z_hat, p_hat = z[0], p[1] if len(p)>1 else 1
    for i in range(1, len(z)): z_hat = alpha*z[i] + (1-alpha)*z_hat
    for i in range(1, len(p)): p_hat = alpha*p[i] + (1-alpha)*p_hat
    return (z_hat/p_hat) * np.ones(h)

def inventory_policy(forecast, resid_std, on_hand, lead_time, service=0.95):
    z = norm.ppf(service)
    mu_L = forecast[:lead_time].sum()
    sigma_L = resid_std * (lead_time ** 0.5)
    SS = z * sigma_L
    ROP = mu_L + SS
    annual_demand = forecast.mean() * 365
    EOQ = np.sqrt((2 * annual_demand * 500) / (100 * 0.2))
    order_qty = max(0, max(EOQ, ROP - on_hand))
    return dict(mu_L=mu_L, SS=SS, ROP=ROP, EOQ=EOQ, order_qty=order_qty)

# CONTROLS
st.sidebar.header("📋 Planner Inputs")
store = st.sidebar.selectbox("Store", ["S1","S2","S3"])
item = st.sidebar.selectbox("Item", ["I01","I11","I21"])
stock = st.sidebar.number_input("Current Stock", 0, 200, 25)
lead = st.sidebar.number_input("Lead Time (days)", 1, 21, 5)
service = st.sidebar.slider("Service Level", 80, 99, 95)/100

# RECENT HISTORY
@st.cache_data
def get_recent_data():
    np.random.seed(42)
    return np.random.poisson(3, 28).astype(float) * 0.8

if st.button("🚀 Generate Forecast & PO", type="primary"):
    recent = get_recent_data()
    
    # FORECAST
    dates = pd.date_range("today", periods=28)
    X_pred = pd.DataFrame({
        'dow': dates.dayofweek,
        'week_of_year': dates.isocalendar().week,
        'month': dates.month
    })
    for f in features:
        if f not in X_pred.columns: X_pred[f] = recent.mean()
    X_pred = X_pred[features].fillna(0)
    
    ml_fc = np.maximum(0.1, model.predict(X_pred))
    croston_fc = croston_forecast(pd.Series(recent), h=28)
    
    p0 = 1 - (recent > 0).mean()
    hybrid_fc = 0.3*ml_fc + 0.7*croston_fc
    
    # INVENTORY
    policy = inventory_policy(hybrid_fc, resid_std, stock, lead, service)
    
    # RESULTS
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Order Qty", f"{policy['order_qty']:.0f}")
    col2.metric("🎯 Reorder Point", f"{policy['ROP']:.0f}")
    col3.metric("🛡️ Safety Stock", f"{policy['SS']:.1f}")
    
    col1.metric("💰 EOQ", f"{policy['EOQ']:.0f}")
    col2.metric("📊 Lead Demand", f"{policy['mu_L']:.1f}")
    
    # CHART
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=hybrid_fc, name='Hybrid', line=dict(width=4, color='green')))
    fig.add_vline(x=dates[lead-1], line_dash="dash", line_color="red")
    st.plotly_chart(fig)
    
    # CSV
    df_po = pd.DataFrame([{**policy, 'store': store, 'item': item}])
    st.download_button("💾 Export PO", df_po.to_csv(index=False), f"PO_{store}_{item}.csv")


