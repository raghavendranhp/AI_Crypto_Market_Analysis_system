import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from llm_insights import get_market_insights

#configuration
st.set_page_config(page_title="crypto market signal & risk analyzer", layout="wide")

st.title("crypto market signal & risk analyzer")

#load models
@st.cache_resource
def load_models():
    model_dir = "models"
    if not os.path.exists(model_dir):
        model_dir = "app/models"
    
    rf_model_path = os.path.join(model_dir, "random_forest_trend.joblib")
    iso_model_path = os.path.join(model_dir, "isolation_forest_anomaly.joblib")
    
    try:
        rf = joblib.load(rf_model_path)
        iso = joblib.load(iso_model_path)
        return rf, iso
    except Exception as e:
        st.error(f"failed to load models: {e}")
        return None, None

rf_model, iso_model = load_models()

#load data for historical chart
@st.cache_data
def load_data():
    data_path = "../data/crypto_trading_dataset.csv"
    if not os.path.exists(data_path):
        data_path = "data/crypto_trading_dataset.csv"
    try:
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"failed to load data: {e}")
        return pd.DataFrame()

df = load_data()

#sidebar
st.sidebar.header("configuration")
crypto_choice = st.sidebar.selectbox("select cryptocurrency", ["Bitcoin", "Ethereum", "Solana"])

if not df.empty:
    st.subheader(f"historical price trend: {crypto_choice}")
    #filter dataframe by selected coin_name
    filtered_df = df[df['coin_name'] == crypto_choice]
    if not filtered_df.empty:
        st.line_chart(filtered_df.set_index('timestamp')['close_price'])
    else:
        st.write("no historical data available for this selection.")

#dynamic prediction section
st.header("dynamic prediction (what-if analysis)")

col1, col2 = st.columns(2)

with col1:
    price_change = st.slider("price change %", min_value=-20.0, max_value=20.0, value=0.0, step=0.1)
    ma7 = st.number_input("7-period moving average (ma7)", value=40000.0, step=100.0)
    
with col2:
    vol_index = st.slider("volatility index", min_value=0.0, max_value=20.0, value=2.0, step=0.1)
    vol_spike = st.slider("volume spike ratio", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

#predict
if st.button("generate analysis") and rf_model and iso_model:
    #calculate risk score
    risk_score = min(100.0, (vol_index * 10) + (vol_spike * 5))
    
    #prepare features
    trend_features = pd.DataFrame({
        'price_change_pct': [price_change],
        'ma7': [ma7],
        'volatility_index': [vol_index],
        'volume_spike_ratio': [vol_spike]
    })
    
    anomaly_features = pd.DataFrame({
        'price_change_pct': [price_change],
        'ma7': [ma7],
        'volatility_index': [vol_index],
        'volume_spike_ratio': [vol_spike],
        'risk_score': [risk_score]
    })
    
    predicted_trend = rf_model.predict(trend_features)[0]
    anomaly_pred = iso_model.predict(anomaly_features)[0]
    anomaly_status = "anomaly detected" if anomaly_pred == -1 else "normal"
    
    metrics = {
        "price_change_pct": round(price_change, 2),
        "ma7": round(ma7, 2),
        "volatility_index": round(vol_index, 2),
        "volume_spike_ratio": round(vol_spike, 2),
        "risk_score": round(risk_score, 2)
    }
    
    st.subheader("ml predictions")
    st.write(f"math-based risk score: {risk_score:.2f} / 100")
    st.write(f"predicted trend: {predicted_trend}")
    st.write(f"anomaly status: {anomaly_status}")
    
    #llm insights
    st.subheader("agent insights")
    with st.spinner("generating insights via groq..."):
        insight = get_market_insights(metrics, predicted_trend, anomaly_status)
    
    #output json
    final_output = {
        "asset": crypto_choice,
        "metrics": metrics,
        "ml_predictions": {
            "trend": predicted_trend,
            "anomaly": anomaly_status
        },
        "llm_insight": insight,
        "trading_signal": "BUY" if (predicted_trend == "Bullish" and anomaly_pred != -1 and risk_score < 70) else ("SELL" if predicted_trend == "Bearish" or risk_score > 80 else "HOLD")
    }
    
    st.json(final_output)
