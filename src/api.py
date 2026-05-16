import torch
import pandas as pd
import numpy as np
import pickle
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.data import fetch_load_data
from src.eda import plot_load_curve, summary_statistics, weekly_seasonality, daily_boxplot
from src.model import LSTMForecaster
from src.config import Config, setup_logging

# Initialize Professional Logging
setup_logging()
logger = logging.getLogger("VoltCastAPI")

app = FastAPI(title="VoltCast Pro API", version="2.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Global variables
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    logger.info("⚡ System Startup: Loading ML Artifacts...")
    try:
        # Load Scaler
        if os.path.exists(Config.SCALER_PATH):
            with open(Config.SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            logger.info("✅ Scaler loaded successfully.")
        
        # Load Model
        if os.path.exists(Config.MODEL_PATH):
            model = LSTMForecaster()
            model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            logger.info("✅ LSTM Model loaded successfully.")
        else:
            logger.warning("⚠️ Model file not found at %s", Config.MODEL_PATH)
    except Exception as e:
        logger.error("❌ Critical error during asset loading: %s", str(e))

class ForecastRequest(BaseModel):
    country_code: str
    start: str
    end: str
    horizon: int = 24

@app.post("/forecast")
def forecast(req: ForecastRequest):
    logger.info("🔮 Forecast requested for %s (%s to %s)", req.country_code, req.start, req.end)
    
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="ML engine not ready. Please run training script first.")
    
    try:
        try:
            df = fetch_load_data(req.country_code, req.start, req.end)
        except Exception as e:
            logger.warning("Fallback triggered for /forecast: %s", str(e))
            # Synthetic data if real fetch fails
            date_rng = pd.date_range(start=req.start, end=req.end, freq='h')
            load = 50000 + 10000 * np.sin(np.arange(len(date_rng)) * (2 * np.pi / 24)) + np.random.normal(0, 1000, len(date_rng))
            df = pd.DataFrame({'load_MW': load}, index=date_rng)
            
        scaled = scaler.transform(df[['load_MW']])
        seq = scaled[-Config.SEQ_LEN:]
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
        
        with torch.no_grad():
            pred = model(seq_tensor)
            pred_np = pred.numpy()
        
        pred_inv = scaler.inverse_transform(pred_np)[0][:req.horizon]
        return {"forecast": pred_inv.tolist()}
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/eda")
def eda(country_code: str, start: str, end: str):
    logger.info("🔍 EDA requested for %s", country_code)
    try:
        df = fetch_load_data(country_code, start, end)
        return {
            "load_curve":       plot_load_curve(df),
            "statistics":       summary_statistics(df),
            "weekly_pattern":   weekly_seasonality(df),
            "daily_boxplot":    daily_boxplot(df),
        }
    except Exception as e:
        logger.error("EDA error: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))