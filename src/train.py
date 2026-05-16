import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from sklearn.preprocessing import MinMaxScaler
import pickle
from src.data import fetch_load_data
from src.model import LSTMForecaster
from src.config import Config, setup_logging

# Initialize Logging
setup_logging()
logger = logging.getLogger("TrainingEngine")

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['load_MW']])
    
    X, y = [], []
    for i in range(len(scaled_data) - Config.SEQ_LEN - Config.HORIZON):
        X.append(scaled_data[i : i + Config.SEQ_LEN])
        y.append(scaled_data[i + Config.SEQ_LEN : i + Config.SEQ_LEN + Config.HORIZON].flatten())
        
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)), scaler

def train():
    logger.info("🚀 INITIALIZING PROFESSIONAL TRAINING PIPELINE")
    
    # 1. Fetch Data
    try:
        logger.info("📥 Fetching historical data...")
        # For training, we just use the local dataset if available
        df = fetch_load_data("DE", "20230101", "20230301")
    except Exception as e:
        logger.error("❌ Failed to fetch data: %s", str(e))
        return

    # 2. Prepare Data
    logger.info("📊 Preprocessing data (Scaling & Windowing)...")
    X, y, scaler = prepare_data(df)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize Model
    logger.info("🤖 Building LSTM architecture (Layers: 2, Hidden: 64)...")
    model = LSTMForecaster(output_dim=Config.HORIZON)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training Loop
    logger.info("🔥 Starting Training Loop...")
    model.train()
    epochs = 10
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        logger.info(f"Epoch [{epoch+1}/{epochs}] | Avg MSE Loss: {total_loss/len(loader):.6f}")
        
    # 5. Export Artifacts
    logger.info("💾 Exporting model artifacts to 'models/' folder...")
    os.makedirs(os.path.dirname(Config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    with open(Config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    logger.info("✅ SUCCESS: Professional model trained and deployed.")

if __name__ == "__main__":
    train()
