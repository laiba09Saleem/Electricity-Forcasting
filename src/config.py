import logging
import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Logging Configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(BASE_DIR, "app.log")),
            logging.StreamHandler()
        ]
    )

class Config:
    SEQ_LEN = 168
    HORIZON = 24
    MODEL_PATH = os.path.join(MODELS_DIR, 'lstm_model.pt')
    SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
    LOCAL_CSV_PATH = os.path.join(DATA_DIR, 'electricity_load.csv')
    API_PORT = 8000
    API_HOST = "0.0.0.0"
