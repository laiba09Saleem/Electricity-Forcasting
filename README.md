# ⚡ VoltCast Pro: Electricity Load Forecasting System

VoltCast Pro is an end-to-end industry-standard web application designed to forecast electricity demand. It leverages Deep Learning (LSTM), FastAPI, and Streamlit to provide real-time analytical insights and multi-step forecasting using ENTSO-E data.

## Live Demo: https://voltcastpro.streamlit.app/


## 🚀 Features
- **Predictive Engine:** PyTorch-based LSTM model for 24-hour load forecasting.
- **Interactive Dashboard:** Premium Streamlit UI with Plotly visualizations.
- **Robust API:** High-performance FastAPI backend with asset caching.
- **Hybrid Data Ingestion:** Real-time API integration with local CSV fallback.
- **Professional Logging:** Full system activity tracking in `app.log`.

## 📂 Repository Structure
```text
├── app.py              # Streamlit Dashboard (Frontend)
├── src/
│   ├── api.py          # FastAPI Server (Backend)
│   ├── data.py         # Data Ingestion & Caching
│   ├── eda.py          # Analytical Visualization Module
│   ├── model.py        # LSTM Architecture (PyTorch)
│   ├── train.py        # Professional Training Pipeline
│   └── config.py       # Centralized Configurations
├── models/             # Saved ML Artifacts (.pt, .pkl)
├── data/               # Historical Datasets (CSV)
├── notebooks/          # Exploratory Jupyter Notebooks
├── Dockerfile          # Containerization Config
└── requirements.txt    # System Dependencies
```

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/laiba09Saleem/Electricity-Forecas.git
   cd electricity_forcasting
   ```

2. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file and add your ENTSO-E API Key:
   ```text
   ENTSOE_API_KEY=your_actual_key_here
   ```

## 🏃‍♂️ How to Run

### Step 1: Train the Model
If the `models/` folder is empty, run the training pipeline first:
```bash
python -m src.train
```

### Step 2: Start the Backend (API)
```bash
python -m uvicorn src.api:app --reload
```

### Step 3: Launch the Dashboard
```bash
streamlit run app.py
```

## 📊 Technologies Used
- **Logic:** Python 3.14
- **ML:** PyTorch, Scikit-Learn
- **API:** FastAPI, Uvicorn
- **UI:** Streamlit, Plotly
- **Data:** Pandas, ENTSO-E API

---
*Developed as part of the End-to-End Electricity Load Forecasting Assignment.*
