import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

load_dotenv()

# Global cache to speed up local loading
_LOCAL_DATA_CACHE = None

def fetch_load_data(country_code: str, start: str, end: str) -> pd.DataFrame:
    """
    Tries to fetch from ENTSO-E, falls back to local data/electricity_load.csv with caching.
    """
    global _LOCAL_DATA_CACHE

    # 1. Try ENTSO-E API
    api_key = os.getenv("ENTSOE_API_KEY")
    if api_key and api_key != "your_key_here":
        try:
            client = EntsoePandasClient(api_key=api_key)
            start_ts = pd.Timestamp(start, tz='UTC')
            end_ts   = pd.Timestamp(end,   tz='UTC')
            series = client.query_load(country_code, start=start_ts, end=end_ts)
            
            # Robustly check if returned object is a DataFrame or a Series
            if isinstance(series, pd.DataFrame):
                df = series.rename(columns={series.columns[0]: 'load_MW'})[['load_MW']]
            else:
                df = series.to_frame(name='load_MW')
                
            df.index.name = 'timestamp'
            return df.ffill()
        except Exception as e:
            print(f"[WARN] API Fetch failed: {e}")

    # 2. Fallback to local CSV with optimization
    csv_path = 'data/electricity_load.csv'
    if os.path.exists(csv_path):
        if _LOCAL_DATA_CACHE is None:
            print(f"[INFO] Reading and optimizing local dataset from {csv_path} (First time only)...")
            df_raw = pd.read_csv(csv_path)
            df_raw.columns = [c.strip('"') for c in df_raw.columns]
            
            # Parse timestamps once
            df_raw['timestamp'] = df_raw['MTU (UTC)'].str.split(' - ').str[0].str.strip('"')
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], format='%d/%m/%Y %H:%M')
            
            _LOCAL_DATA_CACHE = df_raw.rename(columns={'Actual Total Load (MW)': 'load_MW'})[['timestamp', 'load_MW']]
            _LOCAL_DATA_CACHE = _LOCAL_DATA_CACHE.set_index('timestamp').sort_index()
            _LOCAL_DATA_CACHE = _LOCAL_DATA_CACHE.resample('h').mean().ffill()

        # 3. Filter fallback data to requested range
        try:
            start_ts = pd.Timestamp(start).tz_localize(None)
            end_ts   = pd.Timestamp(end).tz_localize(None)
            
            # If requesting future dates (2026 and beyond), shift back to 2025 for realistic simulation
            if start_ts.year >= 2026:
                shift_years = start_ts.year - 2025
                start_ts = start_ts - pd.DateOffset(years=shift_years)
                end_ts   = end_ts - pd.DateOffset(years=shift_years)
                
            mask = (_LOCAL_DATA_CACHE.index >= start_ts) & (_LOCAL_DATA_CACHE.index <= end_ts)
            df_filtered = _LOCAL_DATA_CACHE.loc[mask]
            
            if len(df_filtered) > 0:
                return df_filtered
        except Exception as filter_err:
            print(f"[WARN] Fallback filtering failed: {filter_err}")
            
        # Default safety fallback (last 8 days of local cache)
        return _LOCAL_DATA_CACHE.iloc[-192:]

    raise RuntimeError("No data found! API failed and local CSV missing.")