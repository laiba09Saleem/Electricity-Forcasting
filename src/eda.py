import io, base64
import pandas as pd
import matplotlib.pyplot as plt

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def plot_load_curve(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df.index, df['load_MW'])
    ax.set_title('Electricity Load Curve')
    ax.set_xlabel('Time'); ax.set_ylabel('Load (MW)')
    return _fig_to_base64(fig)

def summary_statistics(df: pd.DataFrame) -> dict:
    s = df['load_MW'].describe()
    return s.to_dict()

def weekly_seasonality(df: pd.DataFrame) -> str:
    df = df.copy()
    df['day_of_week'] = df.index.day_name()
    weekly = df.groupby('day_of_week')['load_MW'].mean().reindex(
        ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
    fig, ax = plt.subplots()
    weekly.plot(kind='bar', ax=ax)
    ax.set_title('Average Load by Day of Week')
    return _fig_to_base64(fig)

def daily_boxplot(df: pd.DataFrame) -> str:
    df = df.copy()
    df['hour'] = df.index.hour
    fig, ax = plt.subplots(figsize=(12, 5))
    df.boxplot(column='load_MW', by='hour', ax=ax)
    ax.set_title('Load Distribution by Hour')
    plt.suptitle('')
    return _fig_to_base64(fig)