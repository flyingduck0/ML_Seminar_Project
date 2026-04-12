import pandas as pd
import numpy as np
import yfinance as yf
import re
import os
from datetime import datetime

# ==========================================
# 1. CONFIGURATION & DIRECTORIES
# ==========================================
MACRO_CONFIG = {
    'FED': {'hist': 'TOTAL_HISTORY_FED.csv', 'ref': 'FINAL_FED_MARKETS.csv', 'feat': 'NORMALIZED_EV', 'event_col': 'meeting_date'},
    'GDP': {'hist': 'TOTAL_HISTORY_GDP.csv', 'ref': 'FINAL_GDP_MARKETS.csv', 'feat': 'EXPECTED_GDP_INDEX', 'event_col': 'event'},
    'LABOR': {'hist': 'TOTAL_HISTORY_LABOR.csv', 'ref': 'FINAL_LABOR_MARKETS.csv', 'feat': 'EXPECTED_UNEMPLOYMENT_INDEX', 'event_col': 'endDate'},
    'INF_YEARLY': {'hist': 'TOTAL_HISTORY_INF_YEARLY.csv', 'ref': 'FINAL_INFLATION_YEARLY.csv', 'feat': 'EXPECTED_INFLATION_INDEX', 'event_col': 'target_year'},
    'INF_MONTHLY': {'hist': 'TOTAL_HISTORY_INF_MONTHLY.csv', 'ref': None, 'feat': 'EXPECTED_MONTHLY_INF_INDEX', 'event_col': 'event_label'}
}

TICKERS = {"SPY": "SPY_chg", "QQQ": "QQQ_chg", "BTC-USD": "BTC_chg", "^VIX": "VIX_chg", "GC=F": "Gold_chg", "CL=F": "Oil_chg", "^TNX": "US10Y_chg"}
EVENT_DATES = {
    'FOMC': ['2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18'], # Add more as needed
    'CPI': ['2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10'],
    'Employment': ['2025-01-10', '2025-02-07', '2025-03-07'],
    'GDP': ['2025-01-30', '2025-02-27', '2025-03-27']
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_weight(title):
    nums = re.findall(r'(\d+\.\d+|\d+)', str(title))
    return float(nums[0]) if nums else 0.0

def process_macro_pillar(name, cfg):
    if not os.path.exists(cfg['hist']): return pd.DataFrame()
    
    df = pd.read_csv(cfg['hist'])
    df['t'] = pd.to_datetime(df['t'], utc=True)
    df = df[df['t'].dt.minute < 5].copy() # Snap to hour
    
    # Logic for Monthly Inflation (No Ref File) vs Others
    if name == 'INF_MONTHLY':
        df['event_label'] = df['slug'].str.extract(r'([a-z]+)', expand=False).str.capitalize() + "-2025"
    else:
        ref = pd.read_csv(cfg['ref'])
        df = df.merge(ref[['slug', cfg['event_col']]], on='slug', how='left')

    # Pivot & Normalize
    pivot = df.pivot_table(index=df['t'].dt.floor('h'), columns='groupItemTitle', values='p', aggfunc='last')
    pivot['RAW_TOTAL_SUM'] = pivot.sum(axis=1)
    
    # Calculate EV
    raw_ev = 0
    for col in [c for c in pivot.columns if c not in ['RAW_TOTAL_SUM', cfg['event_col']]]:
        raw_ev += pivot[col].fillna(0) * get_weight(col)
    
    pivot[cfg['feat']] = (raw_ev / pivot['RAW_TOTAL_SUM'])
    if 'INDEX' in cfg['feat']: pivot[cfg['feat']] += 100
    
    return pivot[[cfg['feat']]].reset_index().rename(columns={'t': 'Date'})

# ==========================================
# 3. CORE PIPELINE EXECUTION
# ==========================================
def run_unified_pipeline():
    print("🚀 Starting Unified Macro-Asset Pipeline...")

    # A. Build Continuous Timeline
    full_range = pd.date_range(start='2025-01-01', end=datetime.now(), freq='h', tz='UTC')
    master_df = pd.DataFrame({'Date': full_range})

    # B. Process Macro Pillars (Deltas & Dummies)
    for name, cfg in MACRO_CONFIG.items():
        print(f"Processing {name}...")
        pillar_df = process_macro_pillar(name, cfg)
        if not pillar_df.empty:
            # Shift to Delta
            pillar_df[cfg['feat']] = pillar_df[cfg['feat']].diff()
            master_df = pd.merge(master_df, pillar_df, on='Date', how='left')

    # C. Asset Integration (yfinance)
    print("Downloading Financial Data...")
    assets = yf.download(list(TICKERS.keys()), start='2025-01-01', interval='1h')['Close']
    assets.index = assets.index.tz_convert('UTC').ceil('h')
    asset_changes = assets.pct_change().reindex(master_df['Date']).fillna(0)
    master_df = master_df.join(asset_changes.rename(columns=TICKERS), on='Date')

    # D. Announcement Dummies (NYC to UTC)
    print("Generating Announcement Dummies...")
    for label, dates in EVENT_DATES.items():
        ny_time = '14:00:00' if label == 'FOMC' else '08:00:00'
        ts_utc = pd.to_datetime([f"{d} {ny_time}" for d in dates]) \
                    .tz_localize('America/New_York', ambiguous='infer') \
                    .tz_convert('UTC')
        master_df[f'Ann_{label}'] = master_df['Date'].isin(ts_utc).astype(int)

    # E. Final Cleanup & Integrity Dummies
    print("Finalizing Feature Set...")
    poly_features = [cfg['feat'] for cfg in MACRO_CONFIG.values()]
    
    # Trim to when all markets are live
    valid_start = master_df[poly_features].dropna(how='all').index[0]
    master_df = master_df.iloc[valid_start:].reset_index(drop=True)

    # Missing Data Dummies & Zero-Fill
    for feat in poly_features:
        if feat in master_df.columns:
            master_df[f"{feat}_is_missing"] = master_df[feat].isna().astype(int)
            master_df[feat] = master_df[feat].fillna(0)

    # Save
    master_df.to_csv('FEATURES_PREPARED_UNIFIED.csv', index=False)
    print("✅ Success! File saved: FEATURES_PREPARED_UNIFIED.csv")

if __name__ == "__main__":
    run_unified_pipeline()