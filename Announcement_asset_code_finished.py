# %%
# ==========================================
# PHASE : Asset Data Inclusion
# ==========================================
import os
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

OUTPUT_FILE = "financial_markets_hourly_utc.csv"
LEVELS_FILE = "financial_markets_hourly_levels_utc.csv"
METADATA_FILE = "financial_markets_metadata.json"

START_UTC = "2025-01-01 00:00:00"
UTC = "UTC"


MAIN_PRICE_TICKERS = {
    "SPY": "SPY_chg",
    "QQQ": "QQQ_chg",
    "GC=F": "Gold_chg",        # gold futures
    "CL=F": "Oil_chg",         # WTI crude futures
    "DX-Y.NYB": "DXY_chg",     # Dollar Index
    "BTC-USD": "BTC_chg",      # Bitcoin
    "^VIX": "VIX_chg"          # VIX
}


YIELD_CANDIDATES = {
    "US2Y_chg": ["ZT=F"],
    "US10Y_chg": ["^TNX", "ZN=F"]
}


FUTURES_SPEC = {
    "ES=F": "SP500_fut_chg",   # S&P 500 futures
}


def now_utc():
    return pd.Timestamp.now(tz=UTC).floor("h")


def to_utc_index(idx):
    idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        return idx.tz_localize(UTC)
    return idx.tz_convert(UTC)


def make_hourly_index(start_ts, end_ts):
    return pd.date_range(start=start_ts, end=end_ts, freq="1h", tz=UTC, name="Date")


def chunk_time_ranges(start_ts, end_ts, days_per_chunk=60):
    """
    Yahoo intraday history is limited, so we download in chunks.
    """
    ranges = []
    current = start_ts

    while current < end_ts:
        chunk_end = min(current + pd.Timedelta(days=days_per_chunk), end_ts)
        ranges.append((current, chunk_end))
        current = chunk_end

    return ranges


def extract_close_series(df, ticker):
    """
    yfinance can return either normal columns or MultiIndex columns.
    This safely extracts the Close series.
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].copy()
        else:
            close_block = df["Close"]
            if isinstance(close_block, pd.DataFrame):
                s = close_block.iloc[:, 0].copy()
            else:
                s = close_block.copy()
    else:
        s = df["Close"].copy()

    s = pd.Series(s)
    s.index = to_utc_index(s.index)


    s.index = s.index.ceil("h")


    s = s.groupby(s.index).last().sort_index()

    s.name = ticker
    return s


def download_hourly_series(ticker, start_ts, end_ts, interval="1h"):
    """
    Download hourly close series in chunks.
    """
    pieces = []

    for chunk_start, chunk_end in chunk_time_ranges(start_ts, end_ts, days_per_chunk=60):
        try:
            df = yf.download(
                ticker,
                start=chunk_start.strftime("%Y-%m-%d"),
                end=(chunk_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False
            )

            s = extract_close_series(df, ticker)

            if not s.empty:
                s = s[(s.index >= chunk_start) & (s.index <= chunk_end)]
                pieces.append(s)

        except Exception as e:
            print(f"Download failed for {ticker} | {chunk_start} to {chunk_end}: {e}")

    if not pieces:
        return pd.Series(dtype=float, name=ticker)

    out = pd.concat(pieces).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.name = ticker
    return out


def first_working_candidate(candidates, start_ts, end_ts):
    """
    Try multiple symbols and return first one with usable data.
    """
    for ticker in candidates:
        s = download_hourly_series(ticker, start_ts, end_ts)
        if not s.empty and s.notna().sum() > 10:
            return ticker, s
    return None, pd.Series(dtype=float)


def series_to_changes(series, full_index, fill_closed_with_zero=True):
    """
    Convert level series to simple first differences.
    Reindex to the full 24h UTC timeline.
    """
    chg = series.pct_change()
    chg = chg.reindex(full_index)

    if fill_closed_with_zero:
        chg = chg.fillna(0.0)

    return chg


def save_metadata(metadata, filepath):
    with open(filepath, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

def get_pipeline_start():
    """
    If output file exists, continue from a small overlap window
    so recently revised hourly bars can be refreshed.
    Otherwise start from START_UTC.
    """
    start_default = pd.Timestamp(START_UTC, tz=UTC)

    if os.path.exists(OUTPUT_FILE):
        old = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])
        if len(old) > 0:
            last_ts = pd.to_datetime(old["Date"], utc=True).max()
            next_ts = last_ts - pd.Timedelta(hours=24)
            return max(next_ts, start_default)

    return start_default

def run_pipeline():
    start_ts = get_pipeline_start()
    end_ts = now_utc()

    if start_ts > end_ts:
        print("No new data needed. File is already up to date.")
        return None, None, None

    print(f"Updating from {start_ts} to {end_ts}")

    full_index = make_hourly_index(start_ts, end_ts)

    changes_df = pd.DataFrame(index=full_index)
    levels_df = pd.DataFrame(index=full_index)

    used_symbols = {}

    # -------------------------
    # MAIN PRICE ASSETS
    # -------------------------
    for ticker, colname in MAIN_PRICE_TICKERS.items():
        print(f"Pulling {ticker} ...")
        s = download_hourly_series(ticker, start_ts, end_ts)

        if s.empty:
            print(f"  No data for {ticker}")
            changes_df[colname] = np.nan
            levels_df[colname.replace("_chg", "_level")] = np.nan
            continue

        levels_df[colname.replace("_chg", "_level")] = s.reindex(full_index)
        changes_df[colname] = series_to_changes(s, full_index, fill_closed_with_zero=False)
        used_symbols[colname] = ticker

    # -------------------------
    # FUTURES ROBUSTNESS SPEC
    # -------------------------
    for ticker, colname in FUTURES_SPEC.items():
        print(f"Pulling robustness series {ticker} ...")
        s = download_hourly_series(ticker, start_ts, end_ts)

        if s.empty:
            print(f"  No data for {ticker}")
            changes_df[colname] = np.nan
            levels_df[colname.replace("_chg", "_level")] = np.nan
            continue

        levels_df[colname.replace("_chg", "_level")] = s.reindex(full_index)
        changes_df[colname] = series_to_changes(s, full_index, fill_closed_with_zero=False)
        used_symbols[colname] = ticker

    # -------------------------
    # YIELDS
    # -------------------------
    for outcol, candidates in YIELD_CANDIDATES.items():
        chosen, s = first_working_candidate(candidates, start_ts, end_ts)
        used_symbols[outcol] = chosen

        if s.empty:
            print(f"  No working symbol found for {outcol}")
            changes_df[outcol] = np.nan
            levels_df[outcol.replace("_chg", "_level")] = np.nan
            continue

        print(f"{outcol} using {chosen}")
        levels_df[outcol.replace("_chg", "_level")] = s.reindex(full_index)
        changes_df[outcol] = series_to_changes(s, full_index, fill_closed_with_zero=False)

 
    changes_df["is_weekend"] = (changes_df.index.dayofweek >= 5).astype(int)
    changes_df["hour_utc"] = changes_df.index.hour
    changes_df["dow_utc"] = changes_df.index.dayofweek

    changes_df = changes_df.reset_index()
    levels_df = levels_df.reset_index()


    if os.path.exists(OUTPUT_FILE):
        old = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])
        old["Date"] = pd.to_datetime(old["Date"], utc=True)

        combined = pd.concat([old, changes_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date")
    else:
        combined = changes_df.copy()

    if os.path.exists(LEVELS_FILE):
        old_levels = pd.read_csv(LEVELS_FILE, parse_dates=["Date"])
        old_levels["Date"] = pd.to_datetime(old_levels["Date"], utc=True)

        combined_levels = pd.concat([old_levels, levels_df], ignore_index=True)
        combined_levels = combined_levels.drop_duplicates(subset=["Date"], keep="last")
        combined_levels = combined_levels.sort_values("Date")
    else:
        combined_levels = levels_df.copy()

    combined.to_csv(OUTPUT_FILE, index=False)
    combined_levels.to_csv(LEVELS_FILE, index=False)

    metadata = {
        "last_run_utc": str(now_utc()),
        "start_requested": str(start_ts),
        "end_requested": str(end_ts),
        "output_file": OUTPUT_FILE,
        "levels_file": LEVELS_FILE,
        "main_spec": MAIN_PRICE_TICKERS,
        "futures_spec": FUTURES_SPEC,
        "yield_symbols_used": {
            k: used_symbols.get(k) for k in YIELD_CANDIDATES.keys()
        },
        "all_symbols_used": used_symbols
    }
    save_metadata(metadata, METADATA_FILE)

    print("\nDone.")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {LEVELS_FILE}")
    print(f"Saved: {METADATA_FILE}")

    return combined, combined_levels, metadata


df_changes, df_levels, metadata = run_pipeline()

df = pd.read_csv(OUTPUT_FILE, parse_dates=["Date"])
df["Date"] = pd.to_datetime(df["Date"], utc=True)

print(df.tail(20).to_string())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

check_cols = [
    "SPY_chg", "QQQ_chg", "Gold_chg", "Oil_chg", "DXY_chg",
    "BTC_chg", "VIX_chg", "SP500_fut_chg", "US2Y_chg", "US10Y_chg"
]

summary = pd.DataFrame({
    "non_null": df[check_cols].notna().sum(),
    "non_zero": (df[check_cols] != 0).sum(),
    "zero_share": (df[check_cols] == 0).mean()
}).sort_index()

print(summary)
# %%

# %%
# ==========================================
# PHASE : Announcement data added
# ==========================================
def create_master_timeline():
    # 1. Define the timeframe (2025-2026)
    start_date = '2025-01-01 00:00:00'
    end_date = '2026-12-31 23:00:00'

    # 2. Generate the range in UTC
    hourly_index = pd.date_range(start=start_date, end=end_date, freq='h', tz='UTC')

    # 3. Create the DataFrame
    df_master = pd.DataFrame(index=hourly_index)
    
    # 4. Reset index and rename column to "Date" as seen in your image
    df_master = df_master.reset_index()
    df_master = df_master.rename(columns={'index': 'Date'})
    
    # 5. Format to match your image (removing the '+00:00' suffix)
    # DELETED: df_master['Date'] = df_master['Date'].dt.tz_localize(None)
    
    return df_master

# Run and save
df_timeline = create_master_timeline()
df_timeline.to_csv('Announcement_data.csv', index=False)

print("File saved as 'Announcement_data.csv' in the format from your image.")
print(df_timeline.head(3))
# %%

# %%
# ==========================================
# PHASE: THE EVENT REGISTRY (Placeholders)
# ==========================================
# When you get the real dates, just paste them into these lists.
# Format must be 'YYYY-MM-DD'

fomc_dates = ['2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', '2024-09-18', '2024-11-07', '2024-12-18', '2025-01-29', '2025-03-19', '2025-05-07', '2025-06-18', '2025-07-30', '2025-09-17', '2025-10-29', '2025-12-10', '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09']
cpi_dates = ['2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', '2024-06-12', '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', '2024-11-13', '2024-12-11', '2025-01-15', '2025-02-12', '2025-03-12', '2025-04-10', '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12', '2025-09-11', '2025-10-24', '2025-12-18', '2026-01-13', '2026-02-13', '2026-03-11', '2026-04-10', '2026-05-12', '2026-06-10', '2026-07-14', '2026-08-12', '2026-09-11', '2026-10-14', '2026-11-10', '2026-12-10']
emp_dates = ['2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', '2024-06-07', '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', '2024-11-01', '2024-12-06', '2025-01-10', '2025-02-07', '2025-03-07', '2025-04-04', '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01', '2025-09-05', '2025-11-20', '2025-12-16', '2026-01-09', '2026-02-11', '2026-03-06', '2026-04-03', '2026-05-08', '2026-06-05', '2026-07-02', '2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04']
gdp_dates = ['2024-01-25', '2024-02-28', '2024-03-28', '2024-04-25', '2024-05-30', '2024-06-27', '2024-07-25', '2024-08-29', '2024-09-26', '2024-10-30', '2024-11-27', '2024-12-19', '2025-01-30', '2025-02-27', '2025-03-27', '2025-04-30', '2025-05-29', '2025-06-26', '2025-07-30', '2025-08-28', '2025-09-25', '2025-12-23', '2026-01-22', '2026-02-20', '2026-03-13', '2026-04-09', '2026-04-30', '2026-05-28', '2026-06-25', '2026-07-30', '2026-08-26', '2026-09-30', '2026-10-29', '2026-11-25', '2026-12-23']

# ==========================================
# PHASE 2: THE RULE ENGINE
# ==========================================
def process_event_dates():
    # 1. Build a raw list combining the dates, event types, and the New York Time hour rules.
    # Notice we use '08:00:00' to snap the 8:30 AM events to the top of the hour bucket.
    raw_events = []
    
    for d in fomc_dates:
        raw_events.append({'date': d, 'event_type': 'FOMC', 'ny_time': '14:00:00'})
        
    for d in cpi_dates:
        raw_events.append({'date': d, 'event_type': 'CPI', 'ny_time': '08:00:00'})
        
    for d in emp_dates:
        raw_events.append({'date': d, 'event_type': 'Employment', 'ny_time': '08:00:00'})
        
    for d in gdp_dates:
        raw_events.append({'date': d, 'event_type': 'GDP', 'ny_time': '08:00:00'})

    # 2. Convert to a Pandas DataFrame
    df_events = pd.DataFrame(raw_events)
    
    # 3. Combine Date and Time into a single string column
    df_events['datetime_str'] = df_events['date'] + ' ' + df_events['ny_time']
    
    # 4. Turn that string into an actual Pandas Datetime object
    df_events['datetime_obj'] = pd.to_datetime(df_events['datetime_str'])
    
    # 5. THE MAGIC: Tell Pandas this is New York time, then convert it to UTC.
    # This automatically calculates Daylight Saving Time for every individual date.
    df_events['datetime_ny'] = df_events['datetime_obj'].dt.tz_localize('America/New_York', ambiguous='infer')
    df_events['timestamp_utc'] = df_events['datetime_ny'].dt.tz_convert('UTC')
    
    # 6. Preserve UTC timezone instead of removing it
    df_events['Date'] = df_events['timestamp_utc']
    
    # 7. Clean up: Keep only the two columns we need for the merge
    df_final_events = df_events[['Date', 'event_type']]
    
    return df_final_events

# Run the engine
df_processed_events = process_event_dates()

# Print the results to verify
print("Phase 2 Complete. Here is the processed mini-dataset:\n")
print(df_processed_events)
# %%

# %%
# ==========================================
# PHASE 3: THE FINAL MERGE Annnouncement and Assets
# ==========================================

# 1. Merge the Master Timeline with our Processed Events
df_merged = pd.merge(df_timeline, df_processed_events, on='Date', how='left')

# 2. Pivot the 'event_type' into individual columns (Dummies)
# This creates a column for each: FOMC, CPI, Employment, GDP
df_final = pd.get_dummies(df_merged, columns=['event_type'], prefix='Ann')

# 3. Clean up column names (e.g., 'Ann_CPI', 'Ann_FOMC')
# get_dummies puts 0s and 1s, but we need to ensure the hourly rows with NO events stay 0
# Currently, those rows would be all 0s anyway because of the 'left' merge.

# 4. (Optional) If an event appears multiple times in one hour, we sum/max them 
# to keep exactly one row per hour.
df_final = df_final.groupby('Date').max().reset_index()

# Fill any NaNs with 0 (rows where no announcement happened)
df_final = df_final.fillna(0)

# Cast dummy columns to integer to avoid boolean/float errors
for col in df_final.columns:
    if col.startswith('Ann_'):
        df_final[col] = df_final[col].astype(int)

# Save the final product
df_final.to_csv('Final_Announcement_Dummies.csv', index=False)

print("\nMISSION ACCOMPLISHED: 'Final_Announcement_Dummies.csv' is ready for the model.")
print(df_final[df_final['Ann_CPI'] == 1].head()) # Sanity check for a CPI event

# ==========================================
# THE ULTIMATE MERGE
# ==========================================
df_master_merged = pd.merge(df, df_final, how='left', on='Date')
df_master_merged.to_csv('MASTER_ASSET_AND_ANNOUNCEMENTS.csv', index=False)
print("Merged DataFrame saved to MASTER_ASSET_AND_ANNOUNCEMENTS.csv")
# %%