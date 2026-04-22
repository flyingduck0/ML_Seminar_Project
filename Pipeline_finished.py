# %%
# ==========================================
# PHASE 0: ALL IMPORTS & GLOBAL CONFIGURATION
# ==========================================
# --- Standard Library Imports ---
import ast
import json
import os
import random
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from urllib.parse import urlparse

# --- Third-Party Imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
import requests as req
import yfinance as yf
from py_clob_client import ClobClient, OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams, BalanceAllowanceParams, AssetType
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Global Configurations ---
warnings.filterwarnings("ignore")

# ==========================================
# API CONFIGURATION
# ==========================================
Gamma_api = "https://gamma-api.polymarket.com"
Data_api = "https://data-api.polymarket.com"
Clob_api = "https://clob.polymarket.com"

# ==========================================
# ROBUST SESSION INITIALIZATION
# ==========================================
# Initialize the global session here so ALL phases can use it!
session = req.Session()

# Add retry logic to prevent pipeline crashes on API 429 (Rate Limit) errors
retry_strategy = Retry(
    total=5,                  
    backoff_factor=2,         
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
})
# %%

# %%
# ==========================================
# PHASE 1: MACRO EVENTS PIPELINE 
# ==========================================
jsonl_filename = "full_list_events.jsonl"

DISCOVERY_KEYWORDS = [
    "fomc", "federal reserve", "interest rate", "interest rates", "rate cut", 
    "fed rates", "fed", "monetary policy", "unemployment", "nonfarm payroll", 
    "nfp", "jobless claims", "jobs report", "cpi", "core cpi", "us inflation", 
    "inflation", "pce", "personal consumption expenditures", "gdp", "us gdp", 
    "economic growth", "economy", "politics", "business", "finance", "macro", 
    "markets", "us", "global", "government", "elections", "financial"
]

# ==========================================
# STEP 0: PRE-FLIGHT MEMORY CHECK 
# ==========================================
seen_event_ids = set()
seen_tokens = set()

if os.path.exists(jsonl_filename):
    print(f"\nFound existing database: {jsonl_filename}. Loading memory to prevent duplicates...")
    
    with open(jsonl_filename, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip(): 
                continue
                
            try:
                event_data = json.loads(line)
                
                # 1. Memorize Event ID
                e_id = event_data.get('id')
                if e_id: 
                    seen_event_ids.add(str(e_id))
                
                # 2. Memorize Token IDs (Brackets)
                for mkt in event_data.get('markets', []):
                    clob_raw = mkt.get('clobTokenIds', '[]')
                    
                    # Handle Polymarket's stringified lists
                    if isinstance(clob_raw, str):
                        try: 
                            clob_ids = json.loads(clob_raw)
                        except: 
                            clob_ids = []
                    else:
                        clob_ids = clob_raw if isinstance(clob_raw, list) else []
                        
                    for tid in clob_ids:
                        seen_tokens.add(str(tid))
                        
            except json.JSONDecodeError:
                continue
                
    print(f"Memory Loaded: {len(seen_event_ids)} Events | {len(seen_tokens)} Tokens.")
else:
    print("\nNo existing database found. Starting fresh.")

# ==========================================
# STEP 1: TAG DISCOVERY (FULL HISTORICAL PULL)
# ==========================================
print("\nDiscovering ALL Tag IDs from the database...")

target_tag_ids = []
all_tags = []

tag_limit = 100
tag_offset = 0

while True:
    try:
        tags_res = session.get(
            f"{Gamma_api}/tags", 
            params={"limit": tag_limit, "offset": tag_offset}, 
            timeout=15
        ).json()
        
        if not tags_res or len(tags_res) == 0: 
            break 
            
        all_tags.extend(tags_res)
        tag_offset += tag_limit
        
    except Exception as e:
        print(f"❌ Error downloading tags at offset {tag_offset}: {e}")
        break

for t in all_tags:
    t_label = str(t.get('label', '')).lower()
    
    if any(k in t_label for k in DISCOVERY_KEYWORDS):
        if t.get('id'):
            target_tag_ids.append(t.get('id'))

target_tag_ids = list(set(target_tag_ids))
print(f"✅ Found {len(target_tag_ids)} unique tags matching the broadened macro/politics net.")

# ==========================================
# STEP 2: EVENTS INGESTION LOOP 
# ==========================================
print("\nStarting Events download... (Looking for NEW data only)")

new_events_downloaded = 0

# Open in 'a' (append) mode to add to the existing file
with open(jsonl_filename, 'a', encoding='utf-8') as f:
    
    for current_tag_index, t_id in enumerate(target_tag_ids):
        
        if current_tag_index > 0 and current_tag_index % 10 == 0:
            print(f"Processing tag {current_tag_index} of {len(target_tag_ids)}... (New events added: {new_events_downloaded})")
        
        for status in ["true", "false"]:
            limit = 100
            offset = 0 
            
            while True:
                params = {
                    "limit": limit, 
                    "offset": offset, 
                    "tag_id": t_id, 
                    "active": status
                }
                
                try:
                    response = session.get(f"{Gamma_api}/events", params=params, timeout=15)
                    response.raise_for_status()
                    events = response.json()

                    if not events or len(events) == 0: 
                        break
                    
                    for event in events:
                        e_id = str(event.get('id', ''))
                        is_new_data = False
                        
                        # Check 1: Is this a completely new Event?
                        if e_id and e_id not in seen_event_ids:
                            is_new_data = True
                            seen_event_ids.add(e_id)
                            
                        # Check 2: Does this existing Event have new Tokens?
                        for mkt in event.get('markets', []):
                            clob_raw = mkt.get('clobTokenIds', '[]')
                            
                            if isinstance(clob_raw, str):
                                try: 
                                    clob_ids = json.loads(clob_raw)
                                except: 
                                    clob_ids = []
                            else:
                                clob_ids = clob_raw if isinstance(clob_raw, list) else []
                                
                            for tid in clob_ids:
                                tid_str = str(tid)
                                if tid_str not in seen_tokens:
                                    is_new_data = True
                                    seen_tokens.add(tid_str)
                        
                        # Only write to the file if it survived the checks
                        if is_new_data:
                            f.write(json.dumps(event) + '\n')
                            new_events_downloaded += 1

                    if len(events) < limit: 
                        break
                        
                    offset += limit
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"An error occurred on tag {t_id}: {e}")
                    break

print(f"\nFinished! Appended {new_events_downloaded} NEW updates to {jsonl_filename}.")
# %%

# %%
# ==========================================
# PHASE 2: EXTRACTION AND FILTERING 
# ==========================================
"""
Phase 2 parses the raw downloaded events and applies strict slug-matching 
rules to categorize markets into specific macroeconomic pillars (GDP, Labor, 
Inflation, Fed Policy). It filters out pre-2025 data and saves a cleaned catalog.
"""

RAW_EVENTS_FILE = "full_list_events.jsonl"
MACRO_EVENTS_FILE = "PILLAR_MACRO_EVENTS.jsonl"

CUTOFF_DATE = "2025-01-01"

MONTHS = [
    'january', 'february', 'march', 'april', 'may', 'june', 
    'july', 'august', 'september', 'october', 'november', 'december'
]

def categorize_market(slug):
    """
    Applies the strict slug patterns to categorize the market.
    
    Args:
        slug (str): The raw Polymarket event slug.
        
    Returns:
        tuple: (macro_pillar, sub_category) or (None, None) if no match.
    """
    if not isinstance(slug, str):
        return None, None
        
    slug = slug.lower().strip()
    
    # --- 1. GDP GROWTH ---
    if slug.startswith("us-gdp-growth-in-q"):
        return "GDP_GROWTH", "GDP_QUARTERLY"
        
    # --- 2. LABOR MARKET (UNEMPLOYMENT) ---
    if "india" in slug or "indian" in slug:
        pass  # Explicit exclusion
    elif "unemployment-rate" in slug:
        if any(m in slug for m in MONTHS):
            return "LABOR_MARKET", "UNEMPLOYMENT_MONTHLY"
            
    # --- 3. INFLATION ---
    if slug.startswith("how-high-will-inflation-get-in-"):
        return "INFLATION", "INFLATION_YEAR_ANCHOR"
    if "-inflation-annual" in slug or "-inflation-us-annual" in slug:
        return "INFLATION", "INFLATION_MONTH_YOY"
        
    # --- 4. FED POLICY ---
    if "fed-interest-rates-" in slug or "fed-decision-in-" in slug:
        if any(m in slug for m in MONTHS):
            return "FED_POLICY", "FED_MEETING_DECISION"
            
    return None, None

print("\n--- PART 2: THE STRICT GOLDEN CATALOG EXTRACTION ---")

if not os.path.exists(RAW_EVENTS_FILE):
    print(f"[Error] {RAW_EVENTS_FILE} not found. Run Phase 1 first.")
else:
    tracking_dict = {}
    total_scanned = 0
    dropped_date = 0
    
    # To track exactly what we extracted
    category_counts = defaultdict(int)
    
    print("Scanning local database offline with strict slug patterns...")
    
    with open(RAW_EVENTS_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Skip empty lines
            if not line.strip(): 
                continue
                
            # Safely parse JSON
            try: 
                event = json.loads(line)
            except json.JSONDecodeError: 
                continue
                
            total_scanned += 1
            event_id = str(event.get('id', ''))
            event_slug = str(event.get('slug', '')).strip().lower()
            
            # FILTER 1: Date Cutoff (Check if endDate exists and is >= 2025)
            end_date_str = str(event.get('endDate', ''))
            if not end_date_str or end_date_str[:10] < CUTOFF_DATE:
                dropped_date += 1
                continue
            
            # FILTER 2 & 3: Strict Category Logic (Includes the India exclusion)
            pillar, sub_category = categorize_market(event_slug)
            
            # If it passed the strict slug tests, save it to our dictionary
            if pillar and sub_category and event_id:
                event['macro_pillar'] = pillar
                event['sub_category'] = sub_category
                
                # Dictionary inherently handles deduplication (keeps newest iteration)
                tracking_dict[event_id] = event

    # Save everything into ONE single JSONL file
    with open(MACRO_EVENTS_FILE, 'w', encoding='utf-8') as macro_out:
        for event_id, event in tracking_dict.items():
            macro_out.write(json.dumps(event) + '\n')
            
            # Track for the terminal report
            category_counts[event['sub_category']] += 1
            
    # Terminal Audit Report
    print("\n" + "=" * 50)
    print("GOLDEN CATALOG EXTRACTION REPORT")
    print("=" * 50)
    print(f"Total Raw Events Scanned:  {total_scanned}")
    print(f"Dropped (Before 2025):     {dropped_date}")
    print(f"Total Target Events Kept:  {len(tracking_dict)}")
    print("-" * 50)
    print("SUB-CATEGORY BREAKDOWN:")
    
    for cat, count in sorted(category_counts.items()):
        print(f"  -> {cat}: {count} events")
        
    print("=" * 50)
    print(f"Saved highly filtered, single dataset to: {MACRO_EVENTS_FILE}")
# %%

# %%
# ==========================================
# PHASE 3: FLATTENING MARKETS
# ==========================================
"""
Phase 3 takes the nested JSON structure of the filtered macro events 
and flattens it. It extracts the individual nested markets (brackets) 
for each event so they can be processed individually by the Polymarket API.
"""

MACRO_EVENTS_FILE = "PILLAR_MACRO_EVENTS.jsonl"
FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"

def flatten_macro_markets():
    if not os.path.exists(MACRO_EVENTS_FILE):
        print(f"[Error] Could not find {MACRO_EVENTS_FILE}")
        return

    print("\n--- PART 3: FLATTENING THE GOLDEN CATALOG (PHASE 3) ---")
    
    total_events_processed = 0
    total_markets_flattened = 0
    
    # We use a list to keep track of what we're about to write for the summary
    flat_records = []

    with open(MACRO_EVENTS_FILE, 'r', encoding='utf-8') as infile, \
         open(FLAT_MARKETS_FILE, 'w', encoding='utf-8') as flat_out:
        
        for line in infile:
            # Skip empty lines
            if not line.strip(): 
                continue
                
            # Safely parse JSON
            try:
                event = json.loads(line)
            except:
                continue
            
            total_events_processed += 1
            
            # Pull our custom labels from Phase 2
            event_id     = event.get('id')
            macro_pillar = event.get('macro_pillar')
            sub_category = event.get('sub_category')
            event_title  = event.get('title')
            
            markets = event.get('markets', [])
            
            if isinstance(markets, list):
                for mkt in markets:
                    # Extract the critical CLOB/Data IDs
                    market_id        = mkt.get("id")
                    condition_id     = mkt.get("conditionId")
                    group_item_title = mkt.get("groupItemTitle")
                    
                    if not market_id or not condition_id:
                        continue
                    
                    flat_record = {
                        "event_id":         event_id,
                        "macro_pillar":     macro_pillar,
                        "sub_category":     sub_category,
                        "market_id":        market_id,
                        "condition_id":     condition_id,
                        "event_title":      event_title,
                        "group_item_title": group_item_title,
                        "slug":             mkt.get("slug")
                    }
                    
                    flat_out.write(json.dumps(flat_record) + '\n')
                    flat_records.append(flat_record)
                    total_markets_flattened += 1

    print("✅ Flattening Complete:")
    print(f"  -> Events Processed: {total_events_processed}")
    print(f"  -> Individual Markets Extracted: {total_markets_flattened}")
    print(f"  -> Output: {FLAT_MARKETS_FILE}")

flatten_macro_markets()
# %%

# %%
# ==========================================
# PHASE 4: SMART SYNC (JSONL EDITION)
# ==========================================
"""
Phase 4 performs a delta sync against the Polymarket API. It loads previously 
synced markets into memory and only executes new API calls for markets that are 
new or still actively trading, preserving API limits and historical data.
"""

FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"
SYNCED_MARKET_FILE = "SYNCED_MARKET_DATA.jsonl"

def fetch_synced_macro_data():
    if not os.path.exists(FLAT_MARKETS_FILE):
        print(f"[Error] {FLAT_MARKETS_FILE} not found.")
        return
        
    print("\n--- PART 4: DELTA SYNC (ACTIVE + HISTORICAL) ---")

    # 1. Load what we already have into memory
    existing_data = {}
    if os.path.exists(SYNCED_MARKET_FILE):
        with open(SYNCED_MARKET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if not line.strip(): 
                    continue
                    
                d = json.loads(line)
                existing_data[str(d.get("id"))] = d

    # 2. Prepare the Flat List for processing
    with open(FLAT_MARKETS_FILE, 'r', encoding='utf-8') as f:
        target_records = [json.loads(line) for line in f]

    print(f"Syncing {len(target_records)} total markets (Active & Non-Active)...")

    new_api_calls = 0
    reused_historical = 0

    with open(SYNCED_MARKET_FILE, 'w', encoding='utf-8') as outfile:
        for i, record in enumerate(target_records):
            m_id = str(record["market_id"])
            
            # THE DELTA CHECK:
            # We only skip the API call if:
            # 1. We already have it in our sync file
            # 2. AND it was already 'closed' (Non-Active)
            cached = existing_data.get(m_id)
            if cached and cached.get("closed") is True:
                outfile.write(json.dumps(cached) + '\n')
                reused_historical += 1
                continue
            
            # OTHERWISE: It's either New OR it's still Active (needs a volume update)
            try:
                response = session.get(f"{Gamma_api}/markets/{m_id}", timeout=15)
                
                if response.status_code == 200:
                    api_data = response.json()
                    
                    # Keep our labels from Phase 2
                    api_data["macro_pillar"] = record.get("macro_pillar")
                    api_data["sub_category"] = record.get("sub_category")
                    
                    outfile.write(json.dumps(api_data) + '\n')
                    new_api_calls += 1
                
                if (i + 1) % 50 == 0:
                    print(f"  Processed {i + 1}/{len(target_records)}... (Updated: {new_api_calls} | Cached: {reused_historical})")
                
                time.sleep(0.2)
                
            except Exception as e:
                # If API fails, keep old data if we have it, otherwise skip
                if cached: 
                    outfile.write(json.dumps(cached) + '\n')
                    
                print(f"  Error on Market {m_id}: {e}")

    print("\n✅ Sync Complete!")
    print(f"  -> Total Markets in Catalog: {new_api_calls + reused_historical}")
    print(f"  -> Fresh Updates Pulled: {new_api_calls}")
    print(f"  -> Historical Markets Preserved: {reused_historical}")

fetch_synced_macro_data()
# %%

# %%
# ==========================================
# PHASE 5: THE RESTORED IDENTITY REFINERY (2025+ Edition)
# ==========================================
"""
Phase 5 acts as the final quality gate before historical data extraction. 
It reads the synced data, purges irrelevant 2024 noise and foreign markets, 
and applies strict string-matching to assign the final feature labels 
before splitting the data into distinct pillar CSVs.
"""

SYNCED_FILE = "SYNCED_MARKET_DATA.jsonl"
DROPPED_FILE = "DROPPED_MARKETS.csv"

def run_final_refinery(min_volume=0): 
    print("\n--- PART 5: REFINERY (2025+ FOCUS / KILLING 2024 NOISE) ---")
    
    clean_records = []
    dropped_audit = []
    FOREIGN_EXCLUSIONS = ["japan", "canada", "india", "uk", "brazil", "mexico", "euro"]

    with open(SYNCED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip empty lines
            if not line.strip(): 
                continue
                
            mkt = json.loads(line)
            slug = str(mkt.get("slug", "")).lower()
            
            # --- 1. THE 2024 EXORCISM ---
            # Direct removal of historical data to focus on the 2025-2026 cycle.
            if "2024" in slug:
                dropped_audit.append({
                    "id": mkt.get("id"), 
                    "slug": slug, 
                    "reason": "Historical Noise (2024)"
                })
                continue

            # --- 2. GEOGRAPHIC BLOCK (Removes foreign noise) ---
            if any(country in slug for country in FOREIGN_EXCLUSIONS):
                dropped_audit.append({
                    "id": mkt.get("id"), 
                    "slug": slug, 
                    "reason": "Foreign Geography"
                })
                continue

            # --- 3. EXPANDED US IDENTITY PATTERNS ---
            is_us_macro = False
            feature = "OTHER"

            # GDP Patterns
            # Tightened to only accept Quarterly GDP; Yearly is now filtered out
            if slug.startswith("us-gdp-growth-in-q") or ("us-gdp" in slug and "in-q" in slug):
                is_us_macro = True
                feature = "GDP_QUARTERLY"
            
            # Labor Patterns (Preserves Jan/Feb 2025 which lack year tags)
            elif "unemployment" in slug:
                if slug.startswith("will-the-") or slug.startswith("will-us-unemployment-") or slug.startswith("us-"):
                    is_us_macro = True
                    feature = "UNEMPLOYMENT_MONTHLY"
                
            # Inflation Patterns
            elif "inflation" in slug or "pce" in slug:
                if (slug.startswith("will-the-") or 
                    slug.startswith("will-inflation-") or 
                    slug.startswith("will-annual-inflation-") or 
                    "how-high" in slug or 
                    "-us-annual" in slug or 
                    "-inflation-annual" in slug):
                    
                    is_us_macro = True
                    feature = "INFLATION_YEAR_ANCHOR" if "how-high" in slug else "INFLATION_MONTH_YOY"

            # Fed Policy Patterns
            elif (slug.startswith("fed-") or 
                  slug.startswith("no-change-") or 
                  slug.startswith("will-the-fed-") or 
                  slug.startswith("will-there-be-no-change-in-fed-")):
                  
                is_us_macro = True
                feature = "FED_MEETING_DECISION"

            # FINAL ACTION
            if is_us_macro:
                mkt["specific_feature"] = feature
                mkt["total_volume_usd"] = float(mkt.get("volumeAmm") or 0) + float(mkt.get("volumeClob") or 0)
                clean_records.append(mkt)
            else:
                dropped_audit.append({
                    "id": mkt.get("id"), 
                    "slug": slug, 
                    "reason": "Failed Pattern Filter"
                })

    # Export to CSV
    df = pd.DataFrame(clean_records)
    pd.DataFrame(dropped_audit).to_csv(DROPPED_FILE, index=False)

    pillar_map = {
        "GDP": "GDP_GROWTH", 
        "INFLATION": "INFLATION", 
        "LABOR": "LABOR_MARKET", 
        "FED": "FED_POLICY"
    }
    
    # Ensure clobTokenIds is included here so it's ready for Phase 6
    cols = ['id', 'clobTokenIds', 'macro_pillar', 'specific_feature', 'question', 'slug', 'total_volume_usd', 'endDate', 'resolved']
    available_cols = [c for c in cols if c in df.columns]

    for label, pillar_name in pillar_map.items():
        subset = df[df['macro_pillar'] == pillar_name].copy()
        
        if not subset.empty:
            subset.sort_values(by='endDate').to_csv(f"FINAL_{label}_MARKETS.csv", index=False)
            print(f"  -> Restored Export: FINAL_{label}_MARKETS.csv ({len(subset)} markets)")

    print(f"\n✅ Total US Markets Safely Exported: {len(clean_records)}")
    print(f"✅ Total Dropped (Foreign/2024/Junk): {len(dropped_audit)}")

run_final_refinery()
# %%

# %%
# ==========================================
# PHASE 6: FINAL FEATURE LABELING & CLEANUP
# ==========================================
"""
Phase 6 finalizes the individual pillar datasets. 
Step 1 promotes the correct sub-category labels, sorts the markets chronologically, 
and audits the counts. 
Step 2 specifically isolates the inflation markets, splitting them into 
Yearly and Monthly datasets for specialized downstream processing.
"""

# ==========================================
# STEP 1: RUNNING FINAL CLEANUP & SORTING
# ==========================================
csv_files = [
    "FINAL_GDP_MARKETS.csv", 
    "FINAL_INFLATION_MARKETS.csv", 
    "FINAL_LABOR_MARKETS.csv", 
    "FINAL_FED_MARKETS.csv"
]

print("--- STEP 1: RUNNING FINAL CLEANUP & SORTING ---")

for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        
        # Guard to warn you if you forgot to run Phase 5 first
        if len(df) == 0:
            print(f"⚠️ {file} is empty! Please re-run Phase 5 to restore the data.")
            continue
            
        # 1. THE REVELATION: Phase 2 already perfectly labeled everything!
        # We completely bypass market-slug guessing and promote the true label.
        df['feature'] = df['sub_category']
        
        # 2. CHRONOLOGICAL SORT: Order by endDate
        if 'endDate' in df.columns:
            df['endDate'] = pd.to_datetime(df['endDate'])
            df = df.sort_values(by='endDate')
            
        # 3. Save the finalized table
        df.to_csv(file, index=False)
        
        # Print the audit report to the terminal
        print(f"\n✅ {file} - Total Clean Markets: {len(df)}")
        
        for feature, count in df['feature'].value_counts().items():
            print(f"  -> {feature}: {count}")
            
    else:
        print(f"⚠️ {file} not found. Skipping.")

# ==========================================
# STEP 2: SPLIT INFLATION BY SUB-CATEGORY
# ==========================================
print("\n--- STEP 2: SPLITTING INFLATION MARKETS ---")

# We load the file we just saved and split it using the correct Phase 2 labels.
df_inf = pd.read_csv("FINAL_INFLATION_MARKETS.csv")

# Yearly Anchor = Slugs like 'will-inflation-reach-more-than-6-in-2025'
df_inf[df_inf['sub_category'] == 'INFLATION_YEAR_ANCHOR'].to_csv("FINAL_INFLATION_YEARLY.csv", index=False)

# Monthly YoY = Slugs like 'will-annual-inflation-increase-by-2pt6-in-january'
df_inf[df_inf['sub_category'] == 'INFLATION_MONTH_YOY'].to_csv("FINAL_INFLATION_MONTHLY_YOY.csv", index=False)

print("✅ Split Complete:")
print(f"  -> FINAL_INFLATION_YEARLY.csv: {len(df_inf[df_inf['sub_category'] == 'INFLATION_YEAR_ANCHOR'])} markets")
print(f"  -> FINAL_INFLATION_MONTHLY_YOY.csv: {len(df_inf[df_inf['sub_category'] == 'INFLATION_MONTH_YOY'])} markets")
# %%

# %%
# ==========================================
# PHASE 7: ADDITIVE INCREMENTAL VACUUM HOURLY DATA
# ==========================================
"""
Phase 7 incrementally pulls the full CLOB price history for the 'Yes' tokens 
across all filtered markets. It checks local storage to resume downloading 
only new historical data, minimizing API load and execution time.
"""

macro_pillars = {
    "FED": "FINAL_FED_MARKETS.csv",
    "GDP": "FINAL_GDP_MARKETS.csv",
    "LABOR": "FINAL_LABOR_MARKETS.csv",
    "INF_YEARLY": "FINAL_INFLATION_YEARLY.csv",
    "INF_MONTHLY": "FINAL_INFLATION_MONTHLY_YOY.csv"
}

def parse_tokens(val):
    """Safely extracts ONLY the first token ID (The 'YES' probability)."""
    try:
        data = ast.literal_eval(val)
        # CRITICAL FIX: Only return the first ID in the list (Index 0 is ALWAYS 'Yes')
        return [data[0]] if isinstance(data, list) and len(data) > 0 else []
    except Exception:
        try:
            data = json.loads(val)
            return [data[0]] if isinstance(data, list) and len(data) > 0 else []
        except Exception:
            return []

def vacuum_token_incremental(tid, slug, start_ts):
    """Pulls history forward starting from a specific timestamp."""
    all_history = []
    current_pointer = start_ts
    
    while True:
        url = f"https://clob.polymarket.com/prices-history?market={tid}&fidelity=60&startTs={current_pointer}"
        try:
            # Updated to 'req.get' to match the global import alias from Phase 0
            r = req.get(url, timeout=15)
            
            if r.status_code != 200: 
                break
            
            data = r.json().get('history', [])
            if not data: 
                break 
            
            chunk_df = pd.DataFrame(data)
            all_history.append(chunk_df)
            
            last_ts = int(chunk_df['t'].max())
            if last_ts <= current_pointer: 
                break
            
            current_pointer = last_ts + 1
            time.sleep(0.1)
            
            if last_ts > (time.time() - 7200): 
                break
                
        except Exception:
            break
            
    return pd.concat(all_history, ignore_index=True) if all_history else pd.DataFrame()

# ==========================================
# MAIN EXECUTION LOOP
# ==========================================
for pillar, csv_file in macro_pillars.items():
    if not os.path.exists(csv_file): 
        continue
        
    output_name = f"TOTAL_HISTORY_{pillar}.csv"
    last_timestamps = {}
    
    if os.path.exists(output_name):
        existing_data = pd.read_csv(output_name, usecols=['token_id', 't'])
        existing_data['t_numeric'] = pd.to_datetime(existing_data['t']).astype(int) // 10**9
        last_timestamps = existing_data.groupby('token_id')['t_numeric'].max().to_dict()
        print(f"\n📂 SYNCING PILLAR: {pillar} (Incremental)")
    else:
        print(f"\n📂 SYNCING PILLAR: {pillar} (New Database)")

    df_ref = pd.read_csv(csv_file)
    new_pillar_data = []

    for idx, row in df_ref.iterrows():
        tokens = parse_tokens(row['clobTokenIds'])
        
        if not tokens: 
            continue
        
        # Now 'tokens' only contains ONE ID (The Yes token)
        tid = tokens[0]
            
        if tid in last_timestamps:
            start_point = int(last_timestamps[tid]) + 1
        else:
            try:
                start_point = int(pd.to_datetime(row.get('createdAt')).timestamp())
            except:
                start_point = int(time.time()) - (90 * 24 * 60 * 60)

        mkt_new_history = vacuum_token_incremental(tid, row['slug'], start_point)
        
        if not mkt_new_history.empty:
            mkt_new_history['token_id'] = tid
            mkt_new_history['slug'] = row['slug']
            # Normalizing titles to prevent "No Change" duplicates
            mkt_new_history['groupItemTitle'] = str(row.get('groupItemTitle', 'N/A')).strip().title()
            new_pillar_data.append(mkt_new_history)

    if new_pillar_data:
        final_df = pd.concat(new_pillar_data, ignore_index=True)
        final_df['t'] = pd.to_datetime(final_df['t'], unit='s', utc=True)
        
        # Apply the floor fix so every row hits the exact hour mark
        final_df['t'] = final_df['t'].dt.floor('h')
        
        file_exists = os.path.exists(output_name)
        final_df.to_csv(output_name, mode='a', header=not file_exists, index=False)
        print(f"🏁 PILLAR {pillar} UPDATED: Added {len(final_df)} 'Yes' probability rows.")
# %%

# %%
# ==========================================
# PHASE 8A: TRANSPARENT FED PIPELINE
# ==========================================
"""
Phase 8A processes the Federal Reserve interest rate markets.
It generates hourly probability grids, applies quality filtering bounds,
and calculates a normalized Expected Value (EV) to serve as a clean macro signal.
"""

print("--- STARTING TRANSPARENT FED PIPELINE: STEPS 1-4 ---")

history_file = 'TOTAL_HISTORY_FED.csv'
reference_file = 'FINAL_FED_MARKETS.csv'
output_file = 'FINAL_FED_SIGNAL.csv'

LOWER_BOUND = 0.90
UPPER_BOUND = 1.10

if not os.path.exists(history_file) or not os.path.exists(reference_file):
    print("❌ Error: Files not found.")
else:
    # Load Data
    df_history = pd.read_csv(history_file)
    df_ref = pd.read_csv(reference_file)
    df_history['t'] = pd.to_datetime(df_history['t'], utc=True)

    # --- STEP 1: TIME FILTER (YOUR ORIGINAL LOGIC) ---
    df_history = df_history[df_history['t'].dt.minute < 5].copy()
    mapping = df_ref[['slug', 'endDate']].drop_duplicates()
    df_mapped = df_history.merge(mapping, on='slug', how='left')
    df_mapped['t_hour'] = df_mapped['t'].dt.floor('h')

    # --- STEP 2: GRID GENERATION (YOUR ORIGINAL LOGIC) ---
    all_meeting_grids = []
    unique_meetings = df_mapped['endDate'].dropna().unique()
    
    for meeting_date in sorted(unique_meetings):
        m_data = df_mapped[df_mapped['endDate'] == meeting_date]
        m_pivot = m_data.pivot_table(index='t_hour', columns='groupItemTitle', values='p', aggfunc='last')
        
        # Timeline strictly from first to last trade
        full_range = pd.date_range(start=m_pivot.index.min(), end=m_pivot.index.max(), freq='h', tz='UTC')
        m_pivot = m_pivot.reindex(full_range).ffill()
        
        # Cleanup column noise
        m_pivot = m_pivot.loc[:, m_pivot.columns.notna()]
        
        if 'Nan' in m_pivot.columns: 
            m_pivot = m_pivot.drop(columns=['Nan'])
        
        # Calculate Sum for Step 3
        bracket_cols = [c for c in m_pivot.columns if c not in ['meeting_date']]
        m_pivot['RAW_TOTAL_SUM'] = m_pivot[bracket_cols].sum(axis=1)
        m_pivot['meeting_date'] = meeting_date
        
        all_meeting_grids.append(m_pivot.reset_index().rename(columns={'index': 't'}))

    # --- STEP 3: QUALITY FILTERING (YOUR ORIGINAL LOGIC) ---
    full_grid = pd.concat(all_meeting_grids, ignore_index=True)
    
    df_clean = full_grid[
        (full_grid['RAW_TOTAL_SUM'] >= LOWER_BOUND) & 
        (full_grid['RAW_TOTAL_SUM'] <= UPPER_BOUND)
    ].copy()

    # --- STEP 4: AUTOMATED EV & TRANSPARENT NORMALIZATION ---
    metadata = ['t', 'meeting_date', 'RAW_TOTAL_SUM']
    prob_cols = [c for c in df_clean.columns if c not in metadata]
    
    # Map weights
    weights = {}
    for col in prob_cols:
        numbers = re.findall(r'\d+', str(col))
        val = int(numbers[0]) if numbers else 0
        
        if "decrease" in col.lower() or "cut" in col.lower(): 
            weights[col] = -val
        elif "increase" in col.lower() or "hike" in col.lower(): 
            weights[col] = val
        else: 
            weights[col] = 0

    # 4A. Calculate RAW_EV (The "dirty" weighted sum)
    df_clean['RAW_EV'] = 0
    for col, weight in weights.items():
        df_clean['RAW_EV'] += df_clean[col].fillna(0) * weight
    
    # 4B. Calculate NORMALIZED_EV (The final "clean" signal)
    # This is the division that protects your deltas from sum-drift noise
    df_clean['NORMALIZED_EV'] = df_clean['RAW_EV'] / df_clean['RAW_TOTAL_SUM']

    # Export
    final_cols = ['t', 'meeting_date', 'NORMALIZED_EV', 'RAW_EV', 'RAW_TOTAL_SUM'] + prob_cols
    df_clean[final_cols].to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! File saved: {output_file}")
    print("Side-by-side comparison (Raw vs Normalized):")
    print(df_clean[['t', 'RAW_EV', 'NORMALIZED_EV', 'RAW_TOTAL_SUM']].head())
# %%

# %%
# ==========================================
# PHASE 8B: TRANSPARENT GDP PIPELINE
# ==========================================
"""
Phase 8B processes the US GDP Growth markets.
It handles parallel market sets (negRisk mapping), generates hourly probability 
grids, extracts numeric boundaries from complex slugs, and calculates a 
Base-100 Expected GDP Index signal.
"""

print("--- STARTING TRANSPARENT GDP PIPELINE: STEPS 1-4 ---")

history_file = 'TOTAL_HISTORY_GDP.csv'
reference_file = 'FINAL_GDP_MARKETS.csv'
output_file = 'FINAL_GDP_SIGNAL.csv'

LOWER_BOUND = 0.90
UPPER_BOUND = 1.10

if not os.path.exists(history_file) or not os.path.exists(reference_file):
    print("❌ Error: Files not found.")
else:
    # Load Data
    df_history = pd.read_csv(history_file)
    df_ref = pd.read_csv(reference_file)
    df_history['t'] = pd.to_datetime(df_history['t'], utc=True)

    # --- STEP 1: TIME FILTER & MAPPING ---
    df_history = df_history[df_history['t'].dt.minute < 5].copy()
    
    # GDP Parsing Logic
    def get_event_from_slug(slug):
        m = re.search(r'(q\d-\d{4})', slug)
        return m.group(1).upper() if m else 'UNKNOWN'

    df_ref['event'] = df_ref['slug'].apply(get_event_from_slug)
    
    # Mapping dictionaries (Using internal Polymarket IDs)
    slug_to_negRisk = dict(zip(df_ref['slug'], df_ref['negRiskMarketID']))
    slug_to_event = dict(zip(df_ref['slug'], df_ref['event']))

    df_history['event'] = df_history['slug'].map(slug_to_event)
    df_history['market_set_id'] = df_history['slug'].map(slug_to_negRisk)
    
    df_mapped = df_history.dropna(subset=['market_set_id', 'event']).copy()
    df_mapped['t_hour'] = df_mapped['t'].dt.floor('h')

    # --- STEP 2: GRID GENERATION (PARALLEL MARKETS) ---
    all_market_grids = []
    unique_events = df_mapped['event'].unique()
    
    for event in sorted(unique_events):
        e_data = df_mapped[df_mapped['event'] == event]
        
        for set_id in e_data['market_set_id'].unique():
            set_data = e_data[e_data['market_set_id'] == set_id]
            m_pivot = set_data.pivot_table(index='t_hour', columns='slug', values='p', aggfunc='last')
            
            # Timeline strictly from first to last trade
            full_range = pd.date_range(start=m_pivot.index.min(), end=m_pivot.index.max(), freq='h', tz='UTC')
            m_pivot = m_pivot.reindex(full_range).ffill()
            
            # Cleanup column noise
            m_pivot = m_pivot.loc[:, m_pivot.columns.notna()]
            
            if 'Nan' in m_pivot.columns: 
                m_pivot = m_pivot.drop(columns=['Nan'])
            
            # Calculate Sum for Step 3
            bracket_cols = [c for c in m_pivot.columns]
            m_pivot['RAW_TOTAL_SUM'] = m_pivot[bracket_cols].sum(axis=1)
            m_pivot['event'] = event
            m_pivot['market_set_id'] = set_id
            
            all_market_grids.append(m_pivot.reset_index().rename(columns={'index': 't'}))

    # --- STEP 3: QUALITY FILTERING ---
    full_grid = pd.concat(all_market_grids, ignore_index=True)
    
    df_clean = full_grid[
        (full_grid['RAW_TOTAL_SUM'] >= LOWER_BOUND) & 
        (full_grid['RAW_TOTAL_SUM'] <= UPPER_BOUND)
    ].copy()

    # --- STEP 4: AUTOMATED EV & TRANSPARENT NORMALIZATION ---
    metadata = ['t', 'event', 'market_set_id', 'RAW_TOTAL_SUM']
    prob_cols = [c for c in df_clean.columns if c not in metadata]
    
    # Map weights
    def get_weight(slug):
        event_match = re.search(r'(q\d-\d{4})', slug)
        ev = event_match.group(1) if event_match else 'unknown'
        
        bracket_part = slug.replace(ev, '').replace('pt', '.')
        nums = re.findall(r'(\d+(?:\.\d+)?)', bracket_part)
        nums = [float(n) for n in nums]
        
        if 'between' in slug and len(nums) >= 2:
            is_legacy = (ev in ['q1-2025', 'q2-2025']) and ('pt' not in slug)
            mid = (nums[0] + nums[1]) / 2
            return -mid if (is_legacy and nums[0] < nums[1]) else mid
            
        elif 'less-than' in slug and nums:
            is_legacy = (ev in ['q1-2025', 'q2-2025']) and ('pt' not in slug)
            return -nums[0] if is_legacy else nums[0]
            
        elif ('greater-than' in slug or 'more' in slug) and nums:
            return nums[0]
            
        return 0

    weights = {col: get_weight(col) for col in prob_cols}

    # 4A. Calculate RAW_EV (The "dirty" weighted sum)
    df_clean['RAW_EV'] = 0.0
    for col, weight in weights.items():
        df_clean['RAW_EV'] += df_clean[col].fillna(0) * weight
    
    # 4B. Calculate NORMALIZED_EV (The final "clean" signal percentage)
    df_clean['NORMALIZED_EV'] = df_clean['RAW_EV'] / df_clean['RAW_TOTAL_SUM']

    # 4C. Calculate BASE-100 INDEX (The fix for macro comparisons)
    df_clean['EXPECTED_GDP_INDEX'] = 100 + df_clean['NORMALIZED_EV']

    # Export
    final_cols = ['t', 'event', 'market_set_id', 'EXPECTED_GDP_INDEX', 'NORMALIZED_EV', 'RAW_EV', 'RAW_TOTAL_SUM'] + prob_cols
    df_clean = df_clean.sort_values(['event', 't'])
    df_clean[final_cols].to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! File saved: {output_file}")
    print("Side-by-side comparison (Raw EV vs Normalized % vs Base-100 Index):")
    print(df_clean[['t', 'RAW_EV', 'NORMALIZED_EV', 'EXPECTED_GDP_INDEX']].head())

print("=========================================================")
# %%

# %%
# ==========================================
# PHASE 8C: TRANSPARENT LABOR PIPELINE
# ==========================================
"""
Phase 8C processes the Labor Market (Unemployment) data.
It maps target unemployment releases, generates hourly probability matrices, 
applies quality bounds, and computes both an expected unemployment rate 
and a Base-100 index for macro integration.
"""

print("--- STARTING TRANSPARENT LABOR PIPELINE: STEPS 1-4 ---")

history_file = 'TOTAL_HISTORY_LABOR.csv'
reference_file = 'FINAL_LABOR_MARKETS.csv'
output_file = 'FINAL_LABOR_SIGNAL.csv'

# Quality Gate adjusted to 0.90 - 1.10 to improve data retention while filtering structural errors
LOWER_BOUND = 0.90
UPPER_BOUND = 1.10

if not os.path.exists(history_file) or not os.path.exists(reference_file):
    print("❌ Error: Files not found.")
else:
    # Load Data
    df_history = pd.read_csv(history_file)
    df_ref = pd.read_csv(reference_file)
    df_history['t'] = pd.to_datetime(df_history['t'], utc=True)

    # --- STEP 1: TIME FILTER & MAPPING ---
    df_history = df_history[df_history['t'].dt.minute < 5].copy()
    
    # Map directly to endDate to identify the target unemployment release
    mapping = df_ref[['slug', 'endDate']].drop_duplicates()
    df_mapped = df_history.merge(mapping, on='slug', how='left')
    df_mapped['t_hour'] = df_mapped['t'].dt.floor('h')

    # --- STEP 2: GRID GENERATION ---
    all_grids = []
    unique_events = df_mapped['endDate'].dropna().unique()
    
    for event in sorted(unique_events):
        m_data = df_mapped[df_mapped['endDate'] == event]
        m_pivot = m_data.pivot_table(index='t_hour', columns='groupItemTitle', values='p', aggfunc='last')
        
        # Timeline strictly from first to last trade
        full_range = pd.date_range(start=m_pivot.index.min(), end=m_pivot.index.max(), freq='h', tz='UTC')
        m_pivot = m_pivot.reindex(full_range).ffill()
        
        # Cleanup column noise
        m_pivot = m_pivot.loc[:, m_pivot.columns.notna()]
        
        if 'Nan' in m_pivot.columns: 
            m_pivot = m_pivot.drop(columns=['Nan'])
        
        # Calculate Total Sum for normalization and quality filtering
        bracket_cols = [c for c in m_pivot.columns]
        m_pivot['RAW_TOTAL_SUM'] = m_pivot[bracket_cols].sum(axis=1)
        m_pivot['endDate'] = event
        
        all_grids.append(m_pivot.reset_index().rename(columns={'index': 't'}))

    # --- STEP 3: QUALITY FILTERING ---
    full_grid = pd.concat(all_grids, ignore_index=True)
    
    df_clean = full_grid[
        (full_grid['RAW_TOTAL_SUM'] >= LOWER_BOUND) & 
        (full_grid['RAW_TOTAL_SUM'] <= UPPER_BOUND)
    ].copy()

    # --- STEP 4: AUTOMATED EV, NORMALIZATION & INDEX CONVERSION ---
    metadata = ['t', 'endDate', 'RAW_TOTAL_SUM']
    prob_cols = [c for c in df_clean.columns if c not in metadata]
    
    # The clean Labor Parser: Extracts numerical values from bracket titles
    def get_weight(title):
        nums = re.findall(r'(\d+\.\d+)', str(title))
        return float(nums[0]) if nums else 0.0

    weights = {col: get_weight(col) for col in prob_cols}

    # 4A. Calculate RAW_EV (Dirty weighted sum)
    df_clean['RAW_EV'] = 0.0
    for col, weight in weights.items():
        df_clean['RAW_EV'] += df_clean[col].fillna(0) * weight
    
    # 4B. Calculate EXPECTED UNEMPLOYMENT RATE (Clean normalized percentage)
    # Normalization divides the weighted sum by the total probability to correct for spread drift
    df_clean['EXPECTED_UNEMPLOYMENT_RATE'] = df_clean['RAW_EV'] / df_clean['RAW_TOTAL_SUM']

    # 4C. Calculate BASE-100 INDEX
    # Converting to a Base-100 Index aligns the signal with S&P and GDP index formatting
    df_clean['EXPECTED_UNEMPLOYMENT_INDEX'] = 100 + df_clean['EXPECTED_UNEMPLOYMENT_RATE']

    # Export
    final_cols = ['t', 'endDate', 'EXPECTED_UNEMPLOYMENT_INDEX', 'EXPECTED_UNEMPLOYMENT_RATE', 'RAW_TOTAL_SUM'] + prob_cols
    df_clean = df_clean.sort_values(['endDate', 't'])
    df_clean[final_cols].to_csv(output_file, index=False)
    
    print(f"\n✅ SUCCESS! File saved: {output_file}")
    print("Side-by-side comparison (Raw EV vs Normalized % vs Base-100 Index):")
    print(df_clean[['t', 'RAW_EV', 'EXPECTED_UNEMPLOYMENT_RATE', 'EXPECTED_UNEMPLOYMENT_INDEX']].head())
# %%

# %%
# ==========================================
# PHASE 8D: TRANSPARENT YEARLY INFLATION PIPELINE
# ==========================================
"""
Phase 8D processes the Yearly Inflation data.
It maps target years, generates probability ladders, filters out incoherent 
pricing data (inverted ladders), and calculates a discrete Expected Value 
and Base-100 Index.
"""

print("--- STARTING INFLATION YEARLY PIPELINE: STEPS 1-4 ---")

history_file = 'TOTAL_HISTORY_INF_YEARLY.csv'
reference_file = 'FINAL_INFLATION_YEARLY.csv'
output_file = 'FINAL_INF_YEARLY_SIGNAL.csv'

if not os.path.exists(history_file) or not os.path.exists(reference_file):
    print("❌ Error: Files not found.")
else:
    # Load Data
    df_history = pd.read_csv(history_file)
    df_ref = pd.read_csv(reference_file)
    df_history['t'] = pd.to_datetime(df_history['t'], utc=True)

    # --- STEP 1: TIME FILTER & YEAR MAPPING ---
    print("▶ [Step 1/4] Snapping to hours and mapping target years...")
    df_history = df_history[df_history['t'].dt.minute < 5].copy()
    df_history['t_hour'] = df_history['t'].dt.floor('h')

    def extract_year(slug):
        m = re.search(r'202\d', str(slug))
        return m.group(0) if m else "UNKNOWN"

    mapping = df_ref[['slug', 'endDate']].drop_duplicates()
    df_mapped = df_history.merge(mapping, on='slug', how='left')
    df_mapped['target_year'] = df_mapped['slug'].apply(extract_year)

    # --- STEP 2: GRID GENERATION (THE LADDER) ---
    print("▶ [Step 2/4] Generating probability ladders per year...")
    all_year_grids = []
    unique_years = sorted(df_mapped['target_year'].unique())

    for year in unique_years:
        if year == "UNKNOWN": 
            continue
            
        year_data = df_mapped[df_mapped['target_year'] == year]
        
        # Pivot thresholds into columns
        pivot = year_data.pivot_table(index='t_hour', columns='groupItemTitle', values='p', aggfunc='last')
        
        # Continuous hourly timeline
        full_range = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h', tz='UTC')
        pivot = pivot.reindex(full_range).ffill()
        pivot['target_year'] = year
        all_year_grids.append(pivot.reset_index().rename(columns={'index': 't'}))

    full_grid = pd.concat(all_year_grids, ignore_index=True)

    # --- STEP 3: LADDER FILTER (CONSISTENCY CHECK) ---
    print("▶ [Step 3/4] Filtering incoherent data (Inverted Ladders)...")
    threshold_cols = sorted(
        [c for c in full_grid.columns if 'Above' in str(c)], 
        key=lambda x: float(re.findall(r'\d+\.?\d*', str(x))[0])
    )

    violations = pd.Series(False, index=full_grid.index)
    for i in range(len(threshold_cols) - 1):
        low_t = threshold_cols[i]
        high_t = threshold_cols[i + 1]
        
        # Violation if Higher Threshold is more expensive than Lower Threshold
        mask = full_grid[low_t].notna() & full_grid[high_t].notna()
        violations = violations | (mask & (full_grid[high_t] > full_grid[low_t]))

    df_clean = full_grid[~violations].copy()

    # --- STEP 4: DISCRETE EV & BASE-100 INDEX ---
    print("▶ [Step 4/4] Calculating Expected Value and Base-100 Index...")
    threshold_vals = [float(re.findall(r'\d+\.?\d*', c)[0]) for c in threshold_cols]

    def calculate_discrete_ev(row):
        ev = 0.0
        
        # 1. Below lowest threshold (e.g. 0% to 3%)
        p_bottom = 1.0 - row[threshold_cols[0]]
        w_bottom = threshold_vals[0] / 2
        
        if not pd.isna(p_bottom): 
            ev += p_bottom * w_bottom
        
        # 2. Middle brackets (e.g. 3% to 4%)
        for i in range(len(threshold_vals) - 1):
            p_bracket = row[threshold_cols[i]] - row[threshold_cols[i + 1]]
            w_bracket = (threshold_vals[i] + threshold_vals[i + 1]) / 2
            
            if not pd.isna(p_bracket): 
                ev += p_bracket * w_bracket
            
        # 3. Top bracket (e.g. >10%)
        p_top = row[threshold_cols[-1]]
        w_top = threshold_vals[-1] + 1.0 # 1% conservative buffer
        
        if not pd.isna(p_top): 
            ev += p_top * w_top
        
        return ev

    df_clean['EXPECTED_INFLATION_RATE'] = df_clean.apply(calculate_discrete_ev, axis=1)
    df_clean['EXPECTED_INFLATION_INDEX'] = 100 + df_clean['EXPECTED_INFLATION_RATE']

    # Export
    final_cols = ['t', 'target_year', 'EXPECTED_INFLATION_INDEX', 'EXPECTED_INFLATION_RATE'] + threshold_cols
    df_clean = df_clean.sort_values(['target_year', 't'])
    df_clean[final_cols].to_csv(output_file, index=False)
    
    # Audit Report
    print("\n=========================================================")
    print("  YEARLY INFLATION PIPELINE COMPLETE (APA Standard)")
    print("=========================================================")
    print(f"File Saved:           {output_file}")
    print(f"Hours Retained:       {len(df_clean)} of {len(full_grid)} ({(len(df_clean)/len(full_grid)*100):.1f}%)")
    print(f"Forecasted Years:     {df_clean['target_year'].unique().tolist()}")
    print("\nPreview of Index Signals:")
    print(df_clean[['t', 'target_year', 'EXPECTED_INFLATION_RATE', 'EXPECTED_INFLATION_INDEX']].head())
    print("=========================================================")
# %%

# %%
# ==========================================
# PHASE 8E: TRANSPARENT MONTHLY INFLATION PIPELINE
# ==========================================
"""
Phase 8E processes the Monthly Inflation markets using a Smart Parser. 
It deduces target years dynamically, categorizes bracket formats (cumulative 
vs. discrete), generates hourly probability matrices, and outputs a 
Base-100 Expected Monthly Inflation Index.
"""

print("--- STARTING MONTHLY INFLATION PIPELINE: STEPS 1-4 ---")

history_file = 'TOTAL_HISTORY_INF_MONTHLY.csv'
output_file = 'FINAL_INF_MONTHLY_SIGNAL.csv'

# Quality Gate: 0.90 - 1.10 (Academically defensible range for AMM liquidity)
LOWER_BOUND = 0.90
UPPER_BOUND = 1.10

if not os.path.exists(history_file):
    print(f"❌ Error: {history_file} not found.")
else:
    # 1. LOAD DATA
    df_raw = pd.read_csv(history_file)
    df_raw['t'] = pd.to_datetime(df_raw['t'], utc=True)

    # --- STEP 1: SMART MAPPING & YEAR DEDUCTION ---
    print("▶ [Step 1/4] Extracting metadata from slugs and deducing target years...")
    
    # Time Snapping: Keeping snapshots from the first 5 minutes of each hour
    df_step1 = df_raw[df_raw['t'].dt.minute < 5].copy()
    df_step1['t_hour'] = df_step1['t'].dt.floor('h')

    def parse_smart_slug(slug):
        months = [
            'january', 'february', 'march', 'april', 'may', 'june', 
            'july', 'august', 'september', 'october', 'november', 'december'
        ]
        
        slug_l = slug.lower()
        # Identify Target Month
        target_month = next((m for m in months if m in slug_l), 'unknown')
        
        # Identify Target Value (handles '2pt8' -> 2.8)
        value_match = re.search(r'(\d+)pt(\d+)', slug_l)
        target_val = float(f"{value_match.group(1)}.{value_match.group(2)}") if value_match else 0.0
        
        # Categorize Bracket Format (Discrete, Greater, or Less)
        is_greater = any(x in slug_l for x in ['or-more', '≥', 'greater'])
        is_less = any(x in slug_l for x in ['or-less', '≤', 'less'])
        
        return pd.Series([target_month, target_val, is_greater, is_less])

    df_step1[['target_month', 'target_val', 'is_greater', 'is_less']] = df_step1['slug'].apply(parse_smart_slug)

    # Year Deduction: If trading in late 2025 for 'January', target is Jan 2026
    def deduce_year(row):
        month_map = {
            m: i + 1 for i, m in enumerate([
                'january', 'february', 'march', 'april', 'may', 'june', 
                'july', 'august', 'september', 'october', 'november', 'december'
            ])
        }
        target_m_num = month_map.get(row['target_month'], 0)
        trade_year = row['t'].year
        trade_month = row['t'].month
        
        if target_m_num > 0 and trade_month > (target_m_num + 1):
            return trade_year + 1
            
        return trade_year

    df_step1['target_year'] = df_step1.apply(deduce_year, axis=1)
    df_step1['event_label'] = df_step1['target_month'].str.capitalize() + "-" + df_step1['target_year'].astype(str)

    # --- STEP 2: GRID GENERATION (THE MATRIX) ---
    print("▶ [Step 2/4] Generating hourly probability matrices...")
    all_grids = []
    unique_events = sorted(df_step1['event_label'].unique())

    for event in unique_events:
        event_data = df_step1[df_step1['event_label'] == event]
        
        # Pivot: Time as index, Titles as columns
        pivot = event_data.pivot_table(index='t_hour', columns='groupItemTitle', values='p', aggfunc='last')
        
        # Reindex to ensure continuous hourly history
        full_range = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h', tz='UTC')
        pivot = pivot.reindex(full_range).ffill()
        
        # Calculate RAW_TOTAL_SUM for the Quality Gate
        bracket_cols = [c for c in pivot.columns if c not in ['event_label', 't']]
        pivot['RAW_TOTAL_SUM'] = pivot[bracket_cols].sum(axis=1)
        pivot['event_label'] = event
        
        all_grids.append(pivot.reset_index().rename(columns={'index': 't'}))

    full_grid = pd.concat(all_grids, ignore_index=True)

    # --- STEP 3: QUALITY FILTERING ---
    print(f"▶ [Step 3/4] Filtering data with {LOWER_BOUND}-{UPPER_BOUND} gate...")
    
    df_clean = full_grid[
        (full_grid['RAW_TOTAL_SUM'] >= LOWER_BOUND) & 
        (full_grid['RAW_TOTAL_SUM'] <= UPPER_BOUND)
    ].copy()

    # --- STEP 4: EV, NORMALIZATION & INDEXING ---
    print("▶ [Step 4/4] Calculating Expected Value and Base-100 Index...")
    
    metadata = ['t', 'event_label', 'RAW_TOTAL_SUM']
    prob_cols = [c for c in df_clean.columns if c not in metadata]

    # Weights parser (extracts numerical level from title)
    def get_weight(title):
        nums = re.findall(r'(\d+\.\d+|\d+)', str(title))
        return float(nums[0]) if nums else 0.0

    weights = {col: get_weight(col) for col in prob_cols}

    # 4A. Calculate RAW_EV
    df_clean['RAW_EV'] = 0.0
    for col, weight in weights.items():
        df_clean['RAW_EV'] += df_clean[col].fillna(0) * weight
    
    # 4B. NORMALIZATION
    # Corrects for AMM spread drift (e.g., if sum is 1.017, we divide by 1.017)
    df_clean['EXPECTED_MONTHLY_INF_RATE'] = df_clean['RAW_EV'] / df_clean['RAW_TOTAL_SUM']

    # 4C. BASE-100 INDEX
    # Aligns signal with S&P/GDP indexing standards
    df_clean['EXPECTED_MONTHLY_INF_INDEX'] = 100 + df_clean['EXPECTED_MONTHLY_INF_RATE']

    # EXPORT FINAL SIGNAL
    final_cols = ['t', 'event_label', 'EXPECTED_MONTHLY_INF_INDEX', 'EXPECTED_MONTHLY_INF_RATE', 'RAW_TOTAL_SUM'] + prob_cols
    df_clean = df_clean.sort_values(['event_label', 't'])
    df_clean[final_cols].to_csv(output_file, index=False)

    # --- APA STANDARDIZED AUDIT REPORT ---
    print("\n" + "=" * 60)
    print("  MONTHLY INFLATION PIPELINE COMPLETE (APA Standard)")
    print("=" * 60)
    print(f"File Saved:           {output_file}")
    print(f"Initial Grid Rows:    {len(full_grid)}")
    print(f"Filtered Rows:        {len(df_clean)} ({(len(df_clean)/len(full_grid)*100):.1f}% retention)")
    print(f"Events Processed:     {len(unique_events)}")
    print("\nHead of Final Index:")
    print(df_clean[['t', 'event_label', 'EXPECTED_MONTHLY_INF_RATE', 'EXPECTED_MONTHLY_INF_INDEX']].head())
    print("=" * 60)
# %%    

# %%
# ==========================================
# PHASE 9: INTERNAL INTEGRITY & MASTER MERGE
# ==========================================
"""
Phase 9 acts as the final alignment engine. It takes the individual 
macro signals (Fed, GDP, Labor, Inflation), checks for internal timestamp 
overlaps, squashes any duplicates via averaging, and left-merges them 
onto a continuous, master hourly timeline starting from 2025.
"""

# Configuration: Mapping pillars to files and their specific feature columns
macro_config = {
    "FED": {
        "file": "FINAL_FED_SIGNAL.csv",
        "feature": "NORMALIZED_EV"
    },
    "GDP": {
        "file": "FINAL_GDP_SIGNAL.csv",
        "feature": "EXPECTED_GDP_INDEX"
    },
    "LABOR": {
        "file": "FINAL_LABOR_SIGNAL.csv",
        "feature": "EXPECTED_UNEMPLOYMENT_INDEX"
    },
    "INF_YEARLY": {
        "file": "FINAL_INF_YEARLY_SIGNAL.csv",
        "feature": "EXPECTED_INFLATION_INDEX"
    },
    "INF_MONTHLY": {
        "file": "FINAL_INF_MONTHLY_SIGNAL.csv",
        "feature": "EXPECTED_MONTHLY_INF_INDEX"
    }
}

def run_master_pipeline():
    processed_dfs = {}
    
    print("🔎 STEP 1: INTERNAL DATASET INTEGRITY CHECK")
    print("-" * 50)
    
    for key, config in macro_config.items():
        try:
            # Load the cleaned signal files
            df = pd.read_csv(config['file'])
            df['t'] = pd.to_datetime(df['t'], utc=True)
            
            # CHECK: Are there multiple rows for the same hour WITHIN this file?
            # (e.g. Fed file having September and December rows for the same timestamp)
            internal_overlaps = df.duplicated(subset=['t']).sum()
            
            if internal_overlaps > 0:
                print(f"⚠️  {key}: Found {internal_overlaps} internal timestamp overlaps.")
                
                # SQUASH LOGIC: Average the features to create one single value per hour
                # This prevents row-duplication during the final master merge.
                df_consolidated = df.groupby('t')[config['feature']].mean().reset_index()
                processed_dfs[key] = df_consolidated
            else:
                print(f"✅ {key}: No internal overlaps. Data is already 1-row-per-hour.")
                processed_dfs[key] = df[['t', config['feature']]]
                
        except FileNotFoundError:
            print(f"❌ {key}: File {config['file']} not found. Skipping...")

    # 2. GENERATE THE MASTER TIMELINE (Starting 2025-01-01)
    start_ts = pd.to_datetime('2025-01-01 00:00:00', utc=True)
    
    # Identify the maximum date available across all datasets
    all_max_dates = [df['t'].max() for df in processed_dfs.values() if not df.empty]
    end_ts = max(all_max_dates) if all_max_dates else start_ts
    
    print(f"\n📅 GENERATING MASTER TIMELINE: {start_ts} to {end_ts}")
    master_range = pd.date_range(start=start_ts, end=end_ts, freq='h', tz='UTC')
    master_df = pd.DataFrame({'t': master_range})

    # 3. CONSTRUCT THE 5-FEATURE TABLE
    print("🔗 PULLING VALUES INTO MASTER TABLE...")
    for key, df in processed_dfs.items():
        # Left merge ensures we keep the continuous timeline; missing hours stay NA
        master_df = pd.merge(master_df, df, on='t', how='left')

    # 4. FINAL EXPORT
    output_name = 'MASTER_MACRO_HOURLY_SIGNAL.csv'
    master_df.to_csv(output_name, index=False)
    
    print(f"\n🏁 EXPORT COMPLETE: {output_name}")
    print(f"Total Master Hours: {len(master_df)}")
    print("\n--- Final Column Counts (Non-NA) ---")
    print(master_df.count())

if __name__ == "__main__":
    run_master_pipeline()
# %%

# %%
# ==========================================
# PHASE 10: UNIFIED DUMMIES & DELTA TRANSFORMATION
# ==========================================
"""
Phase 10 applies the final transformations to the master dataset. 
It converts the absolute index levels into 1-hour deltas (differencing) 
and generates boundary dummy variables (1 or 0) to flag the opening 
and closing hours of specific macroeconomic markets.
"""

print("--- GENERATING DELTA-TRANSFORMED FEATURE DATASET ---")

# 1. Configuration: Mapping established names and event columns
pillar_map = {
    "FED": {
        "file": "FINAL_FED_SIGNAL.csv", 
        "event_col": "meeting_date", 
        "feature": "NORMALIZED_EV"
    },
    "GDP": {
        "file": "FINAL_GDP_SIGNAL.csv", 
        "event_col": "event", 
        "feature": "EXPECTED_GDP_INDEX"
    },
    "LABOR": {
        "file": "FINAL_LABOR_SIGNAL.csv", 
        "event_col": "endDate", 
        "feature": "EXPECTED_UNEMPLOYMENT_INDEX"
    },
    "INF_YEARLY": {
        "file": "FINAL_INF_YEARLY_SIGNAL.csv", 
        "event_col": "target_year", 
        "feature": "EXPECTED_INFLATION_INDEX"
    },
    "INF_MONTHLY": {
        "file": "FINAL_INF_MONTHLY_SIGNAL.csv", 
        "event_col": "event_label", 
        "feature": "EXPECTED_MONTHLY_INF_INDEX"
    }
}

input_file = 'MASTER_MACRO_HOURLY_SIGNAL.csv'
output_file = 'MASTER_FEATURE_DATASET.csv'

if not os.path.exists(input_file):
    print(f"❌ Error: {input_file} not found. Run Phase 9 first.")
else:
    # 2. Load the Backbone
    df = pd.read_csv(input_file)
    df['t'] = pd.to_datetime(df['t'], utc=True)

    print("🛠️  Generating Dummies and Transforming Indices to Deltas...")

    for name, cfg in pillar_map.items():
        feat = cfg['feature']
        
        # --- THE DELTA TRANSFORMATION ---
        # As per your snippet, we replace the absolute level with the 1-hour delta
        if feat in df.columns:
            df[feat] = df[feat].diff()
            print(f"✅ {feat}: Converted to Hourly Delta.")

        # --- THE BOUNDARY DUMMIES ---
        try:
            df_source = pd.read_csv(cfg['file'])
            df_source['t'] = pd.to_datetime(df_source['t'], utc=True)
            
            # Find the first and last hour for every unique market/contract
            bounds = df_source.groupby(cfg['event_col'])['t'].agg(['min', 'max']).reset_index()
            
            start_times = set(bounds['min'])
            end_times = set(bounds['max'])
            
            df[f"{name.lower()}_market_start"] = df['t'].isin(start_times).astype(int)
            df[f"{name.lower()}_market_end"] = df['t'].isin(end_times).astype(int)
            
            print(f"✅ {name}: Boundary Dummies added.")
            
        except Exception as e:
            print(f"⚠️  {name}: Dummies skipped ({e}).")
            df[f"{name.lower()}_market_start"] = 0
            df[f"{name.lower()}_market_end"] = 0

    # 3. FINAL EXPORT
    df.to_csv(output_file, index=False)
    
    print(f"\n🏁 SUCCESS: {output_file} created.")
    print("Note: The first row of deltas will be NaN.")
# %%

# %%
# ==========================================
# PHASE 11: ASSET DATA INCLUSION (YFINANCE)
# ==========================================
"""
Phase 11 pulls traditional financial market data (Equities, Commodities, 
Crypto, Yields) via Yahoo Finance. It safely chunks historical downloads, 
extracts 'Close' prices from complex MultiIndex structures, and converts 
absolute levels into hourly deltas (percentage changes), aligning everything 
to the strict UTC hourly timeline.
"""

OUTPUT_FILE = "financial_markets_hourly_utc.csv"
LEVELS_FILE = "financial_markets_hourly_levels_utc.csv"
METADATA_FILE = "financial_markets_metadata.json"

START_UTC = "2025-01-01 00:00:00"
UTC = "UTC"

MAIN_PRICE_TICKERS = {
    "SPY": "SPY_chg",
    "QQQ": "QQQ_chg",
    "GC=F": "Gold_chg",        # Gold futures
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

            time.sleep(random.uniform(2, 5))

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

    # Calculate basic time features
    changes_df["is_weekend"] = (changes_df.index.dayofweek >= 5).astype(int)
    changes_df["hour_utc"] = changes_df.index.hour
    changes_df["dow_utc"] = changes_df.index.dayofweek

    changes_df = changes_df.reset_index()
    levels_df = levels_df.reset_index()

    # Append to existing or create new
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

    # Export
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

# ==========================================
# EXECUTION & AUDIT REPORT
# ==========================================
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

print("\n--- Summary Statistics ---")
print(summary)
# %%

# %%
# ==========================================
# PHASE 12: MACRO ANNOUNCEMENT CALENDAR
# ==========================================
"""
Phase 12 builds a standalone calendar of scheduled macroeconomic data releases.
It takes hardcoded release dates, assigns them their proper New York release 
times (e.g., 8:30 AM for CPI, 2:00 PM for FOMC), and safely converts them to 
a UTC timeline, automatically adjusting for Daylight Saving Time shifts.
"""

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
    
    return df_master

# Run and save
df_timeline = create_master_timeline()
df_timeline.to_csv('Announcement_data.csv', index=False)

print("--- GENERATING MACRO ANNOUNCEMENT CALENDAR ---")
print("File saved as 'Announcement_data.csv'")

# ==========================================
# THE EVENT REGISTRY (Hardcoded Release Dates)
# ==========================================
# Format must be 'YYYY-MM-DD'
fomc_dates = [
    '2024-01-31', '2024-03-20', '2024-05-01', '2024-06-12', '2024-07-31', 
    '2024-09-18', '2024-11-07', '2024-12-18', '2025-01-29', '2025-03-19', 
    '2025-05-07', '2025-06-18', '2025-07-30', '2025-09-17', '2025-10-29', 
    '2025-12-10', '2026-01-28', '2026-03-18', '2026-04-29', '2026-06-17', 
    '2026-07-29', '2026-09-16', '2026-10-28', '2026-12-09'
]

cpi_dates = [
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10', '2024-05-15', 
    '2024-06-12', '2024-07-11', '2024-08-14', '2024-09-11', '2024-10-10', 
    '2024-11-13', '2024-12-11', '2025-01-15', '2025-02-12', '2025-03-12', 
    '2025-04-10', '2025-05-13', '2025-06-11', '2025-07-15', '2025-08-12', 
    '2025-09-11', '2025-10-24', '2025-12-18', '2026-01-13', '2026-02-13', 
    '2026-03-11', '2026-04-10', '2026-05-12', '2026-06-10', '2026-07-14', 
    '2026-08-12', '2026-09-11', '2026-10-14', '2026-11-10', '2026-12-10'
]

emp_dates = [
    '2024-01-05', '2024-02-02', '2024-03-08', '2024-04-05', '2024-05-03', 
    '2024-06-07', '2024-07-05', '2024-08-02', '2024-09-06', '2024-10-04', 
    '2024-11-01', '2024-12-06', '2025-01-10', '2025-02-07', '2025-03-07', 
    '2025-04-04', '2025-05-02', '2025-06-06', '2025-07-03', '2025-08-01', 
    '2025-09-05', '2025-11-20', '2025-12-16', '2026-01-09', '2026-02-11', 
    '2026-03-06', '2026-04-03', '2026-05-08', '2026-06-05', '2026-07-02', 
    '2026-08-07', '2026-09-04', '2026-10-02', '2026-11-06', '2026-12-04'
]

gdp_dates = [
    '2024-01-25', '2024-02-28', '2024-03-28', '2024-04-25', '2024-05-30', 
    '2024-06-27', '2024-07-25', '2024-08-29', '2024-09-26', '2024-10-30', 
    '2024-11-27', '2024-12-19', '2025-01-30', '2025-02-27', '2025-03-27', 
    '2025-04-30', '2025-05-29', '2025-06-26', '2025-07-30', '2025-08-28', 
    '2025-09-25', '2025-12-23', '2026-01-22', '2026-02-20', '2026-03-13', 
    '2026-04-09', '2026-04-30', '2026-05-28', '2026-06-25', '2026-07-30', 
    '2026-08-26', '2026-09-30', '2026-10-29', '2026-11-25', '2026-12-23'
]

# ==========================================
# THE RULE ENGINE (Timezone Translation)
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
print("✅ Phase 12 Complete. Here is the processed mini-dataset:\n")
print(df_processed_events.head())
# %%

# %%
# ==========================================
# PHASE 13: FINAL INTEGRATION & MASTER EXPORT
# ==========================================
"""
Phase 13 consolidates the final integration operations into a single workflow. 
It merges the Yahoo Finance assets, Macro announcements, and Polymarket signals. 
It then enforces strict data trimming to align all feature start dates, documents 
missing data via dummy variables, and drops redundant legacy columns to produce 
the ultimate model-ready dataset.
"""

print("--- STARTING FINAL INTEGRATION & ALIGNMENT ---")

# ==========================================
# STEP 1: MERGE ANNOUNCEMENTS & ASSETS
# ==========================================
# 1A. Merge the Master Timeline with Processed Events (From Phase 12)
df_merged = pd.merge(df_timeline, df_processed_events, on='Date', how='left')

# 1B. Pivot the 'event_type' into individual columns (Dummies)
df_final_ann = pd.get_dummies(df_merged, columns=['event_type'], prefix='Ann')

# 1C. Clean up column names and sum/max to keep exactly one row per hour
df_final_ann = df_final_ann.groupby('Date').max().reset_index()
df_final_ann = df_final_ann.fillna(0)

# Cast dummy columns to integer to avoid boolean/float errors
for col in df_final_ann.columns:
    if col.startswith('Ann_'):
        df_final_ann[col] = df_final_ann[col].astype(int)

df_final_ann.to_csv('Final_Announcement_Dummies.csv', index=False)
print("✅ STEP 1: 'Final_Announcement_Dummies.csv' created.")

# 1D. Merge Announcements with Yahoo Finance Assets (Safely loading Phase 11 output)
df_assets_yf = pd.read_csv("financial_markets_hourly_utc.csv")
df_assets_yf['Date'] = pd.to_datetime(df_assets_yf['Date'], utc=True)
df_final_ann['Date'] = pd.to_datetime(df_final_ann['Date'], utc=True)

df_master_merged = pd.merge(df_assets_yf, df_final_ann, how='left', on='Date')
df_master_merged.to_csv('MASTER_ASSET_AND_ANNOUNCEMENTS.csv', index=False)
print("✅ STEP 1: 'MASTER_ASSET_AND_ANNOUNCEMENTS.csv' created.")


# ==========================================
# STEP 2: MERGE MACRO FEATURES
# ==========================================
asset_file = 'MASTER_ASSET_AND_ANNOUNCEMENTS.csv'
macro_file = 'MASTER_FEATURE_DATASET.csv'

if not os.path.exists(asset_file) or not os.path.exists(macro_file):
    print("❌ ERROR: Files missing. Check your directory.")
else:
    df_assets = pd.read_csv(asset_file)
    df_assets['Date'] = pd.to_datetime(df_assets['Date'], utc=True)

    df_macro = pd.read_csv(macro_file)
    df_macro['t'] = pd.to_datetime(df_macro['t'], utc=True)
    df_macro.rename(columns={'t': 'Date'}, inplace=True)

    # Standardize column names
    rename_map = {
        'NORMALIZED_EV': 'FED_DELTA',
        'EXPECTED_GDP_INDEX': 'GDP_DELTA',
        'EXPECTED_UNEMPLOYMENT_INDEX': 'UNEMPLOYMENT_DELTA',
        'EXPECTED_INFLATION_INDEX': 'INF_YEARLY_DELTA',
        'EXPECTED_MONTHLY_INF_INDEX': 'INF_MONTHLY_DELTA'
    }
    df_macro.rename(columns=rename_map, inplace=True)

    # Merge Macro onto the Asset backbone (No Lag)
    df_integrated = pd.merge(df_assets, df_macro, on='Date', how='left')

    # Dummy Cleanup
    dummy_cols = [c for c in df_integrated.columns if any(x in c.lower() for x in ['start', 'end', 'mkt_'])]
    df_integrated[dummy_cols] = df_integrated[dummy_cols].fillna(0).astype(int)

    df_integrated.to_csv('Final_Cleaned_Dataset_2026.csv', index=False)
    print("✅ STEP 2: 'Final_Cleaned_Dataset_2026.csv' created.")


# ==========================================
# STEP 3: TRIM & HANDLE MISSING DATA
# ==========================================
df_clean = pd.read_csv('Final_Cleaned_Dataset_2026.csv')
df_clean['Date'] = pd.to_datetime(df_clean['Date'])
df_clean = df_clean.sort_values('Date').reset_index(drop=True)

poly_features = [
    'FED_DELTA', 'GDP_DELTA', 'UNEMPLOYMENT_DELTA', 
    'INF_YEARLY_DELTA', 'INF_MONTHLY_DELTA'
]

# Find the "latest" starting point where all 5 markets are live
first_valid_indices = [df_clean[feat].first_valid_index() for feat in poly_features]
global_start_index = max(first_valid_indices)

# Drop early rows
df_trimmed = df_clean.iloc[global_start_index:].reset_index(drop=True)

# Create missing-data dummies and fill NaNs
for feat in poly_features:
    dummy_name = f"{feat}_is_missing"
    df_trimmed[dummy_name] = df_trimmed[feat].isna().astype(int)
    df_trimmed[feat] = df_trimmed[feat].fillna(0)

df_trimmed.to_csv('FEATURES_PREPARED.csv', index=False)
print("✅ STEP 3: 'FEATURES_PREPARED.csv' trimmed and saved.")


# ==========================================
# STEP 4: FINAL FEATURE SELECTION & EXPORT
# ==========================================
df_final_pipe = pd.read_csv('FEATURES_PREPARED.csv')

# Drop legacy yearly inflation columns
cols_to_drop = [col for col in df_final_pipe.columns if 'inf_yearly' in col.lower()]
df_final_pipe = df_final_pipe.drop(columns=cols_to_drop)

output_final = 'Final_Pipeline_Data.csv'
df_final_pipe.to_csv(output_final, index=False)

print("\n" + "="*50)
print(" 🎉 PIPELINE COMPLETE: READY FOR MODELING 🎉")
print("="*50)
print(f"Final Output: {output_final}")
print(f"Total Rows:   {len(df_final_pipe)}")
print(f"Total Columns:{len(df_final_pipe.columns)}")
print("="*50)
# %%