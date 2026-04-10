# %%
# ==========================================
# ALL IMPORTS
# ==========================================
import numpy as np
import matplotlib.pyplot as plt  # Corrected from 'import matplotlib as plt'
import time
import requests as req
import json
import ast
import os
import pyarrow as pa
import pyarrow.parquet as pq
import random
import re
import pandas as pd
from collections import Counter, defaultdict
from urllib.parse import urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from py_clob_client import ClobClient
from py_clob_client import OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams, BalanceAllowanceParams, AssetType

# Newly added time tools for dynamic slug generation and timezone enforcement
from datetime import datetime
import pytz

# ==========================================
# API CONFIGURATION
# ==========================================
Gamma_api = "https://gamma-api.polymarket.com"
Data_api = "https://data-api.polymarket.com"
Clob_api = "https://clob.polymarket.com"

# Initialize the global session here so ALL phases can use it!
session = req.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"
})
# %%

# %%
# ==========================================
# FILE 1: MACRO EVENTS PIPELINE (DELTA LOAD)
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
# STEP 0: PRE-FLIGHT MEMORY CHECK (THE DELTA)
# ==========================================
seen_event_ids = set()
seen_tokens = set()

if os.path.exists(jsonl_filename):
    print(f"\nFound existing database: {jsonl_filename}. Loading memory to prevent duplicates...")
    with open(jsonl_filename, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                event_data = json.loads(line)
                
                # 1. Memorize Event ID
                e_id = event_data.get('id')
                if e_id: seen_event_ids.add(str(e_id))
                
                # 2. Memorize Token IDs (Brackets)
                for mkt in event_data.get('markets', []):
                    clob_raw = mkt.get('clobTokenIds', '[]')
                    # Handle Polymarket's stringified lists
                    if isinstance(clob_raw, str):
                        try: clob_ids = json.loads(clob_raw)
                        except: clob_ids = []
                    else:
                        clob_ids = clob_raw if isinstance(clob_raw, list) else []
                        
                    for tid in clob_ids:
                        seen_tokens.add(str(tid))
            except json.JSONDecodeError:
                continue
    print(f"Memory Loaded: {len(seen_event_ids)} Events | {len(seen_tokens)} Tokens.")
else:
    print(f"\nNo existing database found. Starting fresh.")

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
        tags_res = session.get(f"{Gamma_api}/tags", params={"limit": tag_limit, "offset": tag_offset}, timeout=15).json()
        if not tags_res or len(tags_res) == 0: break 
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
# STEP 2: EVENTS INGESTION LOOP (DELTA APPEND)
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
                params = {"limit": limit, "offset": offset, "tag_id": t_id, "active": status}
                
                try:
                    response = session.get(f"{Gamma_api}/events", params=params, timeout=15)
                    response.raise_for_status()
                    events = response.json()

                    if not events or len(events) == 0: break
                    
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
                                try: clob_ids = json.loads(clob_raw)
                                except: clob_ids = []
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

                    if len(events) < limit: break
                    offset += limit
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"An error occurred on tag {t_id}: {e}")
                    break

print(f"\nFinished! Appended {new_events_downloaded} NEW updates to {jsonl_filename}.")
# %%

# %%
# ==========================================
# FILE 2: EXTRACTION AND FILTERING (PHASE 2)
# ==========================================
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
    Returns a tuple: (macro_pillar, sub_category) or (None, None) if no match.
    """
    if not isinstance(slug, str):
        return None, None
        
    slug = slug.lower().strip()
    
    # --- 1. GDP GROWTH ---
    if slug.startswith("us-gdp-growth-in-q"):
        return "GDP_GROWTH", "GDP_QUARTERLY"
        
    # --- 2. LABOR MARKET (UNEMPLOYMENT) ---
    if "india" in slug or "indian" in slug:
        pass # Explicit exclusion
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

print(f"\n--- PART 2: THE STRICT GOLDEN CATALOG EXTRACTION ---")

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
            if not line.strip(): continue
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
MACRO_EVENTS_FILE = "PILLAR_MACRO_EVENTS.jsonl"
FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"

def flatten_macro_markets():
    if not os.path.exists(MACRO_EVENTS_FILE):
        print(f"[Error] Could not find {MACRO_EVENTS_FILE}")
        return

    print(f"\n--- PART 3: FLATTENING THE GOLDEN CATALOG (PHASE 3) ---")
    total_events_processed = 0
    total_markets_flattened = 0
    
    # We use a list to keep track of what we're about to write for the summary
    flat_records = []

    with open(MACRO_EVENTS_FILE, 'r', encoding='utf-8') as infile, \
         open(FLAT_MARKETS_FILE, 'w', encoding='utf-8') as flat_out:
        
        for line in infile:
            if not line.strip(): continue
            try:
                event = json.loads(line)
            except:
                continue
            
            total_events_processed += 1
            
            # Pull our custom labels from Phase 2
            event_id = event.get('id')
            macro_pillar = event.get('macro_pillar')
            sub_category = event.get('sub_category')
            event_title = event.get('title')
            
            markets = event.get('markets', [])
            
            if isinstance(markets, list):
                for mkt in markets:
                    # Extract the critical CLOB/Data IDs
                    market_id = mkt.get("id")
                    condition_id = mkt.get("conditionId")
                    group_item_title = mkt.get("groupItemTitle")
                    
                    if not market_id or not condition_id:
                        continue
                    
                    flat_record = {
                        "event_id": event_id,
                        "macro_pillar": macro_pillar,
                        "sub_category": sub_category,
                        "market_id": market_id,
                        "condition_id": condition_id,
                        "event_title": event_title,
                        "group_item_title": group_item_title,
                        "slug": mkt.get("slug")
                    }
                    
                    flat_out.write(json.dumps(flat_record) + '\n')
                    flat_records.append(flat_record)
                    total_markets_flattened += 1

    print(f"✅ Flattening Complete:")
    print(f"  -> Events Processed: {total_events_processed}")
    print(f"  -> Individual Markets Extracted: {total_markets_flattened}")
    print(f"  -> Output: {FLAT_MARKETS_FILE}")

flatten_macro_markets()
# %%

# %%
# ==========================================
# PHASE 4: SMART SYNC (JSONL EDITION)
# ==========================================
FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"
SYNCED_MARKET_FILE = "SYNCED_MARKET_DATA.jsonl"

def fetch_synced_macro_data():
    if not os.path.exists(FLAT_MARKETS_FILE):
        print(f"[Error] {FLAT_MARKETS_FILE} not found.")
        return
        
    print(f"\n--- PART 4: DELTA SYNC (ACTIVE + HISTORICAL) ---")

    # 1. Load what we already have into memory
    existing_data = {}
    if os.path.exists(SYNCED_MARKET_FILE):
        with open(SYNCED_MARKET_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
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
                    print(f"  Processed {i+1}/{len(target_records)}... (Updated: {new_api_calls} | Cached: {reused_historical})")
                
                time.sleep(0.2)
                
            except Exception as e:
                # If API fails, keep old data if we have it, otherwise skip
                if cached: outfile.write(json.dumps(cached) + '\n')
                print(f"  Error on Market {m_id}: {e}")

    print(f"\n✅ Sync Complete!")
    print(f"  -> Total Markets in Catalog: {new_api_calls + reused_historical}")
    print(f"  -> Fresh Updates Pulled: {new_api_calls}")
    print(f"  -> Historical Markets Preserved: {reused_historical}")

fetch_synced_macro_data()

# %%
# ==========================================
# PHASE 5: THE RESTORED IDENTITY REFINERY (2025+ Edition)
# ==========================================
SYNCED_FILE = "SYNCED_MARKET_DATA.jsonl"
DROPPED_FILE = "DROPPED_MARKETS.csv"

def run_final_refinery(min_volume=0): 
    print(f"\n--- PART 5: REFINERY (2025+ FOCUS / KILLING 2024 NOISE) ---")
    
    clean_records = []
    dropped_audit = []
    FOREIGN_EXCLUSIONS = ["japan", "canada", "india", "uk", "brazil", "mexico", "euro"]

    with open(SYNCED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            mkt = json.loads(line)
            slug = str(mkt.get("slug", "")).lower()
            
            # --- 1. THE 2024 EXORCISM ---
            # Direct removal of historical data to focus on the 2025-2026 cycle.
            if "2024" in slug:
                dropped_audit.append({"id": mkt.get("id"), "slug": slug, "reason": "Historical Noise (2024)"})
                continue

            # --- 2. GEOGRAPHIC BLOCK (Removes foreign noise) ---
            if any(country in slug for country in FOREIGN_EXCLUSIONS):
                dropped_audit.append({"id": mkt.get("id"), "slug": slug, "reason": "Foreign Geography"})
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
                if slug.startswith("will-the-") or slug.startswith("will-inflation-") or \
                   slug.startswith("will-annual-inflation-") or "how-high" in slug or \
                   "-us-annual" in slug or "-inflation-annual" in slug:
                    is_us_macro = True
                    feature = "INFLATION_YEAR_ANCHOR" if "how-high" in slug else "INFLATION_MONTH_YOY"

            # Fed Policy Patterns
            elif slug.startswith("fed-") or slug.startswith("no-change-") or \
                 slug.startswith("will-the-fed-") or slug.startswith("will-there-be-no-change-in-fed-"):
                is_us_macro = True
                feature = "FED_MEETING_DECISION"

            # FINAL ACTION
            if is_us_macro:
                mkt["specific_feature"] = feature
                mkt["total_volume_usd"] = float(mkt.get("volumeAmm") or 0) + float(mkt.get("volumeClob") or 0)
                clean_records.append(mkt)
            else:
                dropped_audit.append({"id": mkt.get("id"), "slug": slug, "reason": "Failed Pattern Filter"})

    # Export to CSV
    df = pd.DataFrame(clean_records)
    pd.DataFrame(dropped_audit).to_csv(DROPPED_FILE, index=False)

    pillar_map = {"GDP": "GDP_GROWTH", "INFLATION": "INFLATION", "LABOR": "LABOR_MARKET", "FED": "FED_POLICY"}
    
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
# PHASE 6: FINAL FEATURE LABELING & CLEANUP (THE TRUE FIX)
# ==========================================
csv_files = [
    "FINAL_GDP_MARKETS.csv", 
    "FINAL_INFLATION_MARKETS.csv", 
    "FINAL_LABOR_MARKETS.csv", 
    "FINAL_FED_MARKETS.csv"
]

print("--- RUNNING FINAL CLEANUP & SORTING ---")

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
# %%

# %%   
# --- FINAL FIX: SPLIT BY ORIGINAL SUB-CATEGORY ---
# We load the file we just saved and split it using the correct Phase 2 labels.
df_inf = pd.read_csv("FINAL_INFLATION_MARKETS.csv")

# Yearly Anchor = Slugs like 'will-inflation-reach-more-than-6-in-2025'
df_inf[df_inf['sub_category'] == 'INFLATION_YEAR_ANCHOR'].to_csv("FINAL_INFLATION_YEARLY.csv", index=False)

# Monthly YoY = Slugs like 'will-annual-inflation-increase-by-2pt6-in-january'
df_inf[df_inf['sub_category'] == 'INFLATION_MONTH_YOY'].to_csv("FINAL_INFLATION_MONTHLY_YOY.csv", index=False)

print(f"✅ Split Complete:")
print(f"  -> FINAL_INFLATION_YEARLY.csv: {len(df_inf[df_inf['sub_category'] == 'INFLATION_YEAR_ANCHOR'])} markets")
print(f"  -> FINAL_INFLATION_MONTHLY_YOY.csv: {len(df_inf[df_inf['sub_category'] == 'INFLATION_MONTH_YOY'])} markets")
# %%

# %%
# ==========================================
# PHASE 7: THE VACUUM (TOTAL HISTORY PULL) - DIAGNOSTIC EDITION
# ==========================================
import pandas as pd
import requests
import time
import os
import ast

# 1. Configuration & Time Boundaries
JAN_1_2025_TS = 1735689600  # Unix timestamp for Jan 1, 2025, 00:00:00 UTC
FIDELITY = 60               # Hourly data
THIRTY_DAYS = 30 * 24 * 60 * 60  # 30 days in seconds

# Testing with FED pillar first (You can add the others back later)
macro_pillars = {
    "FED": "FINAL_FED_MARKETS.csv"
}

# Helper function to safely parse stringified lists
def parse_array_string(val):
    try:
        if pd.isna(val): return []
        return ast.literal_eval(val)
    except Exception:
        return []

print("🚀 STARTING PRODUCTION DATA VACUUM (WITH DIAGNOSTICS)")
print("📅 Hard Boundary: January 1, 2025")

for category_name, source_csv in macro_pillars.items():
    if not os.path.exists(source_csv):
        print(f"⚠️ Missing source file: {source_csv}")
        continue
    
    df_ref = pd.read_csv(source_csv)
    output_filename = f"RAW_HOURLY_HISTORY_{category_name}.csv"
    
    existing_data = None
    known_max_timestamps = {}
    if os.path.exists(output_filename):
        existing_data = pd.read_csv(output_filename)
        known_max_timestamps = existing_data.groupby('token_id')['t'].max().to_dict()
        print(f"\n📂 CATEGORY: {category_name} (Found existing database. Updating...)")
    else:
        print(f"\n📂 CATEGORY: {category_name} (Creating new database...)")

    all_new_data = []

    for idx, row in df_ref.iterrows():
        tids = parse_array_string(row['clobTokenIds'])
        outcomes = parse_array_string(row.get('outcomes', '[]'))
        
        if not tids:
            continue
            
        print(f"\n  ({idx+1}/{len(df_ref)}) Market: {row['slug'][:50]}...")
        
        token_mapping = dict(zip(tids, outcomes + ['Unknown'] * (len(tids) - len(outcomes))))

        for tid, outcome_label in token_mapping.items():
            print(f"    -> Harvesting [{outcome_label}] (ID: {tid[:8]}...)")
            
            last_saved_ts = known_max_timestamps.get(tid, JAN_1_2025_TS)
            current_end_ts = int(time.time()) # Start from exactly right now
            market_lifetime_data = []
            
            # The Diagnostic Leapfrog Loop
            while current_end_ts > last_saved_ts:
                current_start_ts = current_end_ts - THIRTY_DAYS
                
                # Don't ask for data earlier than our Jan 1st boundary
                if current_start_ts < last_saved_ts:
                    current_start_ts = last_saved_ts

                # THE WINDOW FIX: Explicitly passing both startTs and endTs
                url = f"https://clob.polymarket.com/prices-history?market={tid}&fidelity={FIDELITY}&startTs={current_start_ts}&endTs={current_end_ts}"
                
                try:
                    resp = requests.get(url, timeout=15)
                    
                    if resp.status_code == 429:
                        print("      ⏳ Rate limit hit. Sleeping for 5 seconds...")
                        time.sleep(5)
                        continue 
                        
                    if resp.status_code != 200:
                        # --- ENHANCED ERROR LOGGING ---
                        print(f"      ⚠️ API REJECTION: HTTP {resp.status_code}")
                        print(f"         URL Used: {url}")
                        print(f"         Server Message: {resp.text[:250]}") # Shows exact API complaint
                        print(f"         Action: Leapfrogging backward to bypass error...")
                        current_end_ts = current_start_ts
                        time.sleep(2)
                        continue
                    
                    history = resp.json().get('history', [])
                    
                    # [FIDELITY FALLBACK] API quirk for closed markets
                    if not history:
                        fallback_url = url.replace(f"fidelity={FIDELITY}", "fidelity=720")
                        fallback_resp = requests.get(fallback_url, timeout=15)
                        if fallback_resp.status_code == 200:
                            history = fallback_resp.json().get('history', [])
                            if history:
                                print("      ⚠️ Switched to 12-hour fidelity (Closed market API quirk).")
                    
                    if history:
                        # Data found! Process it.
                        chunk_df = pd.DataFrame(history)
                        chunk_df['t'] = chunk_df['t'].astype(int)
                        
                        valid_chunk = chunk_df[chunk_df['t'] > last_saved_ts]
                        if not valid_chunk.empty:
                            market_lifetime_data.append(valid_chunk)
                            
                        # Set the next target to 1 second before the oldest data point we just got
                        earliest_in_batch = chunk_df['t'].min()
                        new_end = earliest_in_batch - 1
                        
                        # Failsafe: if the API returns weird timestamps, force it backward
                        if new_end >= current_end_ts:
                            new_end = current_start_ts
                            
                        current_end_ts = new_end
                    else:
                        # THE LEAPFROG: No data found? Jump back and keep looking.
                        start_str = pd.to_datetime(current_start_ts, unit='s').strftime('%Y-%m-%d')
                        end_str = pd.to_datetime(current_end_ts, unit='s').strftime('%Y-%m-%d')
                        print(f"      🦘 Liquidity gap ({start_str} to {end_str}). Leapfrogging backward...")
                        current_end_ts = current_start_ts
                        
                    time.sleep(0.15) 
                    
                except requests.exceptions.RequestException as e:
                    # --- ENHANCED NETWORK LOGGING ---
                    print(f"      ❌ Network/Timeout error: {e}")
                    print(f"         Action: Retrying previous window...")
                    time.sleep(5)
                    # We do not change the timestamps here so it retries the same window
            
            # Combine the chunks for this token
            if market_lifetime_data:
                full_token_df = pd.concat(market_lifetime_data, ignore_index=True)
                full_token_df['token_id'] = tid
                full_token_df['outcome_label'] = outcome_label 
                full_token_df['slug'] = row['slug']
                all_new_data.append(full_token_df)

    # Save and combine logic
    if all_new_data:
        new_df = pd.concat(all_new_data, ignore_index=True)
        
        if existing_data is not None:
            final_df = pd.concat([existing_data, new_df], ignore_index=True)
        else:
            final_df = new_df
            
        final_df = final_df.drop_duplicates(subset=['token_id', 't']).sort_values(['token_id', 't'])
        final_df.to_csv(output_filename, index=False)
        print(f"\n🏁 Category [{category_name}] Complete & Saved. Total rows: {len(final_df)}")
    else:
        print(f"\n🏁 Category [{category_name}] had no new data to save.")

print("\n✨ PIPELINE COMPLETE.")
# %%

# %%
# ==========================================
# PHASE 9: FED DATA COLLAPSE (V6 - STATE-DRIVEN FFILL)
# ==========================================
import pandas as pd
import re

# Load raw history (APA Reference: Polymarket, 2026)
input_file = 'TOTAL_HISTORY_FED.csv'
df = pd.read_csv(input_file)

# 1. TIME STANDARDIZATION
df['t'] = pd.to_datetime(df['t'], utc=True)
df['t_hour'] = df['t'].dt.floor('h')

def extract_bp_value(title):
    title = str(title).lower()
    if 'no change' in title: return 0
    match = re.search(r'(\d+)', title)
    if match:
        val = int(match.group(1))
        if 'decrease' in title: return -val
        elif 'increase' in title: return val
    return 0

def extract_meeting_id(slug):
    match = re.search(r'after-(?:the-)?(.*?)(?=-meeting|$)', slug)
    return match.group(1) if match else slug

df['bp_value'] = df['groupItemTitle'].apply(extract_bp_value)
df['meeting'] = df['slug'].apply(extract_meeting_id)

# 2. STAGE 1: DEDUPLICATE (Exact Hour Snapshots)
print("Stage 1: Removing duplicate snapshots within hours...")
df_hourly = df.groupby(['meeting', 't_hour', 'slug', 'bp_value'])['p'].mean().reset_index()

# Save a map of slug to bp_value so we can re-attach it after the pivot
slug_map = df_hourly[['slug', 'bp_value']].drop_duplicates()

# 3. STAGE 2: THE CONTINUOUS GRID (Safe Forward Fill)
print("Stage 2: Building continuous timelines and applying safe forward-fill...")
filled_timelines = []

# Process each FOMC meeting entirely independently
for meeting, group in df_hourly.groupby('meeting'):
    
    # Step A: Pivot to isolate slugs into independent columns
    pivot = group.pivot(index='t_hour', columns='slug', values='p')
    
    # Step B: Create a perfect hourly timeline from the Meeting's birth to last trade
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h')
    pivot = pivot.reindex(full_idx)
    
    # Step C: The Safe Forward-Fill (Leaves pre-birth NaNs alone)
    pivot = pivot.ffill()
    
    # Step D: Convert pre-birth NaNs to 0 probability
    pivot = pivot.fillna(0)
    
    # Step E: Melt it back into our vertical ML format
    melted = pivot.reset_index().melt(id_vars='index', var_name='slug', value_name='p')
    melted.rename(columns={'index': 't_hour'}, inplace=True)
    melted['meeting'] = meeting
    
    filled_timelines.append(melted)

# Reassemble the dataset
df_filled = pd.concat(filled_timelines, ignore_index=True)

# Re-attach the numerical basis points using our map
df_filled = df_filled.merge(slug_map, on='slug', how='left')

# 4. CALCULATE EXPECTED CONTRIBUTION
df_filled['p_times_bp'] = df_filled['p'] * df_filled['bp_value']

# 5. STAGE 3: COLLAPSE
print("Stage 3: Collapsing into total expected moves...")

# Only count slugs as "active" if they had a probability > 0
df_active = df_filled[df_filled['p'] > 0]

final_collapsed = df_active.groupby(['meeting', 't_hour']).agg({
    'p_times_bp': 'sum',      # The True Expected Move BPS
    'p': 'sum',               # Probability Integrity (Healed by the ffill)
    'slug': 'nunique'         # Number of Active Markets (prob > 0)
}).reset_index()

# Final column formatting
final_collapsed.rename(columns={
    't_hour': 't', 
    'p_times_bp': 'expected_move_bps',
    'p': 'sum_prob',
    'slug': 'outcome_count'
}, inplace=True)

# NO FILTERS - Save the raw truth
output_name = 'FED_MEETINGS_FINAL_SIGNAL.csv'
final_collapsed.to_csv(output_name, index=False)
print(f"🏁 DONE: Saved {len(final_collapsed)} unbroken, state-driven hours to {output_name}")
# %%

# %%
# ==========================================
# PHASE 10: GDP DATA COLLAPSE (V6 - STATE-DRIVEN FFILL)
# ==========================================
import pandas as pd
import re

# Load raw history (APA Reference: Polymarket, 2026)
input_file = 'TOTAL_HISTORY_GDP.csv'
df = pd.read_csv(input_file)

# 1. TIME STANDARDIZATION
df['t'] = pd.to_datetime(df['t'], utc=True)
df['t_hour'] = df['t'].dt.floor('h')

def extract_gdp_value(slug):
    """Converts outcome slug to numerical GDP percentages."""
    s = slug.replace('pt', '.')
    match_between = re.search(r'between-([\d\.]+)-and-([\d\.]+)', s)
    if match_between: return (float(match_between.group(1)) + float(match_between.group(2))) / 2.0
    match_less = re.search(r'less-than-([\d\.]+)', s)
    if match_less: return float(match_less.group(1))
    match_greater = re.search(r'greater-than-([\d\.]+)', s)
    if match_greater: return float(match_greater.group(1))
    return 0.0

def extract_gdp_quarter(slug):
    """Isolates the GDP quarter from the outcome slug."""
    match = re.search(r'(q[1-4]-202[5-6])', slug)
    return match.group(1).upper() if match else slug

df['gdp_value'] = df['slug'].apply(extract_gdp_value)
df['quarter'] = df['slug'].apply(extract_gdp_quarter)

# 2. STAGE 1: DEDUPLICATE (Bypass Broken Slugs)
print("Stage 1: Removing duplicate snapshots within hours...")
# WE CRITICALLY GROUP BY gdp_value, NOT slug, TO FIX THE GDP ISSUE
df_hourly = df.groupby(['quarter', 't_hour', 'gdp_value'])['p'].mean().reset_index()

# 3. STAGE 2: THE CONTINUOUS GRID (Safe Forward Fill)
print("Stage 2: Building continuous timelines and applying safe forward-fill...")
filled_timelines = []

for quarter, group in df_hourly.groupby('quarter'):
    # Pivot using the clean math (gdp_value) as the columns
    pivot = group.pivot(index='t_hour', columns='gdp_value', values='p')
    
    # Create the perfect hourly ruler for this quarter's lifetime
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h')
    pivot = pivot.reindex(full_idx)
    
    # Safely carry actual trades forward (leaves unborn markets as NaN)
    pivot = pivot.ffill()
    
    # Convert unborn markets to 0 probability
    pivot = pivot.fillna(0)
    
    # Melt back into the vertical format
    melted = pivot.reset_index().melt(id_vars='index', var_name='gdp_value', value_name='p')
    melted.rename(columns={'index': 't_hour'}, inplace=True)
    melted['quarter'] = quarter
    
    filled_timelines.append(melted)

df_filled = pd.concat(filled_timelines, ignore_index=True)

# 4. CALCULATE EXPECTED CONTRIBUTION
df_filled['p_times_gdp'] = df_filled['p'] * df_filled['gdp_value']

# 5. STAGE 3: COLLAPSE
print("Stage 3: Collapsing into total expected moves...")

# Only count brackets that actually have probability weight
df_active = df_filled[df_filled['p'] > 0]

final_collapsed = df_active.groupby(['quarter', 't_hour']).agg({
    'p_times_gdp': 'sum',      # The True Expected GDP Growth %
    'p': 'sum',                # Probability Integrity (includes AMM spikes)
    'gdp_value': 'nunique'     # Number of Active Brackets
}).reset_index()

# Final formatting
final_collapsed.rename(columns={
    't_hour': 't', 
    'p_times_gdp': 'expected_gdp_growth', 
    'p': 'sum_prob', 
    'gdp_value': 'outcome_count'
}, inplace=True)

# Save the raw, unmanipulated ML feature
output_name = 'GDP_EXPECTED_GROWTH_FINAL_SIGNAL.csv'
final_collapsed.to_csv(output_name, index=False)
print(f"🏁 DONE: Saved {len(final_collapsed)} unbroken, state-driven hours to {output_name}")
# %%

# %%
# ==========================================
# PHASE 11: LABOR DATA COLLAPSE (V6 PORT)
# ==========================================
import pandas as pd
import re

# Load raw history (APA Reference: Polymarket, 2026)
input_file = 'TOTAL_HISTORY_LABOR.csv'
df = pd.read_csv(input_file)

# 1. TIME STANDARDIZATION: Floor everything to the hour
df['t'] = pd.to_datetime(df['t'], utc=True)
df['t_hour'] = df['t'].dt.floor('h')

def extract_labor_value(title):
    """
    Extracts labor value based on the specific rule:
    Find the '%' sign and take the 3 characters to its left.
    """
    title = str(title)
    if '%' in title:
        idx = title.find('%')
        # Grab the 3 characters before the %
        val_str = title[max(0, idx-3):idx]
        # Remove any non-numeric noise (like the ≥ or ≤ symbols)
        val_str = re.sub(r'[^0-9.]', '', val_str)
        try:
            return float(val_str)
        except:
            return 0.0
    return 0.0

def extract_labor_event(slug):
    """Isolates the report month and year from the slug."""
    match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)-?(202[4-6])?', slug, re.I)
    if match:
        month = match.group(1).lower()
        year = match.group(2)
        if not year: year = "2025" # Default for missing year tags in slugs
        return f"{month}-{year}"
    return slug

df['labor_val'] = df['groupItemTitle'].apply(extract_labor_value)
df['event'] = df['slug'].apply(extract_labor_event)

# 2. STAGE 1: DEDUPLICATE outcomes within the same hour
print("Stage 1: Averaging duplicate snapshots within hours...")
df_hourly = df.groupby(['event', 't_hour', 'labor_val'])['p'].mean().reset_index()

# 3. STAGE 2: THE CONTINUOUS GRID (Safe Forward-Fill)
print("Stage 2: Building continuous timelines and applying forward-fill...")
filled_timelines = []

for event, group in df_hourly.groupby('event'):
    # Pivot so each numerical bracket is its own column
    pivot = group.pivot(index='t_hour', columns='labor_val', values='p')
    
    # Create the hourly ruler from the first to last trade of this report
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h')
    pivot = pivot.reindex(full_idx)
    
    # Carry last traded price forward into silent hours
    pivot = pivot.ffill()
    
    # Pre-launch hours remain 0
    pivot = pivot.fillna(0)
    
    # Melt back into the ML structure
    melted = pivot.reset_index().melt(id_vars='index', var_name='labor_val', value_name='p')
    melted.rename(columns={'index': 't_hour'}, inplace=True)
    melted['event'] = event
    filled_timelines.append(melted)

df_filled = pd.concat(filled_timelines, ignore_index=True)

# 4. CALCULATE CONTRIBUTION
df_filled['p_times_val'] = df_filled['p'] * df_filled['labor_val']

# 5. STAGE 3: COLLAPSE into final Expected Unemployment Rate
print("Stage 3: Collapsing into final signal...")
df_active = df_filled[df_filled['p'] > 0]

final_collapsed = df_active.groupby(['event', 't_hour']).agg({
    'p_times_val': 'sum',      # THE EXPECTED UNEMPLOYMENT RATE
    'p': 'sum',                # Probability Integrity (Target: 1.0)
    'labor_val': 'nunique'     # Number of brackets active
}).reset_index()

# Final column formatting
final_collapsed.rename(columns={
    't_hour': 't', 
    'p_times_val': 'expected_unemployment_rate', 
    'p': 'sum_prob', 
    'labor_val': 'outcome_count'
}, inplace=True)

# Save the mock data for the team
output_name = 'LABOR_EXPECTED_UNEMPLOYMENT_FINAL_SIGNAL.csv'
final_collapsed.to_csv(output_name, index=False)
print(f"🏁 DONE: Saved {len(final_collapsed)} state-driven hours to {output_name}") 
# %%

# %%
# ==========================================
# PHASE 12: INFLATION YEARLY (V6 PORT - CUMULATIVE FIX)
# ==========================================
import pandas as pd
import re
import numpy as np

# Load raw history (APA Reference: Polymarket, 2026)
input_file = 'TOTAL_HISTORY_INF_YEARLY.csv'
df = pd.read_csv(input_file)

# 1. TIME STANDARDIZATION
df['t'] = pd.to_datetime(df['t'], utc=True)
df['t_hour'] = df['t'].dt.floor('h')

def extract_inf_value(title):
    """Rule: Find % and take 3 characters to the left."""
    title = str(title)
    if '%' in title:
        idx = title.find('%')
        val_str = title[max(0, idx-3):idx]
        val_str = re.sub(r'[^0-9.]', '', val_str)
        try: return float(val_str)
        except: return 0.0
    return 0.0

def extract_year(slug):
    """Isolates the target year (2025/2026) from the slug."""
    match = re.search(r'(202[4-6])', slug)
    return match.group(1) if match else "2025"

df['inf_val'] = df['groupItemTitle'].apply(extract_inf_value)
df['year'] = df['slug'].apply(extract_year)

# 2. STAGE 1: DEDUPLICATE (Exact Hour Snapshots)
print("Stage 1: Averaging duplicate snapshots within hours...")
df_hourly = df.groupby(['year', 't_hour', 'inf_val'])['p'].mean().reset_index()

# 3. STAGE 2: THE CONTINUOUS GRID (Safe Forward-Fill)
print("Stage 2: Building continuous timelines and applying forward-fill...")
filled_timelines = []

for year, group in df_hourly.groupby('year'):
    # Pivot so each cumulative threshold (e.g., 3.0, 4.0) is its own column
    pivot = group.pivot(index='t_hour', columns='inf_val', values='p')
    
    # Create the perfect hourly timeline ruler
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h')
    pivot = pivot.reindex(full_idx)
    
    # Safe Forward-Fill existing markets
    pivot = pivot.ffill()
    
    # Pre-launch hours remain 0 (or NaNs)
    pivot = pivot.fillna(0)
    
    # Melt back into the vertical format
    melted = pivot.reset_index().melt(id_vars='index', var_name='inf_val', value_name='p')
    melted.rename(columns={'index': 't_hour'}, inplace=True)
    melted['year'] = year
    filled_timelines.append(melted)

df_filled = pd.concat(filled_timelines, ignore_index=True)

# 4. STAGE 3: THE "CORRESPONDING" CUMULATIVE COLLAPSE
print("Stage 3: Converting cumulative probabilities into expected inflation rate...")

def calculate_expected_inflation(group):
    # Sort by value to get thresholds in order (e.g. 3, 4, 5, 6...)
    group = group.sort_values('inf_val')
    vals = group['inf_val'].values
    probs = group['p'].values # These are P(>X)
    
    # We define buckets: [Below min], [Between i and i+1], [Above max]
    d_probs = []    # Probability Density
    midpoints = []  # Bucket Midpoints
    
    # Bucket 0: Below lowest threshold
    d_probs.append(1.0 - probs[0])
    midpoints.append(max(0, vals[0] - 0.5))
    
    # Buckets i: Between thresholds
    for i in range(len(vals) - 1):
        # Probability of being between threshold A and B
        density = probs[i] - probs[i+1]
        d_probs.append(max(0, density)) # Handle minor market noise
        midpoints.append((vals[i] + vals[i+1]) / 2.0)
    
    # Bucket Last: Above highest threshold
    d_probs.append(probs[-1])
    midpoints.append(vals[-1] + 0.5)
    
    # Normalize density to handle raw market noise
    total_d = sum(d_probs)
    if total_d > 0:
        d_probs = [d / total_d for d in d_probs]
    
    # Calculate Expected Value
    ev = sum(p * m for p, m in zip(d_probs, midpoints))
    
    return pd.Series({
        'expected_inflation_rate': ev,
        'sum_prob_raw': np.sum(probs),
        'outcome_count': len(vals)
    })

final_collapsed = df_filled.groupby(['year', 't_hour']).apply(calculate_expected_inflation).reset_index()

# Final column formatting
final_collapsed.rename(columns={'t_hour': 't'}, inplace=True)

# Save the final signal
output_name = 'INF_YEARLY_FINAL_SIGNAL.csv'
final_collapsed.to_csv(output_name, index=False)
print(f"🏁 DONE: Saved {len(final_collapsed)} state-driven hours to {output_name}")
# %%

# %%
# ==========================================
# PHASE 13: MONTHLY INFLATION (V7 - ROBUST)
# ==========================================
import pandas as pd
import re

# Load raw history (APA Reference: Polymarket, 2026)
input_file = 'TOTAL_HISTORY_INF_MONTHLY.csv'
df = pd.read_csv(input_file)

# 1. TIME STANDARDIZATION: Floor everything to the hour
df['t'] = pd.to_datetime(df['t'], utc=True)
df['t_hour'] = df['t'].dt.floor('h')

def extract_inf_val(title):
    """
    Revised Extraction: Finds digits and dots before the % sign.
    Ensures '2.8' and '2.80' are treated as the same mathematical value.
    """
    title = str(title)
    if '%' in title:
        # Extract the part before the %
        match = re.search(r'([\d\.]+)%', title)
        if match:
            val_str = match.group(1)
            try:
                return float(val_str)
            except:
                return 0.0
    return 0.0

def get_month_event(slug):
    """Isolates the CPI month from the slug."""
    match = re.search(r'(january|february|march|april|may|june|july|august|september|october|november|december)', slug, re.I)
    return match.group(1).lower() if match else slug

df['inf_val'] = df['groupItemTitle'].apply(extract_inf_val)
df['event'] = df['slug'].apply(get_month_event)

# Capture slug mapping to re-attach math values after the timeline fill
slug_map = df[['slug', 'inf_val', 'event']].drop_duplicates()

# 2. STAGE 1: DEDUPLICATE (Exact Hour Snapshots)
print("Stage 1: Removing duplicate snapshots within hours...")
df_hourly = df.groupby(['slug', 't_hour'])['p'].mean().reset_index()

# 3. STAGE 2: THE CONTINUOUS GRID (Safe Forward-Fill)
print("Stage 2: Building continuous timelines and applying forward-fill...")
filled_timelines = []

# Process each monthly report independently
for event, group_event in df_hourly.merge(slug_map[['slug', 'event']], on='slug').groupby('event'):
    
    # Pivot so each outcome is an independent column
    pivot = group_event.pivot(index='t_hour', columns='slug', values='p')
    
    # Create the perfect hourly ruler for this event
    full_idx = pd.date_range(start=pivot.index.min(), end=pivot.index.max(), freq='h', tz='UTC')
    pivot = pivot.reindex(full_idx)
    
    # Safe Forward-Fill (Carries last known price, ignores pre-birth hours)
    pivot = pivot.ffill()
    pivot = pivot.fillna(0)
    
    # Melt it back safely (Fixing the KeyError by explicitly naming columns)
    pivot.index.name = 't_hour'
    melted = pivot.reset_index().melt(id_vars='t_hour', var_name='slug', value_name='p')
    melted['event'] = event
    
    filled_timelines.append(melted)

df_filled = pd.concat(filled_timelines, ignore_index=True)

# 4. RE-ATTACH NUMERICAL VALUES
df_filled = df_filled.merge(slug_map[['slug', 'inf_val']], on='slug', how='left')

# 5. STAGE 3: COLLAPSE (Weighted Average Expectation)
print("Stage 3: Collapsing into final expected monthly inflation rates...")
df_filled['p_times_val'] = df_filled['p'] * df_filled['inf_val']

# Only count outcomes with real market weight
df_active = df_filled[df_filled['p'] > 0]

final_collapsed = df_active.groupby(['event', 't_hour']).agg({
    'p_times_val': 'sum',      # The Raw Expected Move
    'p': 'sum',                # Probability Integrity (Target: 1.0)
    'slug': 'nunique'          # Number of active outcomes
}).reset_index()

# Final column formatting
final_collapsed.rename(columns={
    't_hour': 't', 
    'p_times_val': 'expected_monthly_inflation', 
    'p': 'sum_prob', 
    'slug': 'outcome_count'
}, inplace=True)

# Save the master signal file
output_name = 'INF_MONTHLY_YOY_FINAL_SIGNAL.csv'
final_collapsed.to_csv(output_name, index=False)
print(f"🏁 DONE: Saved {len(final_collapsed)} state-driven hours to {output_name}")
# %%

# %%
# ==========================================
# PHASE 14: INTERNAL INTEGRITY & MASTER MERGE
# ==========================================
import pandas as pd

# Configuration: Mapping pillars to files and their specific feature columns
macro_config = {
    "FED": {
        "file": "FED_MEETINGS_FINAL_SIGNAL.csv",
        "feature": "expected_move_bps"
    },
    "GDP": {
        "file": "GDP_EXPECTED_GROWTH_FINAL_SIGNAL.csv",
        "feature": "expected_gdp_growth"
    },
    "LABOR": {
        "file": "LABOR_EXPECTED_UNEMPLOYMENT_FINAL_SIGNAL.csv",
        "feature": "expected_unemployment_rate"
    },
    "INF_YEARLY": {
        "file": "INF_YEARLY_FINAL_SIGNAL.csv",
        "feature": "expected_inflation_rate"
    },
    "INF_MONTHLY": {
        "file": "INF_MONTHLY_YOY_FINAL_SIGNAL.csv",
        "feature": "expected_monthly_inflation"
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
# PHASE 15: MASTER TABLE WITH 10 BOUNDARY DUMMIES
# ==========================================
import pandas as pd

# Define our 5 processed signal files and their identifying event column
pillar_map = {
    "FED": {"file": "FED_MEETINGS_FINAL_SIGNAL.csv", "event_col": "meeting", "feature": "expected_move_bps"},
    "GDP": {"file": "GDP_EXPECTED_GROWTH_FINAL_SIGNAL.csv", "event_col": "quarter", "feature": "expected_gdp_growth"},
    "LABOR": {"file": "LABOR_EXPECTED_UNEMPLOYMENT_FINAL_SIGNAL.csv", "event_col": "event", "feature": "expected_unemployment_rate"},
    "INF_YEARLY": {"file": "INF_YEARLY_FINAL_SIGNAL.csv", "event_col": "year", "feature": "expected_inflation_rate"},
    "INF_MONTHLY": {"file": "INF_MONTHLY_YOY_FINAL_SIGNAL.csv", "event_col": "event", "feature": "expected_monthly_inflation"}
}

def create_master_with_dummies():
    # 1. Load the Master Hourly Backbone we created in Phase 14
    # (Starting 2025-01-01)
    master_df = pd.read_csv('MASTER_MACRO_HOURLY_SIGNAL.csv')
    master_df['t'] = pd.to_datetime(master_df['t'], utc=True)
    
    print("🛠️  Generating 10 Dummies (Start/End flags for all pillars)...")
    
    for name, cfg in pillar_map.items():
        try:
            # Load the original signal file to find the true birth/death of each market
            df = pd.read_csv(cfg['file'])
            df['t'] = pd.to_datetime(df['t'], utc=True)
            
            # Find the FIRST hour and LAST hour for every single market in this pillar
            # (e.g. Find when the 'September Fed Meeting' started and ended)
            event_bounds = df.groupby(cfg['event_col'])['t'].agg(['min', 'max']).reset_index()
            
            start_times = set(event_bounds['min'])
            end_times = set(event_bounds['max'])
            
            # Create the 2 dummy columns for this pillar
            start_col = f"{name.lower()}_market_start"
            end_col = f"{name.lower()}_market_end"
            
            # Map the dummies: 1 if this timestamp is a start/end, 0 otherwise
            master_df[start_col] = master_df['t'].isin(start_times).astype(int)
            master_df[end_col] = master_df['t'].isin(end_times).astype(int)
            
            print(f"✅ Created {start_col} and {end_col}")
            
        except FileNotFoundError:
            print(f"⚠️  Skipping {name}: {cfg['file']} not found.")
            # Create columns as 0s so the team's code doesn't break
            master_df[f"{name.lower()}_market_start"] = 0
            master_df[f"{name.lower()}_market_end"] = 0

    # 2. THE DELTA CHECK (Capturing "Change")
    # As we discussed, dummies fix the noise, but 'Change' captures the momentum.
    # Let's add the 1-hour change for the Fed as an example:
    master_df['fed_move_delta_1h'] = master_df['expected_move_bps'].diff()

    # 3. EXPORT FINAL MASTER FEATURE DATASET
    output_name = 'MASTER_FEATURE_DATASET.csv'
    master_df.to_csv(output_name, index=False)
    
    print(f"\n🏁 MASTER DATASET READY: {output_name}")
    print(f"Total Rows: {len(master_df)}")
    print(f"Total Columns: {len(master_df.columns)}")

if __name__ == "__main__":
    create_master_with_dummies()
# %%

# %%
import pandas as pd

# Load the dataset
df = pd.read_csv('MASTER_FEATURE_DATASET.csv')

# Identify and delete any columns containing "delta" in the header
cols_to_drop = [c for c in df.columns if 'delta' in c.lower()]
df.drop(columns=cols_to_drop, inplace=True)

# Save the updated file back to the same filename
df.to_csv('MASTER_FEATURE_DATASET.csv', index=False)

print(f"Successfully updated MASTER_FEATURE_DATASET.csv. Removed columns: {cols_to_drop}")
# %%

# %%
import pandas as pd

# Load the master feature dataset
file_name = 'MASTER_FEATURE_DATASET.csv'
df = pd.read_csv(file_name)

# Define the five macro features to be transformed
features = [
    'expected_move_bps', 
    'expected_gdp_growth', 
    'expected_unemployment_rate', 
    'expected_inflation_rate', 
    'expected_monthly_inflation'
]

# Transform each column into a Delta (Current Value - Previous Hour Value)
for col in features:
    if col in df.columns:
        # We replace the absolute level with the 1-hour delta
        df[col] = df[col].diff()

# Save the updated dataset
df.to_csv(file_name, index=False)

print(f"✅ Successfully converted the 5 macro features to Deltas in {file_name}")
# Note: The first row (2025-01-01 00:00) will now be NaN for these columns 
# because there is no previous hour to calculate a delta from.
# %%

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
# ======================================================
# THE FINAL INTEGRATION: ASSETS + ANNOUNCEMENTS + MACRO
# ======================================================
import pandas as pd

# 1. LOAD THE DATASETS
# df_assets: Your prior 'MASTER_ASSET_AND_ANNOUNCEMENTS.csv'
# df_macro: Your 'MASTER_FEATURE_DATASET.csv' (Deltas + Dummies)

df_assets = pd.read_csv('MASTER_ASSET_AND_ANNOUNCEMENTS.csv')
df_assets['Date'] = pd.to_datetime(df_assets['Date'], utc=True)

df_macro = pd.read_csv('MASTER_FEATURE_DATASET.csv')
df_macro['t'] = pd.to_datetime(df_macro['t'], utc=True)
df_macro.rename(columns={'t': 'Date'}, inplace=True)

# 2. THE LAG (THE "ANTI-CHEAT" CHECK)
# We shift only the macro columns. 
# This aligns the Macro-Change from 13:00 with the Price-Target of 14:00.
macro_cols = [c for c in df_macro.columns if c != 'Date']
df_macro[macro_cols] = df_macro[macro_cols].shift(1)

# 3. THE ULTIMATE MERGE
# We use 'left' to ensure we never lose asset price rows
df_final_cleaned = pd.merge(df_assets, df_macro, on='Date', how='left')

# 4. STRUCTURAL CLEANUP
# Fill the Macro Dummies with 0 (if no market exists, it's not starting/ending)
dummy_cols = [c for c in df_final_cleaned.columns if 'mkt_' in c or 'start' in c or 'end' in c]
df_final_cleaned[dummy_cols] = df_final_cleaned[dummy_cols].fillna(0).astype(int)

# 5. EXPORT
output_name = 'Final_Cleaned_dataset.csv'
df_final_cleaned.to_csv(output_name, index=False)

print(f"🏁 DONE: '{output_name}' is ready.")
print("Note: Macro features are lagged by 1h to simulate real-time availability.")
# %%