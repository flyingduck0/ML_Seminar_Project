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
from datetime import dat
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
    if slug.startswith("gdp-growth-in-20"):
        return "GDP_GROWTH", "GDP_YEARLY"
        
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
            if slug.startswith("us-gdp-") or slug.startswith("gdp-growth-in-20") or slug.startswith("will-us-gdp-"):
                is_us_macro = True
                feature = "GDP_QUARTERLY" if "in-q" in slug else "GDP_YEARLY"
            
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
# PHASE 6: FINAL FEATURE LABELING (REAL-WORLD FIX)
# ==========================================
MONTHS = ["january", "february", "march", "april", "may", "june", 
          "july", "august", "september", "october", "november", "december"]

def get_exact_feature(slug):
    slug = str(slug).lower()
    
    # --- 1. GDP GROWTH ---
    # Broadened to catch slugs starting with "will-"
    if "gdp-growth" in slug and any(q in slug for q in ["q1", "q2", "q3", "q4"]):
        return "GDP_QUARTERLY"
    if "gdp-growth" in slug and ("2025" in slug or "2026" in slug):
        return "GDP_YEARLY"
        
    # --- 2. LABOR MARKET (UNEMPLOYMENT) ---
    if "india" in slug or "indian" in slug:
        return "OTHER"
    elif "unemployment-rate" in slug:
        if any(m in slug for m in MONTHS):
            return "UNEMPLOYMENT_MONTHLY"
            
    # --- 3. INFLATION ---
    if "how-high" in slug and "inflation" in slug:
        return "INFLATION_YEAR_ANCHOR"
    if "inflation-annual" in slug or "inflation-us-annual" in slug or "annual-inflation" in slug:
        return "INFLATION_MONTH_YOY"
        
    # --- 4. FED POLICY ---
    # Broadened to catch 'fed-decreases', 'fed-increases', and 'no-change'
    if "fed-" in slug or "fed-decision" in slug or "no-change" in slug:
        if any(m in slug for m in MONTHS):
            return "FED_MEETING_DECISION"
            
    return "OTHER"

# Apply to your 4 files
csv_files = ["FINAL_GDP_MARKETS.csv", "FINAL_INFLATION_MARKETS.csv", 
             "FINAL_LABOR_MARKETS.csv", "FINAL_FED_MARKETS.csv"]

for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        df['feature'] = df['slug'].apply(get_exact_feature)
        df.to_csv(file, index=False)
        print(f"✅ Updated {file}: {df['feature'].value_counts().to_dict()}")
# %%