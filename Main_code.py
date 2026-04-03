# %%
# ==========================================
# ALL IMPORTS (Exactly as you provided them)
# ==========================================
import numpy as np
import matplotlib as plt
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

# ==========================================
# API CONFIGURATION (Exactly as you provided)
# ==========================================
Gamma_api = "https://gamma-api.polymarket.com"
Data_api = "https://data-api.polymarket.com"
Clob_api = "https://clob.polymarket.com"
# %%


# %%
# ==========================================
# FILE 1: PULLING EVENTS (TOP-DOWN ARCHITECTURE)
# ==========================================
limit = 100
offset = 0
total_downloaded = 0
jsonl_filename = "full_list_events.jsonl"

if os.path.exists(jsonl_filename):
    os.remove(jsonl_filename)
    print("Existing file found and deleted. Starting with a clean slate.")

print("\nStarting Events download... (Accessing official database categories)")

# Setup requests session with retry strategy
session = req.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("http://", adapter)
session.mount("https://", adapter)

while True:
    params = {
        "limit": limit,
        "offset": offset
    }
    
    try:
        response = session.get(f"{Gamma_api}/events", params=params, timeout=15)
        response.raise_for_status()
        events = response.json()

        if not events or len(events) == 0:
            break
        
        with open(jsonl_filename, 'a', encoding='utf-8') as f:
            for event in events:
                f.write(json.dumps(event) + '\n')

        total_downloaded += len(events)
        
        if total_downloaded % 500 == 0:
            print(f"Downloaded {total_downloaded} events so far...")

        if len(events) < limit:
            break

        offset += limit
        time.sleep(0.2)
        
    except Exception as e:
        print(f"An error occurred: {e}. Gracefully breaking loop to save collected data.")
        break

print(f"\nFinished! Found and securely saved {total_downloaded} events to {jsonl_filename}.")
# %%

# %%
# ==========================================
# Phase 2: Extraction and Filtering Events (Ultra-Strict US-Only Edition)
# ==========================================
import json
import os
import time
import requests as req
from collections import defaultdict

RAW_EVENTS_FILE = "full_list_events.jsonl"
MACRO_EVENTS_FILE = "PILLAR_MACRO_EVENTS.jsonl"
Gamma_api = "https://gamma-api.polymarket.com"

# THE MASTER JUNK FILTER: Instantly destroys Crypto, Politics, Foreign Markets, and Fed Drama
BLACKLIST = [
    # Crypto & Politics
    "bitcoin", "btc", "ethereum", "eth", "crypto", "trump", "biden", "election", "harris", "debate",
    # Global Spillage (Foreign Markets)
    "brazil", "mexico", "germany", "eurozone", "u.k.", "uk", "france", "china", "canada", 
    "europe", "argentina", "global", "ecb", "boe", "japan",
    # Qualitative Fed Drama
    "powell say", "press conference", "nomination", "out as", "chair", "governor", 
    "senate", "confirm", "speech", "hearing"
]

# PILLAR 1: Strict keywords for local text/slug pulling
SLUG_KEYWORDS = {
    "FED_POLICY": ["fed-funds-rate", "target-rate"],
    "LABOR_MARKET": ["unemployment-rate", "nonfarm-payrolls"],
    "INFLATION": ["cpi-yoy", "cpi-mom", "core-cpi", "us-inflation"], 
    "GDP_GROWTH": ["gdp-growth"] 
}

# PILLAR 2: Exact matching for API tags (NO substrings allowed)
TAG_KEYWORDS = {
    "FED_POLICY": ["fomc", "federal reserve", "interest rate", "rate cut", "fed rates"],
    "LABOR_MARKET": ["unemployment", "nonfarm payroll", "nfp", "jobless claims", "jobs report"],
    "INFLATION": ["cpi", "core cpi", "us inflation"],
    "GDP_GROWTH": ["gdp", "us gdp"]
}

def extract_events():
    print(f"\n--- PART 1: THE ULTIMATE PHASE 2 (US-ONLY HYBRID) ---")
    
    tracking_dict = {}
    local_extracted_count = 0
    
    # Step 1: Local Text Extraction
    if os.path.exists(RAW_EVENTS_FILE):
        print("Scanning local events with strict slug keywords...")
        with open(RAW_EVENTS_FILE, 'r', encoding='utf-8') as infile:
            for line in infile:
                if not line.strip(): continue
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                    
                event_slug = str(event.get('slug', '')).strip().lower()
                event_title = str(event.get('title', '')).strip().lower()
                
                # Check Master Blacklist first
                if any(bad_word in event_slug or bad_word in event_title for bad_word in BLACKLIST):
                    continue

                matched_pillar = None
                for pillar, keywords in SLUG_KEYWORDS.items():
                    if any(kw.lower() in event_slug or kw.lower() in event_title for kw in keywords):
                        matched_pillar = pillar
                        break
                        
                if matched_pillar:
                    event['macro_pillar'] = matched_pillar
                    event_id = str(event.get('id', ''))
                    if event_id and event_id not in tracking_dict:
                        tracking_dict[event_id] = event
                        local_extracted_count += 1
    else:
        print(f"[Warning] Local file {RAW_EVENTS_FILE} not found.")

    # Step 2: Dynamic Tag Discovery (EXACT MATCHING ONLY)
    print("Discovering backend Tags via API (Exact Matches Only)...")
    tags_url = f"{Gamma_api}/tags"
    limit = 100
    offset = 0
    pillar_tags = defaultdict(list)
    total_tags_discovered = 0
    
    while True:
        try:
            response = req.get(tags_url, params={"limit": limit, "offset": offset}, timeout=15)
            response.raise_for_status()
            tags_data = response.json()
            
            if not tags_data or len(tags_data) == 0: break
                
            for tag in tags_data:
                tag_label = str(tag.get('label', '')).lower()
                tag_id = str(tag.get('id', ''))
                
                for pillar, keywords in TAG_KEYWORDS.items():
                    # EXACT MATCH check (==) instead of substring (in)
                    if any(kw.lower() == tag_label for kw in keywords):
                        if tag_id not in pillar_tags[pillar]:
                            pillar_tags[pillar].append(tag_id)
                            total_tags_discovered += 1
                        break
            
            offset += limit
            time.sleep(0.2)
        except Exception as e:
            print(f"[Error] Tag discovery failed: {e}")
            break

    # Step 3: API Fetching
    print("Fetching missing historical events from API using discovered tags...")
    events_url = f"{Gamma_api}/events"
    api_extracted_count = 0
    
    for pillar, tag_ids in pillar_tags.items():
        for t_id in tag_ids:
            limit = 100
            offset = 0
            while True:
                try:
                    params = {"limit": limit, "offset": offset, "tag_id": t_id, "closed": "true", "active": "false"}
                    response = req.get(events_url, params=params, timeout=15)
                    response.raise_for_status()
                    events_data = response.json()
                    
                    if not events_data or len(events_data) == 0: break
                        
                    for event in events_data:
                        event_slug = str(event.get('slug', '')).strip().lower()
                        event_title = str(event.get('title', '')).strip().lower()
                        event_id = str(event.get('id', ''))
                        
                        # Check Master Blacklist before saving API events
                        if any(bad_word in event_slug or bad_word in event_title for bad_word in BLACKLIST):
                            continue

                        if event_id and event_id not in tracking_dict:
                            event['macro_pillar'] = pillar
                            tracking_dict[event_id] = event
                            api_extracted_count += 1
                            
                    offset += limit
                    time.sleep(0.2)
                except Exception as e:
                    print(f"[Error] Event fetching failed for tag {t_id}: {e}")
                    break
                    
    # Step 4: Storage & Auditing
    final_pillar_counts = defaultdict(int)
    
    with open(MACRO_EVENTS_FILE, 'w', encoding='utf-8') as macro_out:
        for event_id, event in tracking_dict.items():
            macro_out.write(json.dumps(event) + '\n')
            pillar = event.get('macro_pillar', 'UNKNOWN')
            final_pillar_counts[pillar] += 1
            
    print("\n" + "=" * 50)
    print("ULTRA-STRICT US-ONLY EXTRACTION SUMMARY REPORT")
    print("=" * 50)
    print(f"Total unique Tag IDs autonomously discovered: {total_tags_discovered}")
    print(f"Total events extracted locally (Strict): {local_extracted_count}")
    print(f"Total missing historical events fetched via API (Tags): {api_extracted_count}")
    print(f"Total unique macro events saved: {len(tracking_dict)}")
    print("-" * 50)
    print("FINAL PILLAR BREAKDOWN:")
    for pillar, count in final_pillar_counts.items():
        print(f"  -> {pillar}: {count} events")
    print("=" * 50)

if __name__ == "__main__":
    extract_events()
# %%

# %%
# ==========================================
# Phase 3: Flattening Markets
# ==========================================
MACRO_EVENTS_FILE = "PILLAR_MACRO_EVENTS.jsonl"
FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"

def flatten_markets():
    if not os.path.exists(MACRO_EVENTS_FILE):
        print(f"Error: Could not find {MACRO_EVENTS_FILE}")
        return

    print(f"\n--- PART 2: FLATTENING (PHASE 3) ---")
    total_events_processed = 0
    total_markets_flattened = 0
    markets_skipped = 0
    flat_records = []

    with open(MACRO_EVENTS_FILE, 'r', encoding='utf-8') as infile, \
         open(FLAT_MARKETS_FILE, 'w', encoding='utf-8') as flat_out:
        
        for line in infile:
            if not line.strip(): 
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            total_events_processed += 1
            event_id = event.get('id')
            macro_pillar = event.get('macro_pillar')
            slug = event.get('slug')
            question = event.get('title')
            
            markets_raw = event.get('markets', [])
            if isinstance(markets_raw, str):
                try:
                    markets = json.loads(markets_raw)
                except json.JSONDecodeError:
                    try:
                        markets = ast.literal_eval(markets_raw)
                    except (ValueError, SyntaxError):
                        markets = []
            else:
                markets = markets_raw
                
            if isinstance(markets, list):
                for mkt in markets:
                    if not isinstance(mkt, dict): 
                        continue
                        
                    condition_id = mkt.get("conditionId")
                    if not condition_id:
                        markets_skipped += 1
                        continue
                        
                    market_id = mkt.get("id") or mkt.get("market_id")
                    group_item_title = mkt.get("groupItemTitle")
                    
                    flat_record = {
                        "event_id": event_id,
                        "macro_pillar": macro_pillar,
                        "market_id": market_id,
                        "slug": slug,
                        "condition_id": condition_id,
                        "question": question,
                        "group_item_title": group_item_title
                    }
                    flat_out.write(json.dumps(flat_record) + '\n')
                    flat_records.append(flat_record)
                    total_markets_flattened += 1

    print("\n" + "=" * 50)
    print("VERIFICATION & SUMMARY REPORT")
    print("=" * 50)
    print(f"Total Events Processed: {total_events_processed}")
    print(f"Total Markets Flattened: {total_markets_flattened}")
    print(f"Markets Skipped: {markets_skipped}")
    print("-" * 50)
    print("QA Sample (2 random rows from flattened file):")
    if len(flat_records) >= 2:
        samples = random.sample(flat_records, 2)
        for i, s in enumerate(samples, 1):
            print(f"Sample {i}:")
            print(json.dumps(s, indent=2))
    else:
        print("Not enough records to sample.")
    print("=" * 50)
    print(f"Data saved to:\n  -> {FLAT_MARKETS_FILE}")

if __name__ == "__main__":
    flatten_markets()
# %%

# %%
# ==========================================
# PHASE 4: ID-TARGETED MARKET DATA FETCH
# ==========================================
FLAT_MARKETS_FILE = "FLAT_MARKETS_FOR_API.jsonl"
SYNCED_MARKET_FILE = "SYNCED_MARKET_DATA.json"
GAMMA_API_BASE = "https://gamma-api.polymarket.com/markets"

def fetch_synced_markets():
    if not os.path.exists(FLAT_MARKETS_FILE):
        print(f"Error: Could not find {FLAT_MARKETS_FILE}")
        return
        
    print(f"\n--- PART 3: ID-TARGETED API FETCHING (PHASE 4) ---")
    final_market_data = []
    total_processed = 0
    total_failed = 0
    
    with open(FLAT_MARKETS_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            market_id = record.get("market_id")
            macro_pillar = record.get("macro_pillar")
            slug = record.get("slug")
            question = record.get("question")
            
            if not market_id:
                continue
                
            total_processed += 1
            
            try:
                # Correct API Endpoint: Call the Gamma API using the specific ID path
                response = req.get(f"{GAMMA_API_BASE}/{market_id}", timeout=15)
                
                if response.status_code == 404:
                    total_failed += 1
                    print(f"[Warning] Market ID {market_id} not found, skipping.")
                    time.sleep(0.3)
                    continue
                    
                response.raise_for_status()
                api_data = response.json()
                
                # Integrity Sync: Verify the id in the response matches the requested market_id
                api_id = str(api_data.get("id", ""))
                if api_id != str(market_id):
                    print(f"[Warning] ID mismatch for requested {market_id} (got {api_id}), skipping.")
                    total_failed += 1
                    time.sleep(0.3)
                    continue
                
                # Inject our macro_pillar and our slug
                api_data["macro_pillar"] = macro_pillar
                api_data["slug"] = slug
                
                final_market_data.append(api_data)
                
            except Exception as e:
                total_failed += 1
                print(f"[Warning] Failed to fetch market ID {market_id}: {e}")
            
            if total_processed % 10 == 0:
                print(f"Attempted {total_processed}: Question: {question} | Market ID: {market_id}")
                
            time.sleep(0.3)
            
    with open(SYNCED_MARKET_FILE, 'w', encoding='utf-8') as outfile:
        json.dump(final_market_data, outfile, indent=4)
        
    print("\n" + "=" * 50)
    print("FETCHING SUMMARY REPORT")
    print("=" * 50)
    print(f"Total Markets Attempted: {total_processed}")
    print(f"Total Markets Fetched & Synced: {len(final_market_data)}")
    print(f"Total Markets Failed/Skipped: {total_failed}")
    print(f"Synced data saved to: {SYNCED_MARKET_FILE}")

if __name__ == "__main__":
    fetch_synced_markets()
# %%

# %%
# ==========================================
# PHASE 5: DYNAMIC CSV EXPORT & FULL RETENTION
# ==========================================
import json
import os
import pandas as pd

GOLDEN_CATALOG_FILE = "GOLDEN_MACRO_CATALOG.json"
STATIC_CSV_FILE = "MACRO_MARKETS_STATIC.csv"

def export_static_csv():
    input_file = GOLDEN_CATALOG_FILE
    
    # Fallback in case the file hasn't been renamed from Phase 4 yet
    if not os.path.exists(input_file):
        if os.path.exists("SYNCED_MARKET_DATA.json"):
            print(f"[Info] {input_file} not found. Falling back to SYNCED_MARKET_DATA.json")
            input_file = "SYNCED_MARKET_DATA.json"
        else:
            print(f"Error: Could not find {input_file} or SYNCED_MARKET_DATA.json")
            return
            
    print(f"\n--- PART 4: DYNAMIC CSV EXPORT (PHASE 5) ---")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        market_data = json.load(f)
        
    enriched_data = []
    initial_markets = len(market_data)
    dropped_markets = 0
    
    for mkt in market_data:
        # Category Check
        mkt["macro_pillar"] = mkt.get("macro_pillar") or "UNASSIGNED"
        
        # Safely calculate total_volume
        try:
            vol_amm = float(mkt.get("volumeAmm") or 0)
        except (ValueError, TypeError):
            vol_amm = 0.0
        try:
            vol_clob = float(mkt.get("volumeClob") or 0)
        except (ValueError, TypeError):
            vol_clob = 0.0
            
        total_vol = vol_amm + vol_clob
        if total_vol < 500:
            dropped_markets += 1
            continue
            
        mkt["total_volume"] = total_vol
        
        # Safely calculate total_liquidity
        try:
            liq_amm = float(mkt.get("liquidityAmm") or 0)
        except (ValueError, TypeError):
            liq_amm = 0.0
        try:
            liq_clob = float(mkt.get("liquidityClob") or 0)
        except (ValueError, TypeError):
            liq_clob = 0.0
        mkt["total_liquidity"] = liq_amm + liq_clob
        
        # Remove text bloat
        mkt.pop("description", None)
        mkt.pop("rules", None)
        
        enriched_data.append(mkt)
        
    # Load into pandas DataFrame (handles Union Schema automatically)
    df = pd.DataFrame(enriched_data)
    
    # Column Reordering (CRITICAL)
    if not df.empty:
        if "id" in df.columns and "market_id" not in df.columns:
            df.rename(columns={"id": "market_id"}, inplace=True)
            
        front_cols = ["market_id", "macro_pillar", "question"]
        existing_front = [c for c in front_cols if c in df.columns]
        remaining = [c for c in df.columns if c not in existing_front]
        df = df[existing_front + remaining]

    # Ensure total_volume is explicitly float for Phase 6 model weighting
    if not df.empty and "total_volume" in df.columns:
        df["total_volume"] = df["total_volume"].astype(float)
    
    # Standardize endDate format to YYYY-MM-DD
    if "endDate" in df.columns:
        df["endDate"] = pd.to_datetime(df["endDate"], errors='coerce').dt.strftime('%Y-%m-%d')
        
    # Export without index
    df.to_csv(STATIC_CSV_FILE, index=False)
    
    # Terminal Audit
    print("\n" + "=" * 50)
    print("CSV EXPORT & FILTERING SUMMARY REPORT")
    print("=" * 50)
    print(f"Total Initial Markets: {initial_markets}")
    print(f"Total Markets Dropped (< $500 volume): {dropped_markets}")
    print(f"Total Golden Markets Exported: {len(df)}")
    
    if not df.empty and "macro_pillar" in df.columns:
        print("\nPillar Breakdown:")
        for pillar, count in df["macro_pillar"].value_counts().items():
            print(f"  - {pillar}: {count}")
            
    print("=" * 50)
    print(f"Filtered data successfully exported to: {STATIC_CSV_FILE}")

if __name__ == "__main__":
    export_static_csv()
# %%

# %%
# ==========================================
# PHASE 6: PILLAR SEPARATION & EXPORT
# ==========================================
import pandas as pd
import os

INPUT_CSV = "MACRO_MARKETS_STATIC.csv"

def split_markets_by_pillar():
    if not os.path.exists(INPUT_CSV):
        print(f"[Error] {INPUT_CSV} not found. Please ensure Phase 5 ran successfully.")
        return
        
    print(f"\n--- PART 5: SPLITTING MARKETS BY PILLAR (PHASE 6) ---")
    
    try:
        # Load the master CSV
        df = pd.read_csv(INPUT_CSV)
        
        # Ensure the macro_pillar column exists
        if 'macro_pillar' not in df.columns:
            print("[Error] The column 'macro_pillar' is missing from the CSV.")
            return
            
        # Get all unique pillars
        unique_pillars = df['macro_pillar'].dropna().unique()
        
        print(f"Found {len(unique_pillars)} unique pillars. Starting separation...\n")
        
        total_exported = 0
        
        for pillar in unique_pillars:
            if pillar == "UNASSIGNED":
                continue
                
            # Filter the dataframe for the current pillar
            pillar_df = df[df['macro_pillar'] == pillar]
            
            # Create a clean filename
            output_filename = f"{pillar}_MARKETS.csv"
            
            # Export without index
            pillar_df.to_csv(output_filename, index=False)
            total_exported += len(pillar_df)
            
            print(f"  -> Exported {output_filename} ({len(pillar_df)} markets)")
            
        print("\n" + "=" * 50)
        print("PHASE 6 SUMMARY REPORT")
        print("=" * 50)
        print(f"Total Markets Separated & Exported: {total_exported}")
        print("=" * 50)
        print("Separation complete. You can now process each pillar individually.")

    except Exception as e:
        print(f"[Error] Failed to process CSV: {e}")

if __name__ == "__main__":
    split_markets_by_pillar()
# %%

# %%
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("GDP_GROWTH_MARKETS.csv")

# 2. Your exact slug logic
def is_production_gdp(slug):
    if not isinstance(slug, str): 
        return None
    
    slug = slug.lower().strip()
    
    # Check for Quarterly
    if slug.startswith("us-gdp-growth-in-q"):
        return "GDP_QUARTERLY"
        
    # Check for Yearly
    if slug.startswith("us-gdp-growth-in-20"):
        return "GDP_YEARLY"
        
    return None

# 3. Apply filter and keep the WHOLE table
df['category'] = df['slug'].apply(is_production_gdp)
final_table = df[df['category'].notnull()].copy()

# 4. Save to the final file name
final_table.to_csv("FINAL_US_GDP_MARKETS.csv", index=False)

print(f"Done. Saved {len(final_table)} rows to FINAL_US_GDP_MARKETS.csv")
# %%

