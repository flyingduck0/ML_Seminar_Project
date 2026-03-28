# ==========================================
# ALL IMPORTS (Exactly as you provided them)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib as plt
import time
import requests as req
import json
import ast
from py_clob_client import ClobClient
from py_clob_client import OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams, BalanceAllowanceParams, AssetType

# ==========================================
# API CONFIGURATION (Exactly as you provided)
# ==========================================
Gamma_api = "https://gamma-api.polymarket.com"
Data_api = "https://data-api.polymarket.com"
Clob_api = "https://clob.polymarket.com"


# ==========================================
# FILE 1: PULLING MARKETS
# ==========================================
all_markets = []
limit = 100
offset = 0

print("Starting download... This will take a few minutes.")

while True:
    params = {"active": True,
              "closed": False,
              "order": "volume24hr", "ascending": False,
              "limit": limit,
              "offset": offset}
    
    response = req.get(f"{Gamma_api}/markets", params = params)
    markets = response.json()

    if not markets:
        break
    all_markets.extend(markets)
    
    # --- ADDITION 1: Simple tracker so you know it is working ---
    if len(all_markets) % 1000 == 0:
        print(f"Downloaded {len(all_markets)} markets so far...")

    if len(markets) < 100:
        break

    offset += limit
    time.sleep(0.2)

print(f"\nFinished! Found {len(all_markets)} markets")

# --- ADDITION 2: Save to CSV immediately ---
print("Saving to polymarket_master_list.csv...")
df = pd.DataFrame(all_markets)
df.to_csv("polymarket_master_list.csv", index=False)
print("Save complete!")

for m in all_markets[0:5]:
    print(f"Question: {m['question']}")
    print(f" Volume 24h: ${m.get('volume24hr', 0):,.0f}")
    print(f" Liquidity: ${m.get('liquidityNum', 0):,.0f}")
    print(f" Prices: {m.get('outcomePrices', 'N/A')}")

# --- ADDITION 3: Clear memory to prevent VS Code freezing ---
del all_markets


# ==========================================
# FILE 2: PULLING EVENTS
# ==========================================
all_events = []
limit = 100
offset = 0

print("\nStarting Events download... (Accessing official database categories)")

while True:
    # Matching the specifications of the original markets pull
    params = {
        "active": True,
        "closed": False,
        "limit": limit,
        "offset": offset
    }
    
    try:
        # Fetching data from the /events endpoint
        response = req.get(f"{Gamma_api}/events", params=params)
        events = response.json()

        # Stop if no more data is returned or if the list is empty
        if not events or len(events) == 0:
            break
        
        all_events.extend(events)
        
        # Tracking progress
        if len(all_events) % 500 == 0:
            print(f"Downloaded {len(all_events)} events so far...")

        # If fewer than the limit are returned, we have reached the end
        if len(events) < limit:
            break

        # Increment offset for next page and sleep to avoid rate limits
        offset += limit
        time.sleep(0.2)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break

print(f"\nFinished! Found {len(all_events)} events.")

# Create the events dataframe
df_events = pd.DataFrame(all_events)

# Save to a new CSV file
print("Saving to polymarket_events_list.csv...")
df_events.to_csv("polymarket_events_list.csv", index=False)

print("Save complete! You now have the second dataset.")


# ==========================================
# FILE 3: CREATING THE MAPPING
# ==========================================
print("\nLoading your existing events file...")
df = pd.read_csv("polymarket_events_list.csv")

# "Lifting the Lid": Convert the text back into actual Python lists
print("Unpacking the nested data...")
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df['markets'] = df['markets'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Extract the Official Label (Politics, Crypto, etc.)
df['official_category'] = df['tags'].apply(lambda x: x[0]['label'] if len(x) > 0 else "Uncategorized")

# THE EXPLOSION: Turn one event row into many market rows
print("Exploding rows to match Market IDs...")
df_exploded = df.explode('markets')

# Clean up: Extract just the ID from the exploded market dictionaries
def get_id(m):
    if isinstance(m, dict):
        return m.get('id')
    return None

df_exploded['market_id'] = df_exploded['markets'].apply(get_id)

# Save the "Master Key"
final_map = df_exploded[['market_id', 'official_category', 'title']].dropna(subset=['market_id'])
final_map.to_csv("market_category_map.csv", index=False)

print(f"Done! Created a clean map with {len(final_map)} rows.")
print("Check 'market_category_map.csv'—it now has a clean Category column.")