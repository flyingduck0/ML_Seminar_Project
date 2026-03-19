import numpy as np
import pandas as pd
import matplotlib as plt
import time
import requests as req
import json
from py_clob_client import ClobClient
from py_clob_client import OrderArgs, MarketOrderArgs, OrderType, OpenOrderParams, BalanceAllowanceParams, AssetType

Gamma_api = "https://gamma-api.polymarket.com"
Data_api = "https://data-api.polymarket.com"
Clob_api = "https://clob.polymarket.com"

all_markets = []
limit = 100
offset = 0

while True:
    params = {"active": True,
                    "closed": False,
                    "order": "volume24hr", "ascending": False,
                    "limit": limit,
                    "offset": offset}
#search_term = "Oil"
    response = req.get(f"{Gamma_api}/markets", params = params)

    markets = response.json()

    if not markets:
        break
    all_markets.extend(markets)

    if len(markets) < 100:
        break

    offset += limit
    time.sleep(0.2)


print(f"Found {len(all_markets)} markets")

for m in all_markets[0:5]:
    print(f"Question: {m['question']}")
    print(f" Volume 24h: ${m.get('volume24hr', 0):,.0f}")
    print(f" Liquidity: ${m.get('liquidityNum', 0):,.0f}")
    print(f" Prices: {m.get('outcomePrices', 'N/A')}")

