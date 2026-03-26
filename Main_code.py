import requests
import pandas as pd
import time
import matplotlib.pyplot as plt

GAMMA_API  = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"
CLOB_API   = "https://clob.polymarket.com"

# We are fetching the markets based on 24 hour volume--

all_markets = []
limit, offset = 100, 0

while True:
    params = {
        "active": True,
        "closed": False,
        "order": "volume24hr",
        "ascending": False,
        "limit": limit,
        "offset": offset
    }
    r = requests.get(f"{GAMMA_API}/markets", params=params)
    r.raise_for_status()
    batch = r.json()
    if not batch:
        break
    all_markets.extend(batch)
    if len(batch) < limit:
        break
    offset += limit
    time.sleep(0.2)

df = pd.DataFrame(all_markets)
print(f"Total markets fetched: {len(df)}")


# This will be the scope of the problem. Taking top 50, constitute 80% of the trading volume, the rest would be noise-


df["volume24hr"]   = pd.to_numeric(df.get("volume24hr", 0),   errors="coerce").fillna(0)
df["liquidityNum"] = pd.to_numeric(df.get("liquidityNum", 0), errors="coerce").fillna(0)

total_vol = df["volume24hr"].sum()
top50_vol = df.nlargest(50, "volume24hr")["volume24hr"].sum()
print(f"Top 50 markets = {top50_vol/total_vol:.0%} of total 24h volume")
# → makes the case that you don't need all 36k markets


# Adding keyword filters based on what theme we would pick-


KEYWORDS = ["fed", "rate cut", "interest rate", "fomc",   # monetary policy
            "trump", "election", "tariff",                  # political
            "oil", "ceasefire", "ukraine", "nato"]          # geopolitical

text_cols = [c for c in ["question", "title", "description"] if c in df.columns]
df["search_text"] = df[text_cols].fillna("").agg(" ".join, axis=1).str.lower()

mask = df["search_text"].str.contains("|".join(KEYWORDS), na=False)
df_themed = df[mask].copy()
print(f"\nThemed markets: {len(df_themed)} out of {len(df)}")
print(df_themed[["question", "volume24hr", "liquidityNum"]].head(15).to_string())

# ── 4. Pulling the odds history for the most liquid themed market-
top_market = df_themed.nlargest(1, "liquidityNum").iloc[0]
market_id  = top_market["id"]
print(f"\nPulling history for: {top_market['question']}")

history = requests.get(
    f"{DATA_API}/history",
    params={"market": market_id, "interval": "1d", "fidelity": 60}
).json()

if history:
    hist_df = pd.DataFrame(history)
    hist_df["t"] = pd.to_datetime(hist_df["t"], unit="s")
    hist_df = hist_df.set_index("t").sort_index()

    # Plotting it-

    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_df.index, hist_df["p"], color="#7F77DD", linewidth=1.8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(top_market["question"], fontsize=11, pad=10)
    ax.set_ylabel("Implied probability")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig("demo_market_odds.png", dpi=150)
    plt.show()
    print("Plot saved.")


