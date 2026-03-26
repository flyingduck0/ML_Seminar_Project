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
    if len(all_markets) >= 500:
        break

    if len(batch) < limit:
        break

    offset += limit
    time.sleep(0.2)

print(f"Fetched {len(all_markets)} markets")

df = pd.DataFrame(all_markets)

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

sports_words = ["vs.", "nba", "nfl", "nhl", "mlb", "championship", 
                "tournament", "match", "game", "bowl", "cup",
                "longhorns", "boilermakers", "lakers", "celtics"]

sports_mask = df_themed["search_text"].str.contains(
    "|".join(sports_words), na=False
)
df_themed = df_themed[~sports_mask].copy()  

df_themed = df_themed[df_themed["liquidityNum"] >= 100_000].copy()

print(f"Filtered themed markets: {len(df_themed)}")
print(df_themed[["question","volume24hr","liquidityNum"]].head(10).to_string())

# ── 4. Pull odds history for the single most liquid themed market ───

top_market = df_themed.nlargest(1, "liquidityNum").iloc[0]

print(f"Market: {top_market['question']}")
print(f"Liquidity: ${top_market['liquidityNum']:,.0f}")

# ── Try both ID fields ────────────────────────────────────────────────────
# Polymarket sometimes uses conditionId, sometimes id — try both
id_fields = ["conditionId", "id", "marketMakerAddress"]
market_id = None

for field in id_fields:
    if field in top_market and pd.notna(top_market[field]):
        market_id = top_market[field]
        print(f"Trying field '{field}': {market_id}")
        
        response = requests.get(
            f"{DATA_API}/history",
            params={"market": market_id, "interval": "1d", "fidelity": 60}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200 and len(response.text) > 10:
            try:
                history = response.json()
                if history:
                    print(f"Got {len(history)} data points — using this ID")
                    break
            except:
                print(f"Could not parse response for field '{field}'")
                history = []
        else:
            history = []

# ── Plot only if we got data ──────────────────────────────────────────────
if history:
    hist_df = pd.DataFrame(history)
    print(f"\nColumns in history data: {hist_df.columns.tolist()}")
    print(hist_df.head())  # see what we got

    # find the time column and price column
    time_col  = "t" if "t" in hist_df.columns else hist_df.columns[0]
    price_col = "p" if "p" in hist_df.columns else hist_df.columns[1]

    hist_df[time_col]  = pd.to_datetime(hist_df[time_col], unit="s")
    hist_df = hist_df.set_index(time_col).sort_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(hist_df.index, hist_df[price_col], color="#7F77DD", linewidth=1.8)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title(top_market["question"], fontsize=11, pad=10)
    ax.set_ylabel("Implied probability")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    plt.savefig("demo_market_odds.png", dpi=150)
    plt.show()
    print("Plot saved.")

else:
    # If top market failed, print IDs of next 5 so you can debug
    print("\nTop market history unavailable. Showing next candidates:")
    print(df_themed[["question", "conditionId" if "conditionId" in df_themed.columns else "id", 
                      "liquidityNum"]].head(5).to_string())


