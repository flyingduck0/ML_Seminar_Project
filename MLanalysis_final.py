"""
============================================================
  UNIFIED PIPELINE — FML Group Project 6
  Polymarket Signals → Traditional Asset Returns
  Erasmus School of Economics — FEM11215
============================================================

MERGE OF:
  ML_analysis.py                      (Leo)
  ols_dlmrevised.ipynb                (Apra)
  sharpe_event_backtestrevised.ipynb  (Apra)

STRUCTURE:
  Step 0  → Load & validate                        [Leo]
  Step 1  → Feature engineering + shock detection  [Leo]
  Step 2  → Baseline OLS + DLM                     [Apra]
  Step 3  → Shock panel construction               [Leo]
  Step 4  → Shock-panel OLS + IRF                  [Leo]
  Step 5  → LightGBM on shock panel                [Leo]
  Step 6  → Event study                            [Apra]
  Step 7  → Backtest + Sharpe                      [Apra]

SHOCK FORMULA ALIGNMENT:
  Both sections use Leo's formula:
    Shock_t = 1  if  |ΔPM_t| > μ_t + 2σ_t
  where μ_t and σ_t are the rolling mean/std of the RAW SIGNED
  series computed on the previous 20 LIQUID hours (missing hours
  NaN-masked before rolling). Shock flags are computed once in
  Step 1 and reused by all downstream sections.
  This replaces Apra's original formula (which used abs() inside
  rolling mean and post-hoc zeroing of missing hours).

OUTPUT (all saved to ./results/):
  current_signals.csv
  results_ols_summary.csv
  results_dlm_peak_lags.csv
  results_regression.csv
  results_irf.csv
  results_lgbm.csv
  results_feature_importance.csv
  results_event_study_activity.csv
  results_event_study_regression.csv
  results_backtest.csv
"""

# ── Imports ───────────────────────────────────────────────────────────────────

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, binomtest, spearmanr, ttest_ind
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.api as sm
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — single source of truth for all sections
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = "Final_Pipeline_Data.csv"
RESULTS_DIR = "results"

# ── Leo config ────────────────────────────────────────────────────────────────
ROLLING_W        = 20
SHOCK_SIGMA      = 2.0
HORIZONS         = [1, 2, 3, 4, 5, 6]
N_FOLDS          = 5
MIN_TRAIN        = 200
IRF_P_THRESHOLD  = 0.10
HOLDING_H_DEFAULT = 1
HOLDING_H        = {}   # populated automatically after IRF (Step 4)
LGBM_MACRO_CONTROLS = ["DXY_chg", "US2Y_chg"]
LGBM_PARAMS = dict(
    num_leaves=8, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.1,
    learning_rate=0.05, n_estimators=200, subsample=0.8, colsample_bytree=0.7,
    objective="regression", metric="rmse", verbose=-1, seed=42,
)

# ── Apra config ───────────────────────────────────────────────────────────────
ROLLING_WINDOW = 20       # same value as ROLLING_W
HAC_LAGS       = 5
DLM_K          = 6
PRE_HOURS      = 6
POST_HOURS     = 3
ANNUAL_HOURS   = 252 * 24
TC = {"SPY": 0.0002, "QQQ": 0.0002, "SP500_fut": 0.0001,
      "BTC": 0.0001, "Gold": 0.0001, "Oil": 0.0002}

# Backtest holding periods — Apra DLM defaults.
# Overwritten at runtime by IRF optimal lags from Step 4.
HOLDING_PERIODS_DEFAULT = {
    ("FED_DELTA",          "SPY"):       2,
    ("FED_DELTA",          "QQQ"):       2,
    ("FED_DELTA",          "SP500_fut"): 2,
    ("FED_DELTA",          "BTC"):       2,
    ("FED_DELTA",          "Gold"):      2,
    ("INF_MONTHLY_DELTA",  "Oil"):       1,
    ("UNEMPLOYMENT_DELTA", "SPY"):       1,
    ("UNEMPLOYMENT_DELTA", "QQQ"):       1,
    ("UNEMPLOYMENT_DELTA", "SP500_fut"): 1,
}

# ── Shared ────────────────────────────────────────────────────────────────────
DELTA_COLS = ["FED_DELTA", "GDP_DELTA", "UNEMPLOYMENT_DELTA", "INF_MONTHLY_DELTA"]
MISS_MAP   = {c: f"{c}_is_missing" for c in DELTA_COLS}
PM_FEATURES          = DELTA_COLS
ANNOUNCEMENT_DUMMIES = ["Ann_CPI", "Ann_FOMC", "Ann_Employment", "Ann_GDP"]

ASSETS_LEO  = ["Gold_chg", "BTC_chg", "SPY_chg", "Oil_chg"]
CORE_ASSETS = {"BTC": "BTC_chg", "Gold": "Gold_chg", "Oil": "Oil_chg", "SPY": "SPY_chg"}
EXTENDED_ASSETS = {"QQQ": "QQQ_chg", "SP500_fut": "SP500_fut_chg"}
ALL_ASSETS  = {**CORE_ASSETS, **EXTENDED_ASSETS}

EVENT_PM_MAP = {
    "Ann_FOMC":       "FED_DELTA",
    "Ann_CPI":        "INF_MONTHLY_DELTA",
    "Ann_Employment": "UNEMPLOYMENT_DELTA",
    "Ann_GDP":        "GDP_DELTA",
}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sep(title="", width=68):
    if title:
        pad = (width - len(title) - 2) // 2
        print("─" * pad + f" {title} " + "─" * pad)
    else:
        print("─" * width)

def banner(title):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "~"
    return ""

def directional_accuracy(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return np.nan
    return (np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean()

def ols_hc3(y, X):
    m    = OLS(y, X).fit()
    m_hc = m.get_robustcov_results(cov_type="HC3")
    return m_hc.params[1], m_hc.pvalues[1], m_hc

def mcnemar_test(y_true, pred_lgbm, pred_ols):
    correct_lgbm = np.sign(pred_lgbm) == np.sign(y_true)
    correct_ols  = np.sign(pred_ols)  == np.sign(y_true)
    n_10 = int(np.sum( correct_lgbm & ~correct_ols))
    n_01 = int(np.sum(~correct_lgbm &  correct_ols))
    n_d  = n_10 + n_01
    if n_d < 5:
        return -1.0
    return binomtest(n_10, n_d, 0.5, alternative="greater").pvalue

def run_ols_hac(data, y_col, x_cols, hac_lags=HAC_LAGS):
    temp = data[[y_col] + x_cols].dropna().copy()
    X    = sm.add_constant(temp[x_cols])
    y    = temp[y_col]
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

def delta_r2_ftest(baseline_model, augmented_model, pm_feature_names):
    delta_r2    = augmented_model.rsquared - baseline_model.rsquared
    param_names = list(augmented_model.params.index)
    present     = [f for f in pm_feature_names if f in param_names]
    if not present:
        return delta_r2, None
    R = np.zeros((len(present), len(param_names)))
    for i, feat in enumerate(present):
        R[i, param_names.index(feat)] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f_test = augmented_model.f_test(R)
    return delta_r2, f_test

def cumulative_return(df_in, asset_col, start_idx, h):
    end_idx = min(start_idx + h, len(df_in) - 1)
    window  = df_in.loc[start_idx + 1: end_idx, asset_col]
    if len(window) < h or window.isna().any():
        return np.nan
    return window.sum()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD & VALIDATE  [Leo]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 0 — Load & Validate Data")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Rows:          {len(df):,}")
print(f"  Columns:       {df.shape[1]}")
print(f"  Date range:    {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Calendar days: {df['Date'].dt.date.nunique()}")

dupes = df["Date"].duplicated().sum()
if dupes > 0:
    raise ValueError(f"Dataset has {dupes} duplicate timestamps.")
print(f"  Duplicate timestamps: 0 ✓")

expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq="h")
missing_hours = expected.difference(df["Date"])
if len(missing_hours) > 0:
    print(f"  WARNING: {len(missing_hours)} missing hours — idx+h may not equal t+h.")
else:
    print(f"  Temporal continuity: OK ✓")

required = DELTA_COLS + ASSETS_LEO + ["Ann_CPI", "Ann_FOMC", "Ann_Employment"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")
print(f"  Required columns present ✓")

for ctrl in LGBM_MACRO_CONTROLS:
    if ctrl in df.columns:
        n = df[ctrl].notna().sum()
        print(f"  LGBM macro control {ctrl}_L1: {n} non-NaN base ({n/len(df):.1%})")
    else:
        print(f"  WARNING: {ctrl} not in dataset — will be excluded from LightGBM")
LGBM_MACRO_CONTROLS = [c for c in LGBM_MACRO_CONTROLS if c in df.columns]

ALL_ASSETS      = {k: v for k, v in ALL_ASSETS.items()      if v in df.columns}
CORE_ASSETS     = {k: v for k, v in CORE_ASSETS.items()     if v in df.columns}
EXTENDED_ASSETS = {k: v for k, v in EXTENDED_ASSETS.items() if v in df.columns}
ASSETS_LEO      = [c for c in ASSETS_LEO if c in df.columns]
HOLDING_PERIODS_DEFAULT = {k: v for k, v in HOLDING_PERIODS_DEFAULT.items()
                            if k[1] in ALL_ASSETS}

sample_years = (df["Date"].max() - df["Date"].min()).days / 365.25
print(f"  Sample length: {sample_years:.2f} years")
print(f"  Assets: {list(ASSETS_LEO)}")
print(f"  Extended: {list(EXTENDED_ASSETS.keys())}")

pd.DataFrame({
    "column":   df.columns,
    "dtype":    df.dtypes.values,
    "n_nonnan": df.notna().sum().values,
}).to_csv(f"{RESULTS_DIR}/00_column_inventory.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE ENGINEERING + SHOCK DETECTION  [Leo]
# Shock flags computed ONCE here and reused by Steps 3-7.
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1 — Feature Engineering + Shock Detection [Leo]")

# Calendar
df["hour_utc"]    = df["Date"].dt.hour
df["dow_utc"]     = df["Date"].dt.weekday
df["is_us_open"]  = df["hour_utc"].between(14, 21).astype(int)
df["is_lon_open"] = df["hour_utc"].between(8,  17).astype(int)
df["sin_hour"]    = np.sin(2 * np.pi * df["hour_utc"] / 24)
df["cos_hour"]    = np.cos(2 * np.pi * df["hour_utc"] / 24)

# PM rolling stats — NaN-mask missing hours BEFORE rolling (Leo's formula)
for col in DELTA_COLS:
    miss = MISS_MAP[col]
    s = df[col].copy()
    if miss in df.columns:
        s[df[miss] == 1] = np.nan
    df[f"{col}_roll_mu"]  = s.shift(1).rolling(ROLLING_W, min_periods=10).mean()
    df[f"{col}_roll_sig"] = s.shift(1).rolling(ROLLING_W, min_periods=10).std()
    df[f"{col}_z"]        = (s - df[f"{col}_roll_mu"]) / (df[f"{col}_roll_sig"] + 1e-8)

# Asset rolling volatility (lagged)
for col in ASSETS_LEO:
    if col in df.columns:
        df[f"{col}_vol20"] = df[col].shift(1).rolling(20, min_periods=10).std()

# LightGBM macro controls — lagged 1h
for ctrl in LGBM_MACRO_CONTROLS:
    df[f"{ctrl}_L1"] = df[ctrl].shift(1)
lgbm_ctrl_cols = [f"{c}_L1" for c in LGBM_MACRO_CONTROLS if f"{c}_L1" in df.columns]

# Shock detection
sep("Shock Detection  |ΔPM| > μ + 2σ  (signed rolling, NaN-masked missing)")
print(f"\n  Threshold: |ΔPM| > rolling_mean + {SHOCK_SIGMA}σ  (window={ROLLING_W}h)\n")

for col in DELTA_COLS:
    mu  = df[f"{col}_roll_mu"]
    sig = df[f"{col}_roll_sig"]
    s   = df[col]
    df[f"{col}_shock"] = (
        (s.abs() > mu + SHOCK_SIGMA * sig) &
        s.notna() &
        (df[MISS_MAP[col]] == 0 if MISS_MAP[col] in df.columns else True)
    ).astype(int)
    n_shock = df[f"{col}_shock"].sum()
    n_liq   = (df[MISS_MAP[col]] == 0).sum() if MISS_MAP[col] in df.columns else len(df)
    print(f"  {col:<25}: {n_shock:>5} shock hours ({n_shock/n_liq:.1%} of liquid hours)")

total_shock = sum(df[f"{col}_shock"].sum() for col in DELTA_COLS)
print(f"\n  Total shock observations: {total_shock:,}")

ann_flag  = df[ANNOUNCEMENT_DUMMIES].any(axis=1)
any_shock = df[[f"{col}_shock" for col in DELTA_COLS]].any(axis=1)
overlap   = (ann_flag & any_shock).sum()
print(f"  Shocks overlapping announcements: {overlap} "
      f"({100*overlap/max(any_shock.sum(),1):.1f}% of all shocks)")
print(f"  → Confirms most shocks are genuinely intraday")

# current_signals.csv
last = df.iloc[-1]
current_rows = []
for col in DELTA_COLS:
    current_rows.append({
        "signal":     col,
        "timestamp":  last["Date"].strftime("%Y-%m-%dT%H:%M:%S"),
        "delta":      last[col],
        "z_score":    last[f"{col}_z"],
        "roll_mu":    last[f"{col}_roll_mu"],
        "roll_sig":   last[f"{col}_roll_sig"],
        "is_shock":   int(last[f"{col}_shock"]),
        "is_missing": int(last[MISS_MAP[col]]) if MISS_MAP[col] in df.columns else 0,
        "threshold":  last[f"{col}_roll_mu"] + SHOCK_SIGMA * last[f"{col}_roll_sig"],
    })
current_df = pd.DataFrame(current_rows)
current_df.to_csv(f"{RESULTS_DIR}/current_signals.csv", index=False)
print(f"\n  Current signals saved ({last['Date']})")
for _, row in current_df.iterrows():
    shock_flag = " ← SHOCK" if row["is_shock"] else ""
    liq_flag   = " [ILLIQUID]" if row["is_missing"] else ""
    print(f"    {row['signal']:<25}: z={row['z_score']:>7.3f}{shock_flag}{liq_flag}")

print(f"\n  Rolling stats computed ✓")
print(f"  Calendar features computed ✓")
print(f"  Asset volatility computed ✓")
print(f"  LightGBM cross-asset features (lagged): {lgbm_ctrl_cols}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — BASELINE OLS + DISTRIBUTED LAG MODEL  [Apra]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 2 — Baseline OLS + Distributed Lag Model [Apra]")

PM_MISSING = [f"{f}_is_missing" for f in PM_FEATURES]
PM_        = PM_FEATURES   # alias used inside print_comparison

BASE_CONTROLS = [
    "VIX_chg", "DXY_chg", "US2Y_chg", "US10Y_chg",
    "hour_utc", "dow_utc",
] + ANNOUNCEMENT_DUMMIES
BASE_CONTROLS = [c for c in BASE_CONTROLS if c in df.columns]

def print_comparison(asset, baseline, augmented):
    dr2, ftest = delta_r2_ftest(baseline, augmented, PM_)
    print(f"\n{'='*60}")
    print(f"  {asset} — OLS Results")
    print(f"{'='*60}")
    print(f"  Baseline  R²  : {baseline.rsquared:.6f}  (N={int(baseline.nobs)})")
    print(f"  Augmented R²  : {augmented.rsquared:.6f}")
    print(f"  ΔR²           : {dr2:.6f}")
    if ftest is not None:
        fval  = float(np.atleast_1d(ftest.fvalue).flat[0])
        fp    = float(ftest.pvalue)
        s     = "**" if fp < 0.05 else "*" if fp < 0.10 else ""
        print(f"  Joint F-test on PM block: F={fval:.4f}, p={fp:.4f} {s}")
    print("\n  PM signal coefficients (augmented model):")
    for feat in PM_:
        if feat in augmented.params.index:
            b = augmented.params[feat]
            p = augmented.pvalues[feat]
            s = "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.10 else ""
            print(f"    {feat:30s}  β={b:+.6f}  p={p:.4f} {s}")

print("\n" + "="*60)
print("  SECTION 2A — OLS (BASELINE vs AUGMENTED)")
print("="*60)

summary_rows = []

for asset_name, col in ALL_ASSETS.items():
    tag = " [EXTENDED]" if asset_name in EXTENDED_ASSETS else " [CORE]"
    print(f"\n--- {asset_name}{tag} ---")

    d             = df.copy()
    d["y"]        = d[col].shift(-1)
    d["own_lag1"] = d[col].shift(1)
    controls      = ["own_lag1"] + BASE_CONTROLS

    baseline  = run_ols_hac(d, "y", controls)
    augmented = run_ols_hac(d, "y", controls + PM_FEATURES + PM_MISSING)

    print_comparison(asset_name, baseline, augmented)

    dr2, ft = delta_r2_ftest(baseline, augmented, PM_FEATURES)
    fp = float(ft.pvalue) if ft is not None else float("nan")
    row = {
        "asset": asset_name, "is_core": asset_name in CORE_ASSETS,
        "r2_base": round(baseline.rsquared, 6),
        "r2_aug":  round(augmented.rsquared, 6),
        "delta_r2": round(dr2, 6),
        "f_pvalue": round(fp, 4),
        "nobs": int(augmented.nobs),
    }
    for feat in PM_FEATURES:
        if feat in augmented.params.index:
            row[f"beta_{feat}"] = round(augmented.params[feat], 6)
            row[f"p_{feat}"]    = round(augmented.pvalues[feat], 4)
    summary_rows.append(row)

print("\n\n" + "="*60)
print("  SUMMARY TABLE — OLS Results")
print("="*60)
print(f"  {'Asset':10}  {'Core':6}  {'R²_base':>10}  {'R²_aug':>10}  "
      f"{'ΔR²':>10}  {'F-p':>8}  {'N':>6}")
print(f"  {'-'*66}")
for row in summary_rows:
    core = "✓" if row["is_core"] else "ext"
    print(f"  {row['asset']:10}  {core:6}  {row['r2_base']:10.6f}  "
          f"{row['r2_aug']:10.6f}  {row['delta_r2']:10.6f}  "
          f"{row['f_pvalue']:8.4f}  {row['nobs']:>6}")

pd.DataFrame(summary_rows).to_csv(f"{RESULTS_DIR}/results_ols_summary.csv", index=False)
print(f"\nSaved: results_ols_summary.csv")

print("\n\n" + "="*60)
print("  SECTION 2B — DISTRIBUTED LAG MODEL (DLM) — K=6 lags")
print("="*60)

def build_dlm(df_in, target_col, own_col, k=6):
    d             = df_in.copy()
    d["y"]        = d[target_col].shift(-1)
    d["own_lag1"] = d[own_col].shift(1)
    lagged_cols   = []
    for feat in PM_FEATURES:
        for lag in range(1, k + 1):
            c = f"{feat}_lag{lag}"
            d[c] = d[feat].shift(lag)
            lagged_cols.append(c)
    x_cols = ["own_lag1"] + BASE_CONTROLS + lagged_cols
    return d, x_cols

def print_impulse_response(model, asset, pm_signal, k=6):
    params = model.params
    pvals  = model.pvalues
    print(f"\n  Impulse response: {pm_signal} → {asset}")
    for lag in range(1, k + 1):
        col = f"{pm_signal}_lag{lag}"
        if col in params.index:
            s = ("***" if pvals[col]<0.01 else "**" if pvals[col]<0.05
                 else "*" if pvals[col]<0.10 else "")
            print(f"    h={lag}: β={params[col]:+.7f}  p={pvals[col]:.4f} {s}")

DLM_PAIRS = {
    "BTC":      [("FED_DELTA", "BTC")],
    "Gold":     [("FED_DELTA", "Gold"), ("GDP_DELTA", "Gold")],
    "Oil":      [("INF_MONTHLY_DELTA", "Oil")],
    "SPY":      [("FED_DELTA", "SPY"), ("UNEMPLOYMENT_DELTA", "SPY")],
    "QQQ":      [("FED_DELTA", "QQQ")],
    "SP500_fut":[("FED_DELTA", "SP500_fut")],
}

dlm_summary = []

for asset_name, col in ALL_ASSETS.items():
    d_dlm, x_cols = build_dlm(df, col, col, k=DLM_K)
    dlm            = run_ols_hac(d_dlm, "y", x_cols)
    tag = " [EXTENDED]" if asset_name in EXTENDED_ASSETS else ""
    print(f"\n{asset_name}{tag} DLM — R²={dlm.rsquared:.6f}  N={int(dlm.nobs)}")
    for pm_signal, _ in DLM_PAIRS.get(asset_name, []):
        print_impulse_response(dlm, asset_name, pm_signal)
    for pm_signal in PM_FEATURES:
        lags_sig = []
        for lag in range(1, DLM_K + 1):
            c = f"{pm_signal}_lag{lag}"
            if c in dlm.params.index and dlm.pvalues[c] < 0.10:
                lags_sig.append((lag, dlm.pvalues[c], dlm.params[c]))
        if lags_sig:
            best = min(lags_sig, key=lambda x: x[1])
            dlm_summary.append({
                "asset": asset_name, "pm_signal": pm_signal,
                "peak_lag_dlm": best[0], "beta_at_peak": round(best[2], 7),
                "p_at_peak": round(best[1], 4),
            })

if dlm_summary:
    pd.DataFrame(dlm_summary).to_csv(f"{RESULTS_DIR}/results_dlm_peak_lags.csv", index=False)
    print(f"\nSaved: results_dlm_peak_lags.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — BUILD SHOCK PANEL  [Leo]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 3 — Build Shock Panel [Leo]")

sep("3A: Building Shock Panel")

panel_rows = []
for col in DELTA_COLS:
    shock_idx = df[df[f"{col}_shock"] == 1].index.tolist()
    for idx in shock_idx:
        if idx < ROLLING_W + 1:
            continue
        if idx + 1 >= len(df):
            continue

        ann_cpi  = df.loc[idx, "Ann_CPI"]        if "Ann_CPI"        in df.columns else 0
        ann_fomc = df.loc[idx, "Ann_FOMC"]       if "Ann_FOMC"       in df.columns else 0
        ann_emp  = df.loc[idx, "Ann_Employment"] if "Ann_Employment" in df.columns else 0
        ann_gdp  = df.loc[idx, "Ann_GDP"]        if "Ann_GDP"        in df.columns else 0

        row = {
            "pm_signal":       col,
            "shock_hour":      df.loc[idx, "Date"],
            "pm_delta":        df.loc[idx, col],
            "pm_sign":         np.sign(df.loc[idx, col]),
            "pm_z":            df.loc[idx, f"{col}_z"],
            "pm_vol":          df.loc[idx, f"{col}_roll_sig"],
            "hour_utc":        df.loc[idx, "hour_utc"],
            "dow_utc":         df.loc[idx, "dow_utc"],
            "is_us_open":      df.loc[idx, "is_us_open"],
            "is_lon_open":     df.loc[idx, "is_lon_open"],
            "sin_hour":        df.loc[idx, "sin_hour"],
            "cos_hour":        df.loc[idx, "cos_hour"],
            "Ann_CPI":         ann_cpi,
            "Ann_FOMC":        ann_fomc,
            "Ann_Employment":  ann_emp,
            "Ann_GDP":         ann_gdp,
            "is_announcement": int(ann_cpi + ann_fomc + ann_emp + ann_gdp > 0),
        }
        for ctrl in lgbm_ctrl_cols:
            row[ctrl] = df.loc[idx, ctrl] if ctrl in df.columns else np.nan
        for asset in ASSETS_LEO:
            if asset not in df.columns:
                continue
            for h in HORIZONS:
                future_idx = idx + h
                row[f"{asset}_h{h}"] = (
                    df.loc[future_idx, asset] if future_idx < len(df) else np.nan
                )
            row[f"{asset}_vol20"] = (
                df.loc[idx, f"{asset}_vol20"] if f"{asset}_vol20" in df.columns else np.nan
            )
        panel_rows.append(row)

panel = pd.DataFrame(panel_rows)
panel = panel.sort_values("shock_hour").reset_index(drop=True)

print(f"\n  Total shock events: {len(panel):,}")
print(panel["pm_signal"].value_counts().to_string())
print(f"\n  Announcement overlap: {panel['is_announcement'].sum()} of {len(panel)}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SHOCK-PANEL OLS + IRF  [Leo]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 4 — Shock-Panel OLS + IRF [Leo]")

sep("4A: Core Regression — ΔPM_shock → Asset Return (bivariate, HC3)")
print()
print(f"  Model: r(t+h) = α + β×ΔPM_t + ε   [bivariate, shock hours only]")
print(f"  Standard errors: HC3 robust (heteroskedasticity-consistent)")
print(f"  β = PM signal predictive power (no macro controls — PDF spec)")
print(f"  + directional t-test on β\n")

reg_results = []

for pm_col in DELTA_COLS:
    sub = panel[panel["pm_signal"] == pm_col].copy()
    if len(sub) < 20:
        continue
    for asset in ASSETS_LEO:
        if asset not in df.columns:
            continue
        for h in HORIZONS:
            ret_col = f"{asset}_h{h}"
            if ret_col not in sub.columns:
                continue
            data = sub[["pm_delta", ret_col]].dropna()
            if len(data) < 10:
                continue
            X = add_constant(data["pm_delta"])
            beta, p_hc3, _ = ols_hc3(data[ret_col], X)
            r_corr  = data["pm_delta"].corr(data[ret_col])
            dir_ret = np.sign(data["pm_delta"]) * data[ret_col]
            _, p_dir = ttest_1samp(dir_ret.dropna(), 0)
            reg_results.append({
                "pm_signal": pm_col, "asset": asset, "h": h,
                "n": len(data), "beta_pm": beta, "p_hc3": p_hc3,
                "r": r_corr, "mean_dir": dir_ret.mean(), "p_dir": p_dir,
            })

res_df = pd.DataFrame(reg_results)

sig = res_df[(res_df["p_hc3"] < 0.10) | (res_df["p_dir"] < 0.10)].sort_values("p_hc3")
print(f"  Significant results (p_HC3 < 0.10):\n")
print(f"  {'PM Signal':<22} {'Asset':<10} {'h':>2} {'n':>4} "
      f"{'beta_pm':>10} {'p_HC3':>8} {'r':>7} {'p_dir':>8}")
print(f"  {'─'*72}")
for _, row in sig.iterrows():
    print(f"  {row['pm_signal']:<22} {row['asset']:<10} {row['h']:>2} {row['n']:>4} "
          f"{row['beta_pm']:>10.6f} {row['p_hc3']:>8.4f}{stars(row['p_hc3']):<3} "
          f"{row['r']:>7.4f} {row['p_dir']:>8.4f}{stars(row['p_dir'])}")

sep("4B: IRF — Automatic Lag Selection")

sig_pairs = (
    res_df[res_df["p_hc3"] < IRF_P_THRESHOLD][["pm_signal", "asset"]]
    .drop_duplicates()
    .sort_values(["pm_signal", "asset"])
    .reset_index(drop=True)
)
key_pairs = [
    (r["pm_signal"], r["asset"], f"{r['pm_signal']} → {r['asset']}")
    for _, r in sig_pairs.iterrows()
]

print(f"\n  Coppie selezionate (p_HC3 < {IRF_P_THRESHOLD}):")
for pm_col, asset, label in key_pairs:
    print(f"    {label}")

irf_results = []

for pm_col, asset, label in key_pairs:
    if asset not in df.columns:
        continue
    sub = panel[panel["pm_signal"] == pm_col].copy()
    print(f"\n  {label}:")
    print(f"  {'h':>3}  {'beta_pm':>12}  {'p_HC3':>8}  {'sig':>4}")
    betas, ps, hs_valid = [], [], []
    for h in HORIZONS:
        ret_col = f"{asset}_h{h}"
        if ret_col not in sub.columns:
            continue
        data = sub[["pm_delta", ret_col]].dropna()
        if len(data) < 8:
            continue
        X = add_constant(data["pm_delta"])
        b, p, _ = ols_hc3(data[ret_col], X)
        betas.append(b); ps.append(p); hs_valid.append(h)
        irf_results.append({"pm": pm_col, "asset": asset, "h": h,
                             "beta_pm": b, "p_hc3": p})
        print(f"  {h:>3}h  {b:>12.7f}  {p:>8.4f}  {stars(p):>4}")
    if betas:
        print(f"  Σβ = {sum(betas):.7f}")
        if len(betas) >= 4:
            rho, p_spear = spearmanr(hs_valid, [abs(b) for b in betas])
            interp = ("→ Rapid absorption" if rho < -0.5
                      else "→ Delayed absorption" if rho > 0.3
                      else "→ No clear pattern")
            print(f"  Spearman: ρ={rho:.3f}  p={p_spear:.4f}  {interp}")

irf_df = pd.DataFrame(irf_results)

if len(irf_df) > 0:
    sig_irf = irf_df[irf_df["p_hc3"] < IRF_P_THRESHOLD]
    if len(sig_irf) > 0:
        best_h = (
            sig_irf.sort_values("p_hc3")
            .groupby(["pm", "asset"]).first()
            .reset_index()[["pm", "asset", "h", "p_hc3"]]
        )
        for _, row in best_h.iterrows():
            HOLDING_H[(row["pm"], row["asset"])] = int(row["h"])

print(f"\n  Optimal lags for LightGBM:")
for (pm, asset), h in sorted(HOLDING_H.items()):
    print(f"    {pm:<25} → {asset:<12}  h={h}")
print(f"  All other pairs → default h={HOLDING_H_DEFAULT}")

irf_optimal = []
for (pm, asset), h in HOLDING_H.items():
    row_irf = irf_df[(irf_df["pm"] == pm) & (irf_df["asset"] == asset) & (irf_df["h"] == h)]
    if len(row_irf) > 0:
        irf_optimal.append({
            "pm_signal": pm, "asset": asset, "optimal_h": h,
            "beta_optimal": row_irf.iloc[0]["beta_pm"],
            "p_optimal":    row_irf.iloc[0]["p_hc3"],
        })
irf_optimal_df = pd.DataFrame(irf_optimal) if irf_optimal else pd.DataFrame()

reg_cols = ["pm_signal", "asset", "h", "n", "beta_pm", "p_hc3", "r", "mean_dir", "p_dir"]
if len(res_df) > 0:
    res_df[res_df["p_hc3"] < 0.10][reg_cols].to_csv(
        f"{RESULTS_DIR}/results_regression.csv", index=False)
    print(f"\n  Saved: results_regression.csv")
if len(irf_optimal_df) > 0:
    irf_optimal_df.to_csv(f"{RESULTS_DIR}/results_irf.csv", index=False)
    print(f"  Saved: results_irf.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — LIGHTGBM ON SHOCK PANEL  [Leo]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 5 — LightGBM on Shock Panel [Leo]")

LGBM_FEATURES_BASE = [
    "pm_delta", "pm_z", "pm_vol",
    "hour_utc", "dow_utc", "sin_hour", "cos_hour",
    "is_us_open", "is_lon_open",
    "Ann_CPI", "Ann_FOMC", "Ann_Employment", "Ann_GDP",
] + lgbm_ctrl_cols

lgbm_results = []
feat_imp_all  = []

for pm_col in DELTA_COLS:
    sub_pm = panel[panel["pm_signal"] == pm_col].copy().reset_index(drop=True)
    if len(sub_pm) < MIN_TRAIN + 50:
        continue
    for asset in ASSETS_LEO:
        if asset not in df.columns:
            continue
        h = HOLDING_H.get((pm_col, asset), HOLDING_H_DEFAULT)
        ret_col = f"{asset}_h{h}"
        if ret_col not in sub_pm.columns:
            continue
        feats = LGBM_FEATURES_BASE.copy()
        vol_col = f"{asset}_vol20"
        if vol_col in sub_pm.columns:
            feats.append(vol_col)
        feats = [f for f in feats if f in sub_pm.columns]
        sub = sub_pm[feats + [ret_col]].dropna().reset_index(drop=True)
        if len(sub) < MIN_TRAIN + 20:
            continue

        X = sub[feats]; y = sub[ret_col]; n = len(sub); fs = n // N_FOLDS
        oos_preds = np.full(n, np.nan)
        feat_imp  = np.zeros(len(feats))
        folds_done = 0

        for k in range(N_FOLDS):
            ts = k * fs; te = min(ts + fs, n)
            tr = np.arange(0, max(0, ts - 1))
            if len(tr) < MIN_TRAIN:
                continue
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X.iloc[tr], y.iloc[tr], callbacks=[lgb.log_evaluation(-1)])
            oos_preds[ts:te] = model.predict(X.iloc[ts:te])
            feat_imp += model.feature_importances_
            folds_done += 1

        if folds_done > 0:
            feat_imp = feat_imp / folds_done

        mask = ~np.isnan(oos_preds)
        if mask.sum() < 20:
            continue

        y_oos = y.values[mask]; p_oos = oos_preds[mask]
        da_oos = directional_accuracy(y_oos, p_oos)
        n_correct = round(da_oos * mask.sum())
        p_binom   = binomtest(n_correct, mask.sum(), 0.5, alternative="greater").pvalue

        sub_ols   = sub[["pm_delta", ret_col]].copy()
        ols_preds = np.full(len(sub_ols), np.nan)
        n_ols = len(sub_ols); fs_ols = n_ols // N_FOLDS
        for k in range(N_FOLDS):
            ts = k * fs_ols; te = min(ts + fs_ols, n_ols)
            tr = np.arange(0, max(0, ts - 1))
            if len(tr) < 20:
                continue
            m_ols = OLS(sub_ols[ret_col].iloc[tr],
                        add_constant(sub_ols["pm_delta"].iloc[tr])).fit()
            ols_preds[ts:te] = m_ols.predict(
                add_constant(sub_ols["pm_delta"].iloc[ts:te]))

        mask_ols  = ~np.isnan(ols_preds)
        da_ols    = directional_accuracy(sub_ols[ret_col].values[mask_ols],
                                          ols_preds[mask_ols])
        mask_both = mask & mask_ols
        p_mcnemar = mcnemar_test(y.values[mask_both], oos_preds[mask_both],
                                  ols_preds[mask_both]) if mask_both.sum() >= 5 else -1.0

        lgbm_beats_ols    = int(da_oos > da_ols)
        lgbm_beats_random = int(p_binom < 0.05)

        lgbm_results.append({
            "pm_signal": pm_col, "asset": asset, "h": h,
            "n_panel": len(sub), "n_oos": mask.sum(),
            "DA_OOS_LGBM": da_oos, "DA_OOS_OLS": da_ols, "DA_random": 0.50,
            "p_binom_vs_random": p_binom, "p_mcnemar_vs_ols": p_mcnemar,
            "lgbm_beats_ols": lgbm_beats_ols, "lgbm_beats_random": lgbm_beats_random,
        })

        imp_df = pd.DataFrame({
            "feature": feats, "importance": feat_imp,
            "pm_signal": pm_col, "asset": asset, "h": h,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1
        feat_imp_all.append(imp_df)

        print(f"\n  {pm_col} → {asset} (h={h}, n={len(sub)}, n_oos={mask.sum()}, folds={folds_done})")
        print(f"  DA(LGBM)={da_oos:.2%}  DA(OLS)={da_ols:.2%}  "
              f"p_binom={p_binom:.4f}{stars(p_binom)}  "
              f"p_mcnemar={p_mcnemar:.4f}{stars(p_mcnemar) if p_mcnemar >= 0 else ''}")
        print(f"  LGBM beats OLS: {lgbm_beats_ols}  |  LGBM beats random: {lgbm_beats_random}")
        print(f"  Top features:")
        for _, r in imp_df.head(5).iterrows():
            is_pm   = "  ← PM signal"  if r["feature"] in ["pm_delta","pm_z","pm_vol"] else ""
            is_ctrl = "  ← macro ctrl" if r["feature"] in lgbm_ctrl_cols else ""
            bar = "█" * int(r["importance"] / max(feat_imp.max(), 1) * 20)
            print(f"    {r['feature']:<30}: {r['importance']:>6.1f}  {bar}{is_pm}{is_ctrl}")

lgbm_df = pd.DataFrame(lgbm_results)
feat_imp_all_df = (pd.concat(feat_imp_all, ignore_index=True) if feat_imp_all
                   else pd.DataFrame(columns=["pm_signal","asset","h","feature","importance","rank"]))

if len(lgbm_df) > 0:
    lgbm_df.to_csv(f"{RESULTS_DIR}/results_lgbm.csv", index=False)
    print(f"\n  Saved: results_lgbm.csv")

if len(feat_imp_all_df) > 0:
    fimp_cols = ["pm_signal", "asset", "h", "rank", "feature", "importance"]
    fimp_out  = (feat_imp_all_df
                 .sort_values(["pm_signal","asset","importance"], ascending=[True,True,False])
                 .groupby(["pm_signal","asset"]).head(5)
                 .reset_index(drop=True))
    fimp_out["rank"] = fimp_out.groupby(["pm_signal","asset"]).cumcount() + 1
    fimp_out[fimp_cols].to_csv(f"{RESULTS_DIR}/results_feature_importance.csv", index=False)
    print(f"  Saved: results_feature_importance.csv")

if len(lgbm_df) > 0:
    print(f"\n  Summary:\n")
    print(f"  {'PM Signal':<22} {'Asset':<10} {'h':>2} {'n_oos':>6} "
          f"{'DA_LGBM':>9} {'DA_OLS':>9} {'p_binom':>9} {'p_mcn':>8} {'beats_OLS':>10}")
    print(f"  {'─'*85}")
    for _, r in lgbm_df.sort_values("DA_OOS_LGBM", ascending=False).iterrows():
        mcn = f"{r['p_mcnemar_vs_ols']:.4f}" if r["p_mcnemar_vs_ols"] >= 0 else "  n/a"
        print(f"  {r['pm_signal']:<22} {r['asset']:<10} {r['h']:>2} "
              f"{r['n_oos']:>6} {r['DA_OOS_LGBM']:>9.2%} {r['DA_OOS_OLS']:>9.2%} "
              f"{r['p_binom_vs_random']:>9.4f}{stars(r['p_binom_vs_random']):<2} "
              f"{mcn:>8} {'✓' if r['lgbm_beats_ols'] else '✗':>10}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — EVENT STUDY  [Apra]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 6 — Event Study [Apra]")

print("\n\n" + "="*65)
print("  SECTION 3C — EVENT STUDY")
print("="*65)

print("\nPre-announcement activity test (6h BEFORE announcement vs normal hours):")
print(f"  {'PM Signal':25s}  {'Event':20s}  {'n_events':>8}  {'Ratio':>8}  {'p-value':>10}")
print(f"  {'-'*78}")

activity_results = []
for ann_col, pm_col in EVENT_PM_MAP.items():
    ann_idx = df.index[df[ann_col] == 1].tolist()
    if len(ann_idx) < 3:
        continue
    pre_idx = set()
    for i in ann_idx:
        for j in range(max(0, i - PRE_HOURS), i):
            pre_idx.add(j)
    non_pre_idx = set(df.index) - pre_idx - set(ann_idx)
    event_vals  = df.loc[list(pre_idx),     pm_col].abs().dropna()
    normal_vals = df.loc[list(non_pre_idx), pm_col].abs().dropna()
    if len(event_vals) < 5 or len(normal_vals) < 5:
        continue
    ratio        = event_vals.mean() / (normal_vals.mean() + 1e-10)
    t_stat, pval = ttest_ind(event_vals, normal_vals, equal_var=False)
    s            = "***" if pval<0.01 else "**" if pval<0.05 else "*" if pval<0.10 else ""
    print(f"  {pm_col:25s}  {ann_col:20s}  {len(ann_idx):8d}  "
          f"{ratio:8.2f}x  {pval:10.4f} {s}")
    activity_results.append({
        "pm": pm_col, "event": ann_col, "n_events": len(ann_idx),
        "activity_ratio": round(ratio, 3), "pval_activity": round(pval, 4),
    })

if activity_results:
    pd.DataFrame(activity_results).to_csv(
        f"{RESULTS_DIR}/results_event_study_activity.csv", index=False)
    print(f"\nSaved: results_event_study_activity.csv")

print(f"\n\nPredictive regressions (S_pre → R_post, HC3 robust SEs):")
print(f"  {'Event':20s}  {'PM':25s}  {'Asset':10s}  {'n':>4}  {'β':>10}  {'p(HC3)':>10}")
print(f"  {'-'*88}")

event_study_results = []
asset_order_evt = list(CORE_ASSETS.keys()) + [k for k in EXTENDED_ASSETS if k in ALL_ASSETS]

for ann_col, pm_col in EVENT_PM_MAP.items():
    ann_hours = df.index[df[ann_col] == 1].tolist()
    if len(ann_hours) < 5:
        continue
    for asset_name in asset_order_evt:
        if asset_name not in ALL_ASSETS:
            continue
        asset_col = ALL_ASSETS[asset_name]
        pre_signals, post_returns = [], []
        for idx in ann_hours:
            pre_window  = df.loc[max(0, idx - PRE_HOURS): idx - 1, pm_col]
            s_pre       = pre_window.sum()
            post_end    = min(len(df) - 1, idx + POST_HOURS)
            post_window = df.loc[idx + 1: post_end, asset_col]
            r_post      = post_window.sum()
            if pd.isna(s_pre) or pd.isna(r_post) or post_window.isna().all():
                continue
            pre_signals.append(s_pre)
            post_returns.append(r_post)
        if len(pre_signals) < 5:
            continue
        try:
            X     = sm.add_constant(pre_signals)
            model = sm.OLS(post_returns, X).fit(cov_type="HC3")
            beta  = model.params[1]
            pval  = model.pvalues[1]
            n     = len(pre_signals)
            s     = ("***" if pval<0.01 else "**" if pval<0.05
                     else "*" if pval<0.10 else "~" if pval<0.15 else "")
            tag   = " [ext]" if asset_name in EXTENDED_ASSETS else ""
            print(f"  {ann_col:20s}  {pm_col:25s}  {asset_name+tag:10s}  "
                  f"{n:4d}  {beta:+10.4f}  {pval:10.4f} {s}")
            event_study_results.append({
                "event": ann_col, "pm": pm_col, "asset": asset_name,
                "n_events": n, "beta": round(beta, 6), "pval_hc3": round(pval, 4),
            })
        except Exception:
            continue

if event_study_results:
    pd.DataFrame(event_study_results).to_csv(
        f"{RESULTS_DIR}/results_event_study_regression.csv", index=False)
    print(f"\nSaved: results_event_study_regression.csv")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — BACKTEST & SHARPE RATIO  [Apra]
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 7 — Backtest & Sharpe Ratio [Apra]")

print("""
Strategy:
  Entry: only at PM shock hours (aligned with Step 1 shocks — Leo's formula)
  Direction: sign(PM shock at time t) — honest real-time rule
  Hold: h hours cumulative return (t+1 through t+h) from IRF
  No overlapping trades: block new entries until current trade closes
  Skip: any trade where the full h-hour window has NaN (e.g. SPY overnight)
""")

HOLDING_PERIODS = dict(HOLDING_PERIODS_DEFAULT)

irf_file = f"{RESULTS_DIR}/results_irf.csv"
if os.path.exists(irf_file):
    irf_loaded = pd.read_csv(irf_file)
    for _, row in irf_loaded.iterrows():
        asset_val = row["asset"]
        for k, v in ALL_ASSETS.items():
            if v == asset_val or k == asset_val:
                HOLDING_PERIODS[(row["pm_signal"], k)] = int(row["optimal_h"])
    for key, h in HOLDING_PERIODS_DEFAULT.items():
        if key[1] in EXTENDED_ASSETS and key not in HOLDING_PERIODS:
            HOLDING_PERIODS[key] = h
    print(f"  Holding periods loaded from {irf_file}:")
else:
    print(f"  WARNING: {irf_file} not found — using DLM-derived defaults.")

HOLDING_PERIODS = {k: v for k, v in HOLDING_PERIODS.items() if k[1] in ALL_ASSETS}
for (pm, asset), h in sorted(HOLDING_PERIODS.items()):
    src = "IRF" if os.path.exists(irf_file) else "default"
    print(f"  {pm:30s} → {asset:12s}  h={h} [{src}]")

def run_backtest(df_in, pm_col, asset_name, asset_col, h_hold=1, tc=0.0001):
    shock_rows    = df_in.index[df_in[f"{pm_col}_shock"] == 1].tolist()
    trade_returns = []
    next_available = 0
    for idx in shock_rows:
        if idx < next_available:
            continue
        signal = np.sign(df_in.loc[idx, pm_col])
        if signal == 0:
            continue
        cum_ret = cumulative_return(df_in, asset_col, idx, h_hold)
        if np.isnan(cum_ret):
            continue
        trade_returns.append(signal * cum_ret)
        next_available = idx + h_hold
    if len(trade_returns) < 10:
        return None
    active   = np.array(trade_returns)
    n        = len(active)
    mean_ret = active.mean()
    std_ret  = active.std()
    gross_sr  = np.sqrt(ANNUAL_HOURS) * mean_ret / (std_ret + 1e-10)
    net_sr    = np.sqrt(ANNUAL_HOURS) * (mean_ret - tc) / (std_ret + 1e-10)
    cum_curve = np.cumsum(active)
    max_dd    = (cum_curve - np.maximum.accumulate(cum_curve)).min()
    dir_acc   = (active > 0).mean()
    binom_p   = binomtest(int((active > 0).sum()), n, 0.5, alternative="greater").pvalue
    return {"pm": pm_col, "asset": asset_name, "h": h_hold,
            "n_trades": n, "mean_ret": mean_ret, "cumulative_ret": active.sum(),
            "gross_sharpe": gross_sr, "net_sharpe": net_sr,
            "max_drawdown": max_dd, "dir_accuracy": dir_acc,
            "binom_p": binom_p, "_returns": active}

print(f"\n  {'PM Signal':25s}  {'Asset':10s}  {'h':>2}  {'Trades':>6}  "
      f"{'Gross SR':>9}  {'Net SR':>8}  {'MaxDD':>8}  {'DirAcc':>7}  {'p(dir)':>8}")
print(f"  {'-'*108}")

backtest_results = []
asset_order_bt = list(CORE_ASSETS.keys()) + [k for k in EXTENDED_ASSETS if k in ALL_ASSETS]

for asset_name in asset_order_bt:
    for pm in PM_FEATURES:
        key = (pm, asset_name)
        if key not in HOLDING_PERIODS:
            continue
        h         = HOLDING_PERIODS[key]
        asset_col = ALL_ASSETS[asset_name]
        tc        = TC.get(asset_name, 0.0002)
        result    = run_backtest(df, pm, asset_name, asset_col, h_hold=h, tc=tc)
        if result is None:
            continue
        backtest_results.append(result)
        r    = result
        flag = "✓" if r["net_sharpe"] > 0.5 else ("~" if r["net_sharpe"] > 0 else "✗")
        dstar = ("***" if r["binom_p"]<0.01 else "**" if r["binom_p"]<0.05
                 else "*" if r["binom_p"]<0.10 else "")
        tag  = " [ext]" if asset_name in EXTENDED_ASSETS else ""
        print(f"  {r['pm']:25s}  {asset_name+tag:10s}  {r['h']:2d}  "
              f"{r['n_trades']:6d}  "
              f"{r['gross_sharpe']:+9.3f}  {r['net_sharpe']:+8.3f} {flag}  "
              f"{r['max_drawdown']:+8.4f}  {r['dir_accuracy']:7.1%}  "
              f"{r['binom_p']:8.4f} {dstar}")

print("\n500-simulation random direction benchmark:")
print(f"  {'PM → Asset':42s}  {'Mean Rand SR':>13}  {'Our Gross SR':>13}  {'Edge':>8}")
print(f"  {'-'*82}")

np.random.seed(42)
N_SIM = 500

for r in backtest_results:
    abs_rets = np.abs(r["_returns"])
    n        = len(abs_rets)
    sim_srs  = []
    for _ in range(N_SIM):
        rd    = np.random.choice([-1, 1], size=n)
        sr    = np.sqrt(ANNUAL_HOURS) * (rd * abs_rets).mean() / ((rd * abs_rets).std() + 1e-10)
        sim_srs.append(sr)
    mean_rand = np.mean(sim_srs)
    edge      = r["gross_sharpe"] - mean_rand
    label     = f"{r['pm']} → {r['asset']}"
    print(f"  {label:42s}  {mean_rand:+9.3f} ±{np.std(sim_srs):.2f}  "
          f"{r['gross_sharpe']:+13.3f}  {edge:+8.3f}")

for r in backtest_results:
    r.pop("_returns", None)

if backtest_results:
    pd.DataFrame(backtest_results).to_csv(f"{RESULTS_DIR}/results_backtest.csv", index=False)
    print(f"\nSaved: results_backtest.csv")


# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────

banner("PIPELINE COMPLETE")
print(f"""
  All results saved in: ./{RESULTS_DIR}/

  File                                   Author  Description
  ──────────────────────────────────────────────────────────────────────
  00_column_inventory.csv                Leo     Dataset column metadata
  current_signals.csv                    Leo     Real-time shock state
  results_ols_summary.csv                Apra    Baseline vs augmented OLS
  results_dlm_peak_lags.csv              Apra    DLM peak lags per pair
  results_regression.csv                 Leo     Shock-panel OLS sig pairs
  results_irf.csv                        Leo     Optimal transmission lag
  results_lgbm.csv                       Leo     LightGBM DA metrics
  results_feature_importance.csv         Leo     Top 5 features per model
  results_event_study_activity.csv       Apra    Pre-ann activity ratios
  results_event_study_regression.csv     Apra    S_pre → R_post regressions
  results_backtest.csv                   Apra    Sharpe, drawdown, dir acc
""")
