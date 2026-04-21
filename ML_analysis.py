"""
PIPELINE STRUCTURE:
  Step 0  → Load & validate data
  Step 1  → Feature engineering
  Step 2  → Shock Panel (Section 3B)
            2A: Shock detection
            2B: Build shock panel
            2C: Bivariate OLS with HC3 robust SE
            2D: IRF — automatic lag selection
  Step 3  → LightGBM (Section 3C)
            Walk-forward, feature importance, DA vs OLS vs random walk
            McNemar test: LightGBM vs OLS
  Step 4  → Output files
            current_signals.csv   — hourly update for dashboard
            results_regression/irf/lgbm/feature_importance.csv — weekly update


"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, binomtest, spearmanr
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH   = "Final_Pipeline_Data.csv"
RESULTS_DIR = "results_Leo"
ROLLING_W   = 20
SHOCK_SIGMA = 2.0
HORIZONS    = [1, 2, 3, 4, 5, 6]
N_FOLDS     = 5
MIN_TRAIN   = 200
IRF_P_THRESHOLD = 0.10

# Cross-asset signals for LightGBM only (lagged t-1)
# NOT used in OLS — OLS 3B is bivariate as per PDF spec.
# LightGBM uses them as conditioning variables to find when PM signals
# are more/less informative — feature importance reveals their role.
LGBM_MACRO_CONTROLS = ["DXY_chg", "US2Y_chg"]

HOLDING_H_DEFAULT = 1
HOLDING_H = {}   # populated automatically after IRF

LGBM_PARAMS = dict(
    num_leaves        = 8,
    min_child_samples = 10,
    reg_alpha         = 0.0,
    reg_lambda        = 0.1,
    learning_rate     = 0.05,
    n_estimators      = 200,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    objective         = "regression",
    metric            = "rmse",
    verbose           = -1,
    seed              = 42,
)

DELTA_COLS = [
    "FED_DELTA", "GDP_DELTA", "UNEMPLOYMENT_DELTA", "INF_MONTHLY_DELTA",
    # INF_YEARLY_DELTA removed: small sample (446 shocks, n_oos=53),
    # DA=37.74% not confirmed by LightGBM. Replaced by INF_MONTHLY_DELTA.
]
MISS_MAP = {c: f"{c}_is_missing" for c in DELTA_COLS}
ASSETS   = ["Gold_chg", "BTC_chg", "SPY_chg", "Oil_chg"]

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
    """
    OLS with HC3 heteroskedasticity-robust standard errors.
    Returns (beta_pm, p_value_pm, full_model) where index 1
    corresponds to the PM signal coefficient (index 0 = constant).
    HC3 is appropriate given extreme kurtosis of PM signals (130-928)
    and volatility clustering in financial returns.
    """
    m    = OLS(y, X).fit()
    m_hc = m.get_robustcov_results(cov_type='HC3')
    return m_hc.params[1], m_hc.pvalues[1], m_hc


def mcnemar_test(y_true, pred_lgbm, pred_ols):
    """
    McNemar test for paired comparison of LightGBM vs OLS directional accuracy.
    Tests H0: the two models make the same number of directional errors.

    Implementation: exact binomial test on discordant pairs.
    n_10 = LGBM correct, OLS wrong  (favors LGBM)
    n_01 = LGBM wrong,  OLS correct (favors OLS)
    H0: P(n_10) = 0.5  →  one-sided test for LGBM superiority.

    Equivalent to scipy.stats.contingency.mcnemar(exact=True, correction=False)
    but uses only binomtest which is already imported.
    """
    correct_lgbm = np.sign(pred_lgbm) == np.sign(y_true)
    correct_ols  = np.sign(pred_ols)  == np.sign(y_true)
    n_10 = int(np.sum( correct_lgbm & ~correct_ols))  # LGBM right, OLS wrong
    n_01 = int(np.sum(~correct_lgbm &  correct_ols))  # LGBM wrong, OLS right
    n_discordant = n_10 + n_01
    if n_discordant < 5:
        return -1.0  # sentinel: insufficient discordant pairs (<5)
    # One-sided: H1 = LGBM makes fewer errors than OLS
    return binomtest(n_10, n_discordant, 0.5, alternative="greater").pvalue


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — LOAD & VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 0 — Load & Validate Data")

df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"], utc=True)
df = df.sort_values("Date").reset_index(drop=True)

print(f"  Rows:          {len(df):,}")
print(f"  Columns:       {df.shape[1]}")
print(f"  Date range:    {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"  Calendar days: {df['Date'].dt.date.nunique()}")

# Duplicate timestamp check
dupes = df["Date"].duplicated().sum()
if dupes > 0:
    raise ValueError(f"Dataset has {dupes} duplicate timestamps.")
print(f"  Duplicate timestamps: 0 ✓")

# Temporal continuity check
expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq="h")
missing_hours = expected.difference(df["Date"])
if len(missing_hours) > 0:
    print(f"  WARNING: {len(missing_hours)} missing hours — idx+h may not equal t+h.")
else:
    print(f"  Temporal continuity: OK ✓")

# Required columns
required = DELTA_COLS + ASSETS + ["Ann_CPI", "Ann_FOMC", "Ann_Employment"]
missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")
print(f"  Required columns present ✓")

# Check availability of LightGBM cross-asset features
for ctrl in LGBM_MACRO_CONTROLS:
    if ctrl in df.columns:
        n = df[ctrl].notna().sum()
        print(f"  LGBM macro control {ctrl}_L1: {n} non-NaN base ({n/len(df):.1%})")
    else:
        print(f"  WARNING: {ctrl} not in dataset — will be excluded from LightGBM")
LGBM_MACRO_CONTROLS = [c for c in LGBM_MACRO_CONTROLS if c in df.columns]

sample_years = (df["Date"].max() - df["Date"].min()).days / 365.25
print(f"  Sample length: {sample_years:.2f} years")

pd.DataFrame({
    "column":   df.columns,
    "dtype":    df.dtypes.values,
    "n_nonnan": df.notna().sum().values,
}).to_csv(f"{RESULTS_DIR}/00_column_inventory.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 1 — Feature Engineering")

# Calendar
df["hour_utc"]    = df["Date"].dt.hour
df["dow_utc"]     = df["Date"].dt.weekday
df["is_us_open"]  = df["hour_utc"].between(14, 21).astype(int)
df["is_lon_open"] = df["hour_utc"].between(8,  17).astype(int)
df["sin_hour"]    = np.sin(2 * np.pi * df["hour_utc"] / 24)
df["cos_hour"]    = np.cos(2 * np.pi * df["hour_utc"] / 24)

# PM signal rolling stats (shock detection + LightGBM features)
for col in DELTA_COLS:
    miss = MISS_MAP[col]
    s = df[col].copy()
    if miss in df.columns:
        s[df[miss] == 1] = np.nan
    df[f"{col}_roll_mu"]  = s.shift(1).rolling(ROLLING_W, min_periods=10).mean()
    df[f"{col}_roll_sig"] = s.shift(1).rolling(ROLLING_W, min_periods=10).std()
    df[f"{col}_z"]        = (s - df[f"{col}_roll_mu"]) / (df[f"{col}_roll_sig"] + 1e-8)

# Asset rolling volatility (lagged)
for col in ASSETS:
    if col not in df.columns:
        continue
    df[f"{col}_vol20"] = df[col].shift(1).rolling(20, min_periods=10).std()

# Cross-asset signals for LightGBM — lagged 1 hour (t-1)
# Lagged to avoid look-ahead and respect semi-strong EMH.
# Used only as LightGBM features, NOT in OLS regressions.
for ctrl in LGBM_MACRO_CONTROLS:
    df[f"{ctrl}_L1"] = df[ctrl].shift(1)

# Only include controls that were actually constructed above
lgbm_ctrl_cols = [f"{c}_L1" for c in LGBM_MACRO_CONTROLS if f"{c}_L1" in df.columns]

print("  Rolling stats computed ✓")
print("  Calendar features computed ✓")
print("  Asset volatility computed ✓")
print(f"  LightGBM cross-asset features (lagged): {lgbm_ctrl_cols}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — SHOCK PANEL (Section 3B)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 2 — Shock Panel (Section 3B)")

# ── 2A. Shock detection ──────────────────────────────────────────────────────
sep("2A: Identifying Shock Hours")
print(f"\n  Threshold: |ΔPM| > rolling_mean + {SHOCK_SIGMA}σ  (window={ROLLING_W}h)\n")

for col in DELTA_COLS:
    mu  = df[f"{col}_roll_mu"]
    sig = df[f"{col}_roll_sig"]
    s   = df[col]
    # FIX: mu (not mu.abs()) — rolling mean is negative ~47% of the time.
    # Using mu.abs() inflated threshold when mu<0, losing 13-18% of shocks.
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

# ── current_signals.csv — real-time state for dashboard ──────────────────────
# Generated here because it needs both rolling stats (Step 1) and
# is_shock flag (Step 2A). Updated every hour when pipeline runs.
last = df.iloc[-1]
current_rows = []
for col in DELTA_COLS:
    current_rows.append({
        "signal":     col,
        "timestamp":  last["Date"].strftime("%Y-%m-%dT%H:%M:%S"),  # UTC, no tz offset
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

# ── 2B. Build shock panel ─────────────────────────────────────────────────────
sep("2B: Building Shock Panel")


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

        # Copy lagged macro features for LightGBM (not used in OLS)
        for ctrl in lgbm_ctrl_cols:
            row[ctrl] = df.loc[idx, ctrl] if ctrl in df.columns else np.nan

        # Asset returns at each horizon
        for asset in ASSETS:
            if asset not in df.columns:
                continue
            for h in HORIZONS:
                future_idx = idx + h
                row[f"{asset}_h{h}"] = (
                    df.loc[future_idx, asset]
                    if future_idx < len(df) else np.nan
                )
            row[f"{asset}_vol20"] = (
                df.loc[idx, f"{asset}_vol20"]
                if f"{asset}_vol20" in df.columns else np.nan
            )

        panel_rows.append(row)

panel = pd.DataFrame(panel_rows)
panel = panel.sort_values("shock_hour").reset_index(drop=True)

print(f"\n  Total shock events: {len(panel):,}")
print(panel["pm_signal"].value_counts().to_string())
print(f"\n  Announcement overlap: {panel['is_announcement'].sum()} of {len(panel)}")
# 2B shock panel not saved to disk

# ── 2C. Core regression with macro controls ──────────────────────────────────
sep("2C: Core Regression — ΔPM_shock → Asset Return (bivariate, HC3)")
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

    for asset in ASSETS:
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

            # HC3 — returns beta and p-value of PM signal (index 1)
            beta, p_hc3, _ = ols_hc3(data[ret_col], X)

            # Pearson r between PM delta and return (bivariate, descriptive)
            r_corr = data["pm_delta"].corr(data[ret_col])

            # Directional t-test on PM signal
            dir_ret = np.sign(data["pm_delta"]) * data[ret_col]
            _, p_dir = ttest_1samp(dir_ret.dropna(), 0)

            reg_results.append({
                "pm_signal":    pm_col,
                "asset":        asset,
                "h":            h,
                "n":            len(data),
                "beta_pm":      beta,
                "p_hc3":        p_hc3,
                "r":            r_corr,
                "mean_dir":     dir_ret.mean(),
                "p_dir":        p_dir,
            })

res_df = pd.DataFrame(reg_results)
# Full regression results not saved — only significant rows go to results_regression.csv

sig = res_df[(res_df["p_hc3"] < 0.10) | (res_df["p_dir"] < 0.10)].sort_values("p_hc3")
print(f"  Significant results (p_HC3 < 0.10):\n")
print(f"  {'PM Signal':<22} {'Asset':<10} {'h':>2} {'n':>4} "
      f"{'beta_pm':>10} {'p_HC3':>8} {'r':>7} {'p_dir':>8}")
print(f"  {'─'*72}")
for _, row in sig.iterrows():
    print(f"  {row['pm_signal']:<22} {row['asset']:<10} {row['h']:>2} {row['n']:>4} "
          f"{row['beta_pm']:>10.6f} {row['p_hc3']:>8.4f}{stars(row['p_hc3']):<3} "
          f"{row['r']:>7.4f} {row['p_dir']:>8.4f}{stars(row['p_dir'])}")

# ── 2D. IRF — automatic lag selection ────────────────────────────────────────
sep("2D: IRF — coppie significative da 2C (automatico)")

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

        betas.append(b)
        ps.append(p)
        hs_valid.append(h)
        irf_results.append({
            "pm": pm_col, "asset": asset, "h": h,
            "beta_pm": b, "p_hc3": p
        })
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
# Full IRF not saved — only optimal lags go to results_irf.csv

# Build HOLDING_H automatically from IRF
sep("HOLDING_H — lag ottimale da IRF")
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

# IRF summary — optimal lag per pair
irf_optimal = []
for (pm, asset), h in HOLDING_H.items():
    row_irf = irf_df[(irf_df["pm"] == pm) &
                     (irf_df["asset"] == asset) &
                     (irf_df["h"] == h)]
    if len(row_irf) > 0:
        irf_optimal.append({
            "pm_signal":   pm,
            "asset":       asset,
            "optimal_h":   h,
            "beta_optimal": row_irf.iloc[0]["beta_pm"],
            "p_optimal":   row_irf.iloc[0]["p_hc3"],
        })
irf_optimal_df = pd.DataFrame(irf_optimal) if irf_optimal else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — LIGHTGBM (Section 3C)
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 3 — LightGBM on Shock Panel (Section 3C)")

# Features: PM signal features + calendar + session + announcements +
#           asset volatility + cross-asset signals (lagged)
# DXY_chg_L1 and US2Y_chg_L1 added as per PDF spec (page 22):
# "Cross-asset signals: lagged VIX change, DXY change, US2Y change"
LGBM_FEATURES_BASE = [
    "pm_delta", "pm_z", "pm_vol",
    "hour_utc", "dow_utc",
    "sin_hour", "cos_hour",
    "is_us_open", "is_lon_open",
    "Ann_CPI", "Ann_FOMC", "Ann_Employment", "Ann_GDP",
]
# Add lagged cross-asset signals for LightGBM only
LGBM_FEATURES_BASE += lgbm_ctrl_cols

lgbm_results = []
feat_imp_all  = []

for pm_col in DELTA_COLS:
    sub_pm = panel[panel["pm_signal"] == pm_col].copy().reset_index(drop=True)
    if len(sub_pm) < MIN_TRAIN + 50:
        continue

    for asset in ASSETS:
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

        X = sub[feats]
        y = sub[ret_col]
        n = len(sub)
        fs = n // N_FOLDS

        oos_preds  = np.full(n, np.nan)
        feat_imp   = np.zeros(len(feats))
        folds_done = 0

        for k in range(N_FOLDS):
            ts = k * fs
            te = min(ts + fs, n)
            tr = np.arange(0, max(0, ts - 1))
            if len(tr) < MIN_TRAIN:
                continue
            model = lgb.LGBMRegressor(**LGBM_PARAMS)
            model.fit(X.iloc[tr], y.iloc[tr], callbacks=[lgb.log_evaluation(-1)])
            oos_preds[ts:te] = model.predict(X.iloc[ts:te])
            feat_imp        += model.feature_importances_
            folds_done      += 1

        if folds_done > 0:
            feat_imp = feat_imp / folds_done

        mask = ~np.isnan(oos_preds)
        if mask.sum() < 20:
            continue

        y_oos  = y.values[mask]
        p_oos  = oos_preds[mask]
        da_oos = directional_accuracy(y_oos, p_oos)

        # Binomial test: LightGBM vs random walk (50%)
        n_correct = round(da_oos * mask.sum())
        bt        = binomtest(n_correct, mask.sum(), 0.5, alternative="greater")
        p_binom   = bt.pvalue

        # OLS baseline — bivariate, same sample as LightGBM
        # Estimated with same walk-forward so comparison with LGBM is fair
        sub_ols   = sub[["pm_delta", ret_col]].copy()
        ols_preds = np.full(len(sub_ols), np.nan)
        n_ols     = len(sub_ols)
        fs_ols    = n_ols // N_FOLDS

        for k in range(N_FOLDS):
            ts = k * fs_ols
            te = min(ts + fs_ols, n_ols)
            tr = np.arange(0, max(0, ts - 1))
            if len(tr) < 20:
                continue
            m_ols = OLS(
                sub_ols[ret_col].iloc[tr],
                add_constant(sub_ols["pm_delta"].iloc[tr])
            ).fit()
            ols_preds[ts:te] = m_ols.predict(
                add_constant(sub_ols["pm_delta"].iloc[ts:te])
            )

        mask_ols = ~np.isnan(ols_preds)
        da_ols   = directional_accuracy(
            sub_ols[ret_col].values[mask_ols],
            ols_preds[mask_ols]
        )

        # McNemar test: LightGBM vs OLS (formal superiority test)
        # Only on observations where both models have predictions
        mask_both = mask & mask_ols
        p_mcnemar = mcnemar_test(
            y.values[mask_both],
            oos_preds[mask_both],
            ols_preds[mask_both]
        ) if mask_both.sum() >= 5 else -1.0

        lgbm_beats_ols    = int(da_oos > da_ols)   # 1/0 — cleaner than True/False in CSV
        lgbm_beats_random = int(p_binom < 0.05)

        lgbm_results.append({
            "pm_signal":       pm_col,
            "asset":           asset,
            "h":               h,
            "n_panel":         len(sub),
            "n_oos":           mask.sum(),
            "DA_OOS_LGBM":     da_oos,
            "DA_OOS_OLS":      da_ols,
            "DA_random":       0.50,
            "p_binom_vs_random": p_binom,
            "p_mcnemar_vs_ols":  p_mcnemar,
            "lgbm_beats_ols":    lgbm_beats_ols,
            "lgbm_beats_random": lgbm_beats_random,
        })

        # Feature importance — top 5 for console, all saved
        imp_df = pd.DataFrame({
            "feature":    feats,
            "importance": feat_imp,
            "pm_signal":  pm_col,
            "asset":      asset,
            "h":          h,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1

        feat_imp_all.append(imp_df)

        # Feature importance accumulated — top 5 saved to results_feature_importance.csv

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
# LightGBM results saved to results_lgbm.csv

feat_imp_all_df = pd.concat(feat_imp_all, ignore_index=True) if feat_imp_all else pd.DataFrame(columns=["pm_signal","asset","h","feature","importance","rank"])

if len(lgbm_df) > 0:
    print(f"\n  Summary:\n")
    print(f"  {'PM Signal':<22} {'Asset':<10} {'h':>2} {'n_oos':>6} "
          f"{'DA_LGBM':>9} {'DA_OLS':>9} {'p_binom':>9} {'p_mcn':>8} {'beats_OLS':>10}")
    print(f"  {'─'*85}")
    for _, r in lgbm_df.sort_values("DA_OOS_LGBM", ascending=False).iterrows():
        mcn = f"{r['p_mcnemar_vs_ols']:.4f}" if r['p_mcnemar_vs_ols'] >= 0 else "  n/a"
        print(f"  {r['pm_signal']:<22} {r['asset']:<10} {r['h']:>2} "
              f"{r['n_oos']:>6} {r['DA_OOS_LGBM']:>9.2%} {r['DA_OOS_OLS']:>9.2%} "
              f"{r['p_binom_vs_random']:>9.4f}{stars(r['p_binom_vs_random']):<2} "
              f"{mcn:>8} {'✓' if r['lgbm_beats_ols'] else '✗':>10}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — OUTPUT FILES
# ─────────────────────────────────────────────────────────────────────────────

banner("STEP 4 — Output Files")

# 4 clean output CSVs — one per section, no sparse columns.
# Each file is self-contained: the dashboard reads whichever it needs.

# 1. Regression — significant rows (p_hc3 < 0.10)
reg_cols = ["pm_signal", "asset", "h", "n", "beta_pm", "p_hc3", "r", "mean_dir", "p_dir"]
if len(res_df) > 0:
    reg_out = res_df[res_df["p_hc3"] < 0.10][reg_cols].copy()
    reg_out.to_csv(f"{RESULTS_DIR}/results_regression.csv", index=False)
    print(f"  results_regression.csv : {len(reg_out)} significant rows")

# 2. IRF — optimal lag per pair
if len(irf_optimal_df) > 0:
    irf_optimal_df.to_csv(f"{RESULTS_DIR}/results_irf.csv", index=False)
    print(f"  results_irf.csv        : {len(irf_optimal_df)} optimal lags")

# 3. LightGBM metrics
if len(lgbm_df) > 0:
    lgbm_df.to_csv(f"{RESULTS_DIR}/results_lgbm.csv", index=False)
    print(f"  results_lgbm.csv       : {len(lgbm_df)} pairs")

# 4. Feature importance — top 5 per pair
if len(feat_imp_all_df) > 0:
    fimp_cols = ["pm_signal", "asset", "h", "rank", "feature", "importance"]
    fimp_out = (
        feat_imp_all_df
        .sort_values(["pm_signal", "asset", "importance"], ascending=[True, True, False])
        .groupby(["pm_signal", "asset"])
        .head(5)
        .reset_index(drop=True)
    )
    fimp_out["rank"] = fimp_out.groupby(["pm_signal", "asset"]).cumcount() + 1
    fimp_out[fimp_cols].to_csv(f"{RESULTS_DIR}/results_feature_importance.csv", index=False)
    print(f"  results_feature_importance.csv : {len(fimp_out)} rows (top 5 per pair)")


banner("PIPELINE COMPLETE")
print(f"""
  Results saved in: ./{RESULTS_DIR}/

  Files (update weekly except current_signals):
    current_signals.csv            → z-score, is_shock, is_missing per PM signal
    results_regression.csv         → significant OLS betas (p_hc3 < 0.10)
    results_irf.csv                → optimal lag per significant PM-asset pair
    results_lgbm.csv               → DA, McNemar, beats_ols, beats_random
    results_feature_importance.csv → top 5 features per pair
""")
