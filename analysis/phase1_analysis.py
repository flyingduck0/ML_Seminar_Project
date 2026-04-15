"""
=============================================================================
Phase 1: Descriptive Statistics & Unconditional Correlation Analysis
=============================================================================
"""

# ── Cell 1: Setup ───────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats as sp_stats
import os
import warnings
warnings.filterwarnings('ignore')

os.makedirs('outputs', exist_ok=True)

plt.rcParams.update({
    'figure.facecolor': '#fafafa', 'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#cccccc', 'axes.labelcolor': '#333333',
    'text.color': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555',
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'figure.dpi': 150,
})


# ── Cell 2: Load Data ──────────────────────────────────────────────────────
df = pd.read_csv('FEATURES_PREPARED.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

POLY_SIGNALS = ['FED_DELTA', 'GDP_DELTA', 'UNEMPLOYMENT_DELTA',
                'INF_YEARLY_DELTA', 'INF_MONTHLY_DELTA']
DUMMIES = {s: f'{s}_is_missing' for s in POLY_SIGNALS}

ASSET_RETURNS = ['SPY_chg', 'QQQ_chg', 'Gold_chg', 'Oil_chg', 'DXY_chg',
                 'BTC_chg', 'VIX_chg', 'SP500_fut_chg', 'US2Y_chg', 'US10Y_chg']

SIGNAL_LABELS = {
    'FED_DELTA': 'Fed Rate', 'GDP_DELTA': 'GDP Growth',
    'UNEMPLOYMENT_DELTA': 'Unemployment', 'INF_YEARLY_DELTA': 'Yearly Inflation',
    'INF_MONTHLY_DELTA': 'Monthly Inflation',
}
ASSET_LABELS = {
    'SPY_chg': 'SPY', 'QQQ_chg': 'QQQ', 'Gold_chg': 'Gold',
    'Oil_chg': 'Oil', 'DXY_chg': 'USD (DXY)', 'BTC_chg': 'Bitcoin',
    'VIX_chg': 'VIX', 'SP500_fut_chg': 'S&P Fut.',
    'US2Y_chg': '2Y Yield', 'US10Y_chg': '10Y Yield',
}
SIG_COLORS = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a']

print("=" * 70)
print("PHASE 1: DESCRIPTIVE STATISTICS & CORRELATION ANALYSIS (REVISED)")
print("=" * 70)
print(f"Dataset: {len(df)} hourly observations")
print(f"Period:  {df['Date'].min()} to {df['Date'].max()}")


# =============================================================================
# STEP 1: SUMMARY STATISTICS
# =============================================================================
print("\n── Step 1: Summary Statistics ──")

rows = []
for sig in POLY_SIGNALS:
    real = df[df[DUMMIES[sig]] == 0][sig]
    rows.append({
        'Variable': SIGNAL_LABELS[sig], 'Type': 'Polymarket',
        'N': len(real), '% liquid': round(len(real) / len(df) * 100, 1),
        'Mean': real.mean(), 'Std': real.std(),
        'Min': real.min(), 'Max': real.max(),
        'Skew': real.skew(), 'Kurt': real.kurtosis(),
        '% near zero (|x|<0.001)': round((real.abs() < 0.001).mean() * 100, 1),
    })

for asset in ASSET_RETURNS:
    real = df[asset].dropna()
    rows.append({
        'Variable': ASSET_LABELS[asset], 'Type': 'Asset',
        'N': len(real), '% liquid': round(len(real) / len(df) * 100, 1),
        'Mean': real.mean(), 'Std': real.std(),
        'Min': real.min(), 'Max': real.max(),
        'Skew': real.skew(), 'Kurt': real.kurtosis(),
        '% near zero (|x|<0.001)': round((real.abs() < 0.001).mean() * 100, 1),
    })

summary = pd.DataFrame(rows)
summary.to_csv('outputs/phase1_summary_statistics.csv', index=False)
print("  Saved: phase1_summary_statistics.csv")

# Print with interpretation
print("\n  Key observation: signal concentration")
for sig in POLY_SIGNALS:
    real = df[df[DUMMIES[sig]] == 0][sig]
    near_zero = (real.abs() < 0.001).mean() * 100
    print(f"    {SIGNAL_LABELS[sig]:20s}: {near_zero:.0f}% of liquid hours have |change| < 0.001")
print("  → Most hours contain near-zero signal. Information is concentrated")
print("    in rare large moves. This is a key challenge for prediction.")


# =============================================================================
# STEP 2: MISSINGNESS HEATMAP
# =============================================================================
print("\n── Step 2: Missingness Heatmap ──")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

hour_avail = pd.DataFrame()
for sig in POLY_SIGNALS:
    hour_avail[SIGNAL_LABELS[sig]] = df.groupby('hour_utc').apply(
        lambda g: (g[DUMMIES[sig]] == 0).mean())
for asset in ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg']:
    hour_avail[ASSET_LABELS[asset]] = df.groupby('hour_utc').apply(
        lambda g: g[asset].notna().mean())

cmap = mpl.colors.LinearSegmentedColormap.from_list('avail', ['#d73027', '#fee08b', '#1a9850'])
im1 = ax1.imshow(hour_avail.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(24)); ax1.set_xticklabels(range(24), fontsize=7)
ax1.set_yticks(range(len(hour_avail.columns)))
ax1.set_yticklabels(hour_avail.columns, fontsize=8)
ax1.set_xlabel('Hour (UTC)'); ax1.set_title('Data Availability by Hour of Day')
plt.colorbar(im1, ax=ax1, shrink=0.8, label='% available')

dow_avail = pd.DataFrame()
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for sig in POLY_SIGNALS:
    dow_avail[SIGNAL_LABELS[sig]] = df.groupby('dow_utc').apply(
        lambda g: (g[DUMMIES[sig]] == 0).mean())
for asset in ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg']:
    dow_avail[ASSET_LABELS[asset]] = df.groupby('dow_utc').apply(
        lambda g: g[asset].notna().mean())

im2 = ax2.imshow(dow_avail.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(7)); ax2.set_xticklabels(dow_labels, fontsize=8)
ax2.set_yticks(range(len(dow_avail.columns)))
ax2.set_yticklabels(dow_avail.columns, fontsize=8)
ax2.set_xlabel('Day of Week'); ax2.set_title('Data Availability by Day of Week')
plt.colorbar(im2, ax=ax2, shrink=0.8, label='% available')

for ax, data in [(ax1, hour_avail.T.values), (ax2, dow_avail.T.values)]:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = 'white' if data[i, j] < 0.5 else '#333333'
            ax.text(j, i, f'{data[i,j]:.0%}', ha='center', va='center',
                    fontsize=5 if ax == ax1 else 7, color=color)

fig.suptitle('Data Availability: Polymarket Signals & Asset Returns\n'
             '(Polymarket trades 24/7; SPY only ~6.5h/day → different lagging strategies needed)',
             fontsize=12, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('outputs/phase1_missingness_heatmap.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_missingness_heatmap.png")


# =============================================================================
# STEP 3: TIME SERIES PLOTS
# =============================================================================
print("\n── Step 3: Time Series Plots ──")

ann_map = {'FED_DELTA': 'Ann_FOMC', 'GDP_DELTA': 'Ann_GDP',
           'UNEMPLOYMENT_DELTA': 'Ann_Employment',
           'INF_YEARLY_DELTA': 'Ann_CPI', 'INF_MONTHLY_DELTA': 'Ann_CPI'}

fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
for i, (sig, color) in enumerate(zip(POLY_SIGNALS, SIG_COLORS)):
    ax = axes[i]
    real_mask = df[DUMMIES[sig]] == 0
    daily = df[real_mask].set_index('Date')[sig].resample('D').sum()
    ax.fill_between(daily.index, 0, daily.values,
                    where=daily.values > 0, color=color, alpha=0.3)
    ax.fill_between(daily.index, 0, daily.values,
                    where=daily.values < 0, color=color, alpha=0.15)
    ax.plot(daily.rolling(7).mean().index, daily.rolling(7).mean().values,
            color=color, linewidth=1.5, label='7-day MA')
    ax.axhline(0, color='#cccccc', linewidth=0.8)
    ax.set_ylabel(SIGNAL_LABELS[sig], fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.text(0.01, 0.95, f'{real_mask.mean()*100:.0f}% liquid',
            transform=ax.transAxes, fontsize=7, va='top', color='#888888')
    ann_col = ann_map.get(sig)
    if ann_col:
        for ad in df[df[ann_col] == 1]['Date']:
            ax.axvline(ad, color='#333333', linewidth=0.5, alpha=0.3, linestyle=':')

axes[-1].set_xlabel('Date')
fig.suptitle('Polymarket Signal Changes Over Time (Daily Sum of Liquid Obs)\n'
             '(Dotted lines = related macro announcements)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/phase1_timeseries.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_timeseries.png")


# =============================================================================
# STEP 4: DISTRIBUTION PLOTS
# =============================================================================
print("\n── Step 4: Distribution Plots ──")

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

for i, (sig, color) in enumerate(zip(POLY_SIGNALS, SIG_COLORS)):
    ax = axes[0, i]
    real = df[df[DUMMIES[sig]] == 0][sig]
    lo, hi = real.quantile(0.01), real.quantile(0.99)
    clipped = real[(real >= lo) & (real <= hi)]
    ax.hist(clipped, bins=60, color=color, alpha=0.7, edgecolor='white',
            linewidth=0.3, density=True)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.axvline(real.mean(), color='#e31a1c', linewidth=1, linestyle='--',
               label=f'μ={real.mean():.4f}')
    ax.set_title(SIGNAL_LABELS[sig], fontsize=9, fontweight='bold')
    ax.legend(fontsize=6)
    ax.text(0.95, 0.95,
            f'n={len(real)}\nσ={real.std():.4f}\nskew={real.skew():.2f}\nkurt={real.kurtosis():.1f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='#cccccc'))

key_assets = ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg', 'VIX_chg']
asset_colors = ['#a6cee3', '#ff7f00', '#6a3d9a', '#1f78b4', '#e31a1c']
for i, (asset, color) in enumerate(zip(key_assets, asset_colors)):
    ax = axes[1, i]
    real = df[asset].dropna()
    lo, hi = real.quantile(0.01), real.quantile(0.99)
    clipped = real[(real >= lo) & (real <= hi)]
    ax.hist(clipped, bins=60, color=color, alpha=0.7, edgecolor='white',
            linewidth=0.3, density=True)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_title(ASSET_LABELS[asset], fontsize=9, fontweight='bold')
    ax.text(0.95, 0.95,
            f'n={len(real)}\nσ={real.std():.4f}\nskew={real.skew():.2f}\nkurt={real.kurtosis():.1f}',
            transform=ax.transAxes, fontsize=6, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='#cccccc'))

axes[0, 0].set_ylabel('Density (Polymarket)', fontsize=9)
axes[1, 0].set_ylabel('Density (Assets)', fontsize=9)
fig.suptitle('Distributions: Polymarket Signals (top) & Asset Returns (bottom)\n'
             '(1st–99th percentile; Polymarket filtered for liquid obs only)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_distributions.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_distributions.png")


# =============================================================================
# STEP 5: CORRELATION ANALYSIS — with proper interpretation
# =============================================================================
print("\n── Step 5: Correlation Analysis ──")
print("  NOTE: Contemporaneous = co-movement (NOT prediction).")
print("  Both Polymarket and assets likely react to the same news simultaneously.")
print("  Lagged = potential prediction, but limited to consecutive trading hours.")

def compute_filtered_corr(df, signals, dummies, assets, lag=0):
    results = []
    for sig in signals:
        dummy = dummies[sig]
        sig_vals = df[sig].shift(lag) if lag != 0 else df[sig]
        dum_vals = df[dummy].shift(lag) if lag != 0 else df[dummy]
        for asset in assets:
            mask = df[asset].notna() & (dum_vals == 0) & sig_vals.notna()
            n = mask.sum()
            if n > 30:
                r, p = sp_stats.pearsonr(sig_vals[mask], df.loc[mask, asset])
                results.append({'signal': sig, 'asset': asset,
                                'r': r, 'p': p, 'n': n, 'significant': p < 0.05})
    return pd.DataFrame(results)


def plot_corr_heatmap(corr_df, title, subtitle, filename):
    pivot_r = corr_df.pivot(index='signal', columns='asset', values='r')
    pivot_p = corr_df.pivot(index='signal', columns='asset', values='p')
    pivot_r = pivot_r.reindex(index=POLY_SIGNALS, columns=ASSET_RETURNS)
    pivot_p = pivot_p.reindex(index=POLY_SIGNALS, columns=ASSET_RETURNS)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'custom', ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'])
    valid = pivot_r.values[~np.isnan(pivot_r.values)]
    vmax = min(max(abs(valid.min()), abs(valid.max())), 0.15)

    im = ax.imshow(pivot_r.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(len(ASSET_RETURNS)))
    ax.set_xticklabels([ASSET_LABELS[a] for a in ASSET_RETURNS], rotation=45, ha='right')
    ax.set_yticks(range(len(POLY_SIGNALS)))
    ax.set_yticklabels([SIGNAL_LABELS[s] for s in POLY_SIGNALS])

    for i in range(len(POLY_SIGNALS)):
        for j in range(len(ASSET_RETURNS)):
            r_val = pivot_r.values[i, j]
            p_val = pivot_p.values[i, j]
            if np.isnan(r_val): continue
            star = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else ''
            color = 'white' if abs(r_val) > vmax * 0.6 else '#333333'
            ax.text(j, i, f'{r_val:.3f}{star}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold' if star else 'normal')

    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(title, pad=15)
    fig.text(0.5, 0.01, subtitle, fontsize=7, color='#888888', style='italic', ha='center')
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', bbox_inches='tight')
    plt.show()


# Contemporaneous — clearly labelled as co-movement
contemp = compute_filtered_corr(df, POLY_SIGNALS, DUMMIES, ASSET_RETURNS, lag=0)
plot_corr_heatmap(contemp,
    'Contemporaneous Correlations: Polymarket Signals vs Asset Returns',
    'CAUTION: These reflect co-movement (simultaneous reaction to news), NOT predictive power.  '
    '* p<0.10  ** p<0.05  *** p<0.01  |  Artificial zeros excluded.',
    'phase1_correlation_contemporaneous.png')
print("  Saved: phase1_correlation_contemporaneous.png")

# Lagged — with caveat about equity coverage
lagged = compute_filtered_corr(df, POLY_SIGNALS, DUMMIES, ASSET_RETURNS, lag=1)
plot_corr_heatmap(lagged,
    'Lagged Correlations: Polymarket Signal (t) → Asset Return (t+1)',
    'NOTE: For SPY/QQQ, this only captures consecutive trading hours (~20% of data). '
    'Overnight/weekend signals are excluded. Phase 2 addresses this.  '
    '* p<0.10  ** p<0.05  *** p<0.01',
    'phase1_correlation_lagged.png')
print("  Saved: phase1_correlation_lagged.png")

# Print significant results
sig_c = contemp[contemp['significant']]
sig_l = lagged[lagged['significant']]

print(f"\n  CONTEMPORANEOUS (co-movement): {len(sig_c)}/{len(contemp)} pairs significant:")
for _, row in sig_c.iterrows():
    print(f"    {SIGNAL_LABELS[row['signal']]:20s} ↔ {ASSET_LABELS[row['asset']]:10s}: "
          f"r={row['r']:+.4f} (p={row['p']:.4f}, n={row['n']})")

print(f"\n  LAGGED (potential prediction): {len(sig_l)}/{len(lagged)} pairs significant:")
for _, row in sig_l.iterrows():
    print(f"    {SIGNAL_LABELS[row['signal']]:20s} → {ASSET_LABELS[row['asset']]:10s}: "
          f"r={row['r']:+.4f} (p={row['p']:.4f}, n={row['n']})")


# =============================================================================
# STEP 6: OUTLIER SENSITIVITY ANALYSIS (NEW)
# =============================================================================
print("\n── Step 6: Outlier Sensitivity Analysis ──")
print("  Testing: do the correlations survive when we remove extreme observations?")
print("  If Pearson drops sharply after winsorisation, the signal is fragile.")

from scipy.stats import mstats

sensitivity_rows = []
for sig in POLY_SIGNALS:
    dummy = DUMMIES[sig]
    for asset in ['SPY_chg', 'QQQ_chg', 'SP500_fut_chg', 'Gold_chg', 'BTC_chg']:
        mask = (df[dummy] == 0) & df[asset].notna()
        if mask.sum() < 50: continue

        x_raw = df.loc[mask, sig].values
        y_raw = df.loc[mask, asset].values

        # Raw Pearson
        r_raw, p_raw = sp_stats.pearsonr(x_raw, y_raw)

        # Winsorised at 1st/99th percentile
        x_wins = mstats.winsorize(x_raw, limits=[0.01, 0.01])
        y_wins = mstats.winsorize(y_raw, limits=[0.01, 0.01])
        r_wins, p_wins = sp_stats.pearsonr(x_wins, y_wins)

        # Spearman (rank-based, outlier-resistant)
        r_spear, p_spear = sp_stats.spearmanr(x_raw, y_raw)

        sensitivity_rows.append({
            'Signal': SIGNAL_LABELS[sig], 'Asset': ASSET_LABELS[asset],
            'r (raw)': round(r_raw, 4), 'r (winsorised)': round(r_wins, 4),
            'r (Spearman)': round(r_spear, 4),
            'Drop after winsorising': round(abs(r_raw) - abs(r_wins), 4),
            'p (raw)': round(p_raw, 4), 'n': mask.sum(),
        })

sens_df = pd.DataFrame(sensitivity_rows)

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

ax.scatter(sens_df['r (winsorised)'], sens_df['r (raw)'],
           c=['#b2182b' if d > 0.01 else '#999999' for d in sens_df['Drop after winsorising']],
           s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

lim = max(abs(sens_df['r (raw)']).max(), abs(sens_df['r (winsorised)']).max()) * 1.3
ax.plot([-lim, lim], [-lim, lim], '--', color='#cccccc', linewidth=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Pearson r (winsorised at 1st/99th percentile)')
ax.set_ylabel('Pearson r (raw)')
ax.set_title('Outlier Sensitivity: Do Correlations Survive Winsorisation?')

for _, row in sens_df[sens_df['Drop after winsorising'] > 0.015].iterrows():
    label = f"{row['Signal']} → {row['Asset']}"
    ax.annotate(label, (row['r (winsorised)'], row['r (raw)']),
                fontsize=6, color='#555555', textcoords='offset points', xytext=(5, 5))

ax.text(0.02, 0.98,
        'Red = drops significantly after winsorising\n'
        '→ Correlation is driven by a few extreme hours\n'
        '→ Signal is fragile, not a stable linear relationship',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig('outputs/phase1_outlier_sensitivity.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_outlier_sensitivity.png")

# Print the pairs where winsorisation matters most
fragile = sens_df[sens_df['Drop after winsorising'] > 0.01].sort_values(
    'Drop after winsorising', ascending=False)
if len(fragile) > 0:
    print(f"\n  FRAGILE correlations (drop > 0.01 after winsorising):")
    for _, row in fragile.iterrows():
        print(f"    {row['Signal']:20s} → {row['Asset']:10s}: "
              f"raw r={row['r (raw)']:+.4f}, winsorised r={row['r (winsorised)']:+.4f}, "
              f"Spearman r={row['r (Spearman)']:+.4f}")
else:
    print("  No correlations drop significantly after winsorisation.")


# =============================================================================
# STEP 7: SPEARMAN vs PEARSON — with BOTH interpretations
# =============================================================================
print("\n── Step 7: Spearman vs Pearson ──")

nonlin_results = []
for sig in POLY_SIGNALS:
    for asset in ASSET_RETURNS:
        mask = (df[DUMMIES[sig]] == 0) & df[asset].notna()
        if mask.sum() > 30:
            x, y = df.loc[mask, sig], df.loc[mask, asset]
            r_p, _ = sp_stats.pearsonr(x, y)
            r_s, _ = sp_stats.spearmanr(x, y)
            nonlin_results.append({'signal': sig, 'asset': asset,
                                   'pearson_r': r_p, 'spearman_r': r_s,
                                   'abs_diff': abs(r_p) - abs(r_s)})

nonlin_df = pd.DataFrame(nonlin_results)

fig, ax = plt.subplots(figsize=(7, 7))
gap = nonlin_df['abs_diff'].values
colors = ['#b2182b' if g > 0.015 else '#2166ac' if g < -0.015 else '#999999' for g in gap]
ax.scatter(nonlin_df['spearman_r'], nonlin_df['pearson_r'],
           c=colors, s=50, alpha=0.7, edgecolors='white', linewidths=0.5)

lim = max(abs(nonlin_df['pearson_r']).max(), abs(nonlin_df['spearman_r']).max()) * 1.3
ax.plot([-lim, lim], [-lim, lim], '--', color='#cccccc', linewidth=1)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel('Spearman ρ (rank correlation)')
ax.set_ylabel('Pearson r (linear correlation)')
ax.set_title('Pearson vs Spearman: Non-Linearity or Fragility?')

for _, row in nonlin_df[nonlin_df['abs_diff'].abs() > 0.02].iterrows():
    label = f"{SIGNAL_LABELS[row['signal']]} → {ASSET_LABELS[row['asset']]}"
    ax.annotate(label, (row['spearman_r'], row['pearson_r']),
                fontsize=6, color='#555555', textcoords='offset points', xytext=(5, 5))

ax.text(0.02, 0.98,
        'Red dots: |Pearson| >> |Spearman|\n'
        'Two possible interpretations:\n'
        '  1. Non-linear relationship → tree-based ML may help\n'
        '  2. Fragile, outlier-driven → signal may not generalise\n'
        'Step 6 (outlier sensitivity) helps distinguish these.',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc'))

plt.tight_layout()
plt.savefig('outputs/phase1_spearman_vs_pearson.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_spearman_vs_pearson.png")


# =============================================================================
# STEP 8: QUINTILE ANALYSIS — with sample size warnings
# =============================================================================
print("\n── Step 8: Quintile Analysis ──")

key_pairs = [
    ('FED_DELTA', 'SPY_chg', 'Fed Rate → SPY'),
    ('FED_DELTA', 'SP500_fut_chg', 'Fed Rate → S&P Futures'),
    ('UNEMPLOYMENT_DELTA', 'SP500_fut_chg', 'Unemployment → S&P Futures'),
    ('INF_YEARLY_DELTA', 'SP500_fut_chg', 'Yearly Inflation → S&P Futures'),
    ('INF_MONTHLY_DELTA', 'BTC_chg', 'Monthly Inflation → Bitcoin'),
    ('INF_YEARLY_DELTA', 'SPY_chg', 'Yearly Inflation → SPY'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
q_labels = ['Q1\n(large drop)', 'Q2', 'Q3', 'Q4', 'Q5\n(large rise)']
q_colors = ['#2166ac', '#67a9cf', '#999999', '#ef8a62', '#b2182b']

for idx, (sig, asset, title) in enumerate(key_pairs):
    ax = axes[idx]
    mask = (df[DUMMIES[sig]] == 0) & df[asset].notna() & (df[sig] != 0)
    pair = df.loc[mask, [sig, asset]].copy()

    if len(pair) < 50:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', color='#999999')
        ax.set_title(title, fontsize=9, fontweight='bold')
        continue

    pair['quintile'] = pd.qcut(pair[sig], 5, labels=False, duplicates='drop')
    means = pair.groupby('quintile')[asset].mean()
    sems = pair.groupby('quintile')[asset].sem()
    counts = pair.groupby('quintile')[asset].count()
    n_q = len(means)
    per_bin = int(counts.mean())

    bars = ax.bar(range(n_q), means.values * 10000, yerr=sems.values * 10000 * 1.96,
                  color=q_colors[:n_q], edgecolor='white', linewidth=0.5,
                  capsize=3, error_kw={'linewidth': 0.8, 'color': '#888888'})
    ax.set_xticks(range(n_q))
    ax.set_xticklabels(q_labels[:n_q], fontsize=7)
    ax.set_ylabel('Mean return (bps)', fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.axhline(0, color='#cccccc', linewidth=0.8)

    spread = (means.iloc[-1] - means.iloc[0]) * 10000
    ax.text(0.95, 0.95, f'Q5−Q1: {spread:+.2f} bps', transform=ax.transAxes,
            fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='#cccccc'))

    # Flag if bins are small
    warn = ' ⚠️ small bins' if per_bin < 200 else ''
    ax.text(0.95, 0.05, f'n={int(counts.sum())}, ~{per_bin}/bin{warn}',
            transform=ax.transAxes, fontsize=6, ha='right', va='bottom', color='#999999')

fig.suptitle('Quintile Analysis: Asset Returns by Polymarket Signal Intensity\n'
             '(Liquid, non-zero obs only. Error bars = 95% CI. ⚠️ = small bin size)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_quintile_analysis.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_quintile_analysis.png")


# =============================================================================
# STEP 9: OVERNIGHT SIGNAL ACCUMULATION TEST (NEW)
# =============================================================================
print("\n── Step 9: Overnight Signal Accumulation ──")
print("  Testing: do Polymarket signals that accumulate overnight/weekend")
print("  predict equity returns at the next market open?")
print("  This is what Phase 2 should properly model (per the handover doc).")

# Identify SPY trading sessions
df['spy_available'] = df['SPY_chg'].notna().astype(int)
df['session_start'] = (df['spy_available'] == 1) & (df['spy_available'].shift(1).fillna(0) == 0)

# For each session start, accumulate the Polymarket signals from the prior close
session_starts = df[df['session_start']].index.tolist()

overnight_rows = []
for idx in session_starts:
    # Find previous session end
    prev_trading = df.loc[:idx-1]
    prev_trading = prev_trading[prev_trading['spy_available'] == 1]
    if len(prev_trading) == 0: continue
    prev_close_idx = prev_trading.index[-1]

    # Overnight window = prev_close+1 to session_start-1
    overnight = df.iloc[prev_close_idx+1:idx]
    if len(overnight) == 0: continue

    row = {'Date': df.loc[idx, 'Date'], 'overnight_hours': len(overnight)}
    # Accumulate signals (sum of changes overnight)
    for sig in POLY_SIGNALS:
        dummy = DUMMIES[sig]
        liquid = overnight[overnight[dummy] == 0]
        row[f'{sig}_overnight'] = liquid[sig].sum()
        row[f'{sig}_overnight_n'] = len(liquid)

    # First-hour SPY return at the open
    row['SPY_open_return'] = df.loc[idx, 'SPY_chg']
    # First 3 hours cumulative
    three_hour = df.iloc[idx:min(idx+3, len(df))]
    spy_3h = three_hour['SPY_chg'].dropna()
    row['SPY_3h_return'] = (1 + spy_3h).prod() - 1 if len(spy_3h) > 0 else np.nan

    overnight_rows.append(row)

overnight_df = pd.DataFrame(overnight_rows)
print(f"  Built {len(overnight_df)} overnight → open observations")

# Correlate accumulated overnight signals with next-day open
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
test_sigs = ['FED_DELTA', 'UNEMPLOYMENT_DELTA', 'INF_YEARLY_DELTA']

for i, sig in enumerate(test_sigs):
    ax = axes[i]
    col = f'{sig}_overnight'
    valid = overnight_df[[col, 'SPY_open_return']].dropna()
    valid = valid[valid[col] != 0]  # exclude nights with no signal

    if len(valid) < 20:
        ax.text(0.5, 0.5, f'Insufficient data\n(n={len(valid)})',
                transform=ax.transAxes, ha='center', va='center', color='#999999')
        ax.set_title(f'{SIGNAL_LABELS[sig]} overnight → SPY open', fontsize=9, fontweight='bold')
        continue

    x = valid[col].values
    y = valid['SPY_open_return'].values * 10000

    colors_scatter = ['#33a02c' if yi > 0 else '#e31a1c' for yi in y]
    ax.scatter(x, y, c=colors_scatter, s=30, alpha=0.6, edgecolors='white', linewidths=0.3)
    ax.axhline(0, color='#cccccc', linewidth=0.8)
    ax.axvline(0, color='#cccccc', linewidth=0.8)

    r, p = sp_stats.pearsonr(x, y)
    ax.set_xlabel(f'Accumulated {SIGNAL_LABELS[sig]} (overnight)', fontsize=8)
    ax.set_ylabel('SPY first-hour return (bps)', fontsize=8)
    ax.set_title(f'{SIGNAL_LABELS[sig]} overnight → SPY open', fontsize=9, fontweight='bold')
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}\nn = {len(valid)}',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='#cccccc'))

fig.suptitle('Overnight Signal Accumulation Test\n'
             '(Do Polymarket signals between market close → next open predict SPY?)',
             fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('outputs/phase1_overnight_accumulation.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_overnight_accumulation.png")

# Clean up temp columns
df.drop(columns=['spy_available', 'session_start'], inplace=True)


# =============================================================================
# STEP 10: FULL CORRELATION SUMMARY TABLE
# =============================================================================
print("\n── Step 10: Correlation Summary Table ──")

summary_rows = []
for sig in POLY_SIGNALS:
    dummy = DUMMIES[sig]
    for asset in ASSET_RETURNS:
        mask_c = (df[dummy] == 0) & df[asset].notna()
        if mask_c.sum() <= 30: continue

        x_c, y_c = df.loc[mask_c, sig], df.loc[mask_c, asset]
        r_p, p_p = sp_stats.pearsonr(x_c, y_c)
        r_s, _ = sp_stats.spearmanr(x_c, y_c)

        # Winsorised
        x_w = mstats.winsorize(x_c.values, limits=[0.01, 0.01])
        y_w = mstats.winsorize(y_c.values, limits=[0.01, 0.01])
        r_w, _ = sp_stats.pearsonr(x_w, y_w)

        # Lagged
        sig_lag = df[sig].shift(1)
        dum_lag = df[dummy].shift(1)
        mask_l = (dum_lag == 0) & df[asset].notna() & sig_lag.notna()
        r_lag, p_lag = (sp_stats.pearsonr(sig_lag[mask_l], df.loc[mask_l, asset])
                        if mask_l.sum() > 30 else (np.nan, np.nan))

        summary_rows.append({
            'Signal': SIGNAL_LABELS[sig], 'Asset': ASSET_LABELS[asset],
            'r (raw)': round(r_p, 4), 'p (raw)': round(p_p, 4),
            'r (winsorised)': round(r_w, 4),
            'r (Spearman)': round(r_s, 4),
            'r (lagged)': round(r_lag, 4), 'p (lagged)': round(p_lag, 4),
            'N': mask_c.sum(),
            'Fragile?': 'Yes' if abs(r_p) - abs(r_w) > 0.01 else 'No',
        })

corr_summary = pd.DataFrame(summary_rows)
corr_summary.to_csv('outputs/phase1_correlation_summary.csv', index=False)
print("  Saved: phase1_correlation_summary.csv")


# =============================================================================
# STEP 11: SIGNAL CONCENTRATION ON NEWS DAYS (inspired by Diercks et al. 2025)
# =============================================================================
print("\n── Step 11: Signal Concentration on News Days ──")
print("  Testing: are Polymarket signals significantly larger on announcement")
print("  days? (cf. Diercks et al. 2025, Figure 16)")

ann_signal_pairs = [
    ('Ann_FOMC', 'FED_DELTA', 'FOMC', 'Fed Rate'),
    ('Ann_CPI', 'INF_MONTHLY_DELTA', 'CPI', 'Monthly Infl.'),
    ('Ann_CPI', 'INF_YEARLY_DELTA', 'CPI', 'Yearly Infl.'),
    ('Ann_Employment', 'UNEMPLOYMENT_DELTA', 'Employment', 'Unemployment'),
    ('Ann_GDP', 'GDP_DELTA', 'GDP', 'GDP Growth'),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))

concentration_results = []
for idx, (ann, sig, ann_label, sig_label) in enumerate(ann_signal_pairs):
    ax = axes[idx]
    dummy = DUMMIES[sig]
    mask_liquid = df[dummy] == 0

    news_vals = df[mask_liquid & (df[ann] == 1)][sig].abs()
    other_vals = df[mask_liquid & (df[ann] == 0)][sig].abs()

    if len(news_vals) < 3:
        ax.set_visible(False)
        continue

    ratio = news_vals.mean() / other_vals.mean() if other_vals.mean() > 0 else np.nan
    stat, p = sp_stats.mannwhitneyu(news_vals, other_vals, alternative='greater')

    concentration_results.append({
        'Announcement': ann_label, 'Signal': sig_label,
        'Mean |Δ| news': round(news_vals.mean(), 6),
        'Mean |Δ| other': round(other_vals.mean(), 6),
        'Ratio': round(ratio, 1), 'p-value': round(p, 4),
        'n_news': len(news_vals),
    })

    # Bar chart: news vs other
    bars = ax.bar(['News\ndays', 'Other\ndays'],
                  [news_vals.mean() * 1000, other_vals.mean() * 1000],
                  color=['#e31a1c', '#a6cee3'], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean |signal change| (×1000)', fontsize=8)
    ax.set_title(f'{ann_label} → {sig_label}', fontsize=9, fontweight='bold')

    sig_marker = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'n.s.'
    ax.text(0.95, 0.95, f'{ratio:.1f}x larger\np={p:.3f} {sig_marker}\nn={len(news_vals)} events',
            transform=ax.transAxes, fontsize=7, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='#cccccc'))

fig.suptitle('Signal Concentration: Are Polymarket Signals Larger on Announcement Days?\n'
             '(Mann-Whitney U test, one-sided. cf. Diercks et al. 2025, Fig. 16)',
             fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('outputs/phase1_signal_concentration.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_signal_concentration.png")

conc_df = pd.DataFrame(concentration_results)
print("\n  Results:")
for _, row in conc_df.iterrows():
    print(f"    {row['Announcement']:12s} → {row['Signal']:15s}: "
          f"{row['Ratio']}x larger on news days (p={row['p-value']})")


# =============================================================================
# STEP 12: ASYMMETRIC CORRELATIONS (inspired by Diercks et al. 2025)
# =============================================================================
print("\n── Step 12: Asymmetric Correlations ──")
print("  Testing: do positive and negative signal changes have different")
print("  relationships with asset returns? (cf. Diercks et al. 2025, Fig. 17)")

asym_pairs = [
    ('FED_DELTA', 'SPY_chg', 'Fed Rate → SPY'),
    ('FED_DELTA', 'SP500_fut_chg', 'Fed Rate → S&P Fut.'),
    ('INF_YEARLY_DELTA', 'SPY_chg', 'Yearly Infl. → SPY'),
    ('INF_YEARLY_DELTA', 'SP500_fut_chg', 'Yearly Infl. → S&P Fut.'),
    ('UNEMPLOYMENT_DELTA', 'SPY_chg', 'Unemployment → SPY'),
    ('UNEMPLOYMENT_DELTA', 'SP500_fut_chg', 'Unemployment → S&P Fut.'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

asym_results = []
for idx, (sig, asset, title) in enumerate(asym_pairs):
    ax = axes[idx]
    dummy = DUMMIES[sig]
    mask = (df[dummy] == 0) & df[asset].notna()

    pos_mask = mask & (df[sig] > 0)
    neg_mask = mask & (df[sig] < 0)

    if pos_mask.sum() < 30 or neg_mask.sum() < 30:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', color='#999999')
        ax.set_title(title, fontsize=9, fontweight='bold')
        continue

    r_pos, p_pos = sp_stats.pearsonr(df.loc[pos_mask, sig], df.loc[pos_mask, asset])
    r_neg, p_neg = sp_stats.pearsonr(df.loc[neg_mask, sig], df.loc[neg_mask, asset])

    asym_results.append({
        'Pair': title,
        'r (positive Δ)': round(r_pos, 4), 'p (pos)': round(p_pos, 4), 'n_pos': pos_mask.sum(),
        'r (negative Δ)': round(r_neg, 4), 'p (neg)': round(p_neg, 4), 'n_neg': neg_mask.sum(),
        'Asymmetry': round(abs(r_pos) - abs(r_neg), 4),
    })

    # Scatter: positive in red, negative in blue
    ax.scatter(df.loc[pos_mask, sig], df.loc[pos_mask, asset] * 10000,
               c='#e31a1c', s=8, alpha=0.3, label=f'Δ>0: r={r_pos:+.3f}')
    ax.scatter(df.loc[neg_mask, sig], df.loc[neg_mask, asset] * 10000,
               c='#2166ac', s=8, alpha=0.3, label=f'Δ<0: r={r_neg:+.3f}')

    # Trend lines
    for m, c, col in [(pos_mask, '#e31a1c', sig), (neg_mask, '#2166ac', sig)]:
        x_vals = df.loc[m, col].values
        y_vals = df.loc[m, asset].values * 10000
        if len(x_vals) > 10:
            z = np.polyfit(x_vals, y_vals, 1)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), color=c, linewidth=1.5, alpha=0.8)

    ax.axhline(0, color='#cccccc', linewidth=0.5)
    ax.axvline(0, color='#cccccc', linewidth=0.5)
    ax.set_xlabel(f'{SIGNAL_LABELS[sig]} change', fontsize=7)
    ax.set_ylabel('Asset return (bps)', fontsize=7)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='best')

    # Flag asymmetry
    if abs(r_pos) > abs(r_neg) * 2 or abs(r_neg) > abs(r_pos) * 2:
        ax.text(0.95, 0.05, 'ASYMMETRIC', transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom', color='#e31a1c', fontweight='bold')

fig.suptitle('Asymmetric Correlations: Do Positive and Negative Signal Changes\n'
             'Have Different Effects on Asset Returns? (cf. Diercks et al. 2025, Fig. 17)',
             fontsize=12, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('outputs/phase1_asymmetric_correlations.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_asymmetric_correlations.png")

asym_df = pd.DataFrame(asym_results)
print("\n  Key asymmetries:")
for _, row in asym_df.iterrows():
    flag = " ← ASYMMETRIC" if row['Asymmetry'] > 0.03 else ""
    print(f"    {row['Pair']:30s}: r_pos={row['r (positive Δ)']:+.4f} vs r_neg={row['r (negative Δ)']:+.4f}{flag}")


# =============================================================================
# STEP 13: ASSET VOLATILITY ON ANNOUNCEMENT DAYS
# =============================================================================
print("\n── Step 13: Asset Volatility on Announcement Days ──")
print("  Testing: are asset returns more volatile on macro announcement days?")
print("  (cf. Diercks et al. 2025: variance of rate distributions drops on news days)")

fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
ann_list = ['Ann_CPI', 'Ann_Employment', 'Ann_FOMC', 'Ann_GDP']
ann_names = ['CPI', 'Employment', 'FOMC', 'GDP']

vol_results = []
for idx, (ann, ann_name) in enumerate(zip(ann_list, ann_names)):
    ax = axes[idx]

    assets_to_test = ['SP500_fut_chg', 'Gold_chg', 'BTC_chg']
    asset_short = ['S&P Fut.', 'Gold', 'Bitcoin']
    bar_colors = ['#1f78b4', '#ff7f00', '#6a3d9a']

    news_vols = []
    other_vols = []
    for asset in assets_to_test:
        news = df[df[ann] == 1][asset].dropna().abs().mean() * 10000
        other = df[df[ann] == 0][asset].dropna().abs().mean() * 10000
        news_vols.append(news)
        other_vols.append(other)

        vol_results.append({
            'Announcement': ann_name, 'Asset': ASSET_LABELS[asset],
            'Vol news (bps)': round(news, 1), 'Vol other (bps)': round(other, 1),
            'Ratio': round(news / other, 2) if other > 0 else np.nan,
        })

    x = np.arange(len(assets_to_test))
    width = 0.35
    ax.bar(x - width/2, news_vols, width, label='News days', color=bar_colors, alpha=0.8,
           edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, other_vols, width, label='Other days', color=bar_colors, alpha=0.3,
           edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(asset_short, fontsize=8)
    ax.set_ylabel('Mean |return| (bps)', fontsize=8)
    ax.set_title(f'{ann_name} Days', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7)

fig.suptitle('Asset Return Volatility: Announcement Days vs Normal Days\n'
             '(Solid = news days, faded = all other days)',
             fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('outputs/phase1_announcement_volatility.png', bbox_inches='tight')
plt.show()
print("  Saved: phase1_announcement_volatility.png")
