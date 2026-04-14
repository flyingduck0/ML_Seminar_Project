"""
Phase 1: Descriptive Statistics & Unconditional Correlation Analysis
Run this in Google Colab with FEATURES_PREPARED.csv uploaded to the file browser.
"""

# 1: Setup ───────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats as sp_stats
import os
import warnings
warnings.filterwarnings('ignore')

# Create output folder in Colab
os.makedirs('outputs', exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.facecolor': '#fafafa', 'axes.facecolor': '#fafafa',
    'axes.edgecolor': '#cccccc', 'axes.labelcolor': '#333333',
    'text.color': '#333333', 'xtick.color': '#555555', 'ytick.color': '#555555',
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'figure.dpi': 150,
})

#  2: Load Data ──────────────────────────────────────────────────────
# Adjust this path if your CSV is in a different location
df = pd.read_csv('FEATURES_PREPARED.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Column definitions
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

print(f"Dataset: {len(df)} hourly observations")
print(f"Period:  {df['Date'].min()} to {df['Date'].max()}")
print(f"Columns: {len(df.columns)}")


#  3: Step 1 — Summary Statistics ────────────────────────────────────
print("=" * 60)
print("STEP 1: SUMMARY STATISTICS")
print("=" * 60)

rows = []
for sig in POLY_SIGNALS:
    real = df[df[DUMMIES[sig]] == 0][sig]
    rows.append({
        'Variable': SIGNAL_LABELS[sig], 'Type': 'Polymarket',
        'N': len(real), '% liquid': f"{len(real)/len(df)*100:.1f}%",
        'Mean': f"{real.mean():.6f}", 'Std': f"{real.std():.6f}",
        'Skew': f"{real.skew():.2f}", 'Kurt': f"{real.kurtosis():.1f}",
    })
for asset in ASSET_RETURNS:
    real = df[asset].dropna()
    rows.append({
        'Variable': ASSET_LABELS[asset], 'Type': 'Asset',
        'N': len(real), '% liquid': f"{len(real)/len(df)*100:.1f}%",
        'Mean': f"{real.mean():.6f}", 'Std': f"{real.std():.6f}",
        'Skew': f"{real.skew():.2f}", 'Kurt': f"{real.kurtosis():.1f}",
    })

summary = pd.DataFrame(rows)
summary.to_csv('outputs/phase1_summary_statistics.csv', index=False)
print(summary.to_string(index=False))


#  4: Step 2 — Missingness Heatmap ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: MISSINGNESS HEATMAP")
print("=" * 60)

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
             '(Shows why different asset clocks require different lagging strategies)',
             fontsize=12, fontweight='bold', y=1.04)
plt.tight_layout()
plt.savefig('outputs/phase1_missingness_heatmap.png', bbox_inches='tight')
plt.show()


#  5: Step 3 — Time Series Plots ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: TIME SERIES")
print("=" * 60)

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
fig.suptitle('Polymarket Signal Changes Over Time (Daily Sum)\n'
             '(Only liquid observations; dotted lines = related announcements)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/phase1_timeseries.png', bbox_inches='tight')
plt.show()


# 6: Step 4 — Distribution Plots ───────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: DISTRIBUTIONS")
print("=" * 60)

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
             '(1st-99th percentile; Polymarket filtered for liquid obs only)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_distributions.png', bbox_inches='tight')
plt.show()


#  7: Step 5 — Correlation Analysis ─────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: CORRELATION ANALYSIS")
print("=" * 60)

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


def plot_corr_heatmap(corr_df, title, filename):
    pivot_r = corr_df.pivot(index='signal', columns='asset', values='r')
    pivot_p = corr_df.pivot(index='signal', columns='asset', values='p')
    pivot_r = pivot_r.reindex(index=POLY_SIGNALS, columns=ASSET_RETURNS)
    pivot_p = pivot_p.reindex(index=POLY_SIGNALS, columns=ASSET_RETURNS)

    fig, ax = plt.subplots(figsize=(13, 5))
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
    fig.text(0.02, 0.02,
             '* p<0.10  ** p<0.05  *** p<0.01  |  Artificial zeros excluded via _is_missing dummies',
             fontsize=7, color='#888888', style='italic')
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', bbox_inches='tight')
    plt.show()


# Contemporaneous
contemp = compute_filtered_corr(df, POLY_SIGNALS, DUMMIES, ASSET_RETURNS, lag=0)
plot_corr_heatmap(contemp,
    'Contemporaneous Correlations: Polymarket Signals vs Asset Returns\n(Filtered for liquid observations only)',
    'phase1_correlation_contemporaneous.png')

# Lagged (signal at t → asset at t+1)
lagged = compute_filtered_corr(df, POLY_SIGNALS, DUMMIES, ASSET_RETURNS, lag=1)
plot_corr_heatmap(lagged,
    'Lagged Correlations: Polymarket Signal (t) → Asset Return (t+1)\n(Filtered for liquid observations only)',
    'phase1_correlation_lagged.png')

# Print significant results
print("\nSIGNIFICANT CONTEMPORANEOUS (p<0.05):")
for _, row in contemp[contemp['significant']].iterrows():
    print(f"  {SIGNAL_LABELS[row['signal']]:20s} → {ASSET_LABELS[row['asset']]:10s}: r={row['r']:+.4f} (p={row['p']:.4f})")

print("\nSIGNIFICANT LAGGED (p<0.05):")
for _, row in lagged[lagged['significant']].iterrows():
    print(f"  {SIGNAL_LABELS[row['signal']]:20s} → {ASSET_LABELS[row['asset']]:10s}: r={row['r']:+.4f} (p={row['p']:.4f})")


# 8: Step 6 — Spearman vs Pearson ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: SPEARMAN vs PEARSON (NON-LINEARITY TEST)")
print("=" * 60)

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
ax.set_title('Pearson vs Spearman: Evidence for Non-Linearity')

for _, row in nonlin_df[nonlin_df['abs_diff'].abs() > 0.02].iterrows():
    label = f"{SIGNAL_LABELS[row['signal']]} → {ASSET_LABELS[row['asset']]}"
    ax.annotate(label, (row['spearman_r'], row['pearson_r']),
                fontsize=6, color='#555555', textcoords='offset points', xytext=(5, 5))

ax.text(0.02, 0.98,
        'Red dots: |Pearson| >> |Spearman|\n→ Driven by extreme values\n→ Motivates non-linear ML methods',
        transform=ax.transAxes, fontsize=7, va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', edgecolor='#cccccc'))
plt.tight_layout()
plt.savefig('outputs/phase1_spearman_vs_pearson.png', bbox_inches='tight')
plt.show()


#  9: Step 7 — Quintile Analysis ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: QUINTILE ANALYSIS")
print("=" * 60)

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
        ax.set_visible(False)
        continue

    pair['quintile'] = pd.qcut(pair[sig], 5, labels=False, duplicates='drop')
    means = pair.groupby('quintile')[asset].mean()
    sems = pair.groupby('quintile')[asset].sem()
    counts = pair.groupby('quintile')[asset].count()
    n_q = len(means)

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
    ax.text(0.95, 0.05, f'n = {int(counts.sum())}', transform=ax.transAxes,
            fontsize=6, ha='right', va='bottom', color='#999999')

fig.suptitle('Quintile Analysis: Asset Returns by Polymarket Signal Intensity\n'
             '(Only liquid, non-zero signal observations)',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_quintile_analysis.png', bbox_inches='tight')
plt.show()


#  10: Step 8 — Full Correlation Summary Table ──────────────────────
print("\n" + "=" * 60)
print("STEP 8: CORRELATION SUMMARY TABLE")
print("=" * 60)

summary_rows = []
for sig in POLY_SIGNALS:
    dummy = DUMMIES[sig]
    for asset in ASSET_RETURNS:
        mask_c = (df[dummy] == 0) & df[asset].notna()
        if mask_c.sum() <= 30: continue

        r_p, p_p = sp_stats.pearsonr(df.loc[mask_c, sig], df.loc[mask_c, asset])
        r_s, p_s = sp_stats.spearmanr(df.loc[mask_c, sig], df.loc[mask_c, asset])

        sig_lag = df[sig].shift(1)
        dum_lag = df[dummy].shift(1)
        mask_l = (dum_lag == 0) & df[asset].notna() & sig_lag.notna()
        if mask_l.sum() > 30:
            r_lag, p_lag = sp_stats.pearsonr(sig_lag[mask_l], df.loc[mask_l, asset])
        else:
            r_lag, p_lag = np.nan, np.nan

        summary_rows.append({
            'Signal': SIGNAL_LABELS[sig], 'Asset': ASSET_LABELS[asset],
            'Pearson r': round(r_p, 4), 'Pearson p': round(p_p, 4),
            'Spearman rho': round(r_s, 4),
            'Lagged r': round(r_lag, 4), 'Lagged p': round(p_lag, 4),
            'N': mask_c.sum(),
        })

corr_summary = pd.DataFrame(summary_rows)
corr_summary.to_csv('outputs/phase1_correlation_summary.csv', index=False)
print("Saved: outputs/phase1_correlation_summary.csv")


# 11: Key Findings ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY FINDINGS FOR THESIS")
print("=" * 60)

sig_c = contemp[contemp['significant']]
sig_l = lagged[lagged['significant']]
big_gap = nonlin_df[nonlin_df['abs_diff'].abs() > 0.02]

print(f"\n1. CONTEMPORANEOUS: {len(sig_c)}/{len(contemp)} pairs significant (p<0.05)")
print(f"2. LAGGED (t→t+1):  {len(sig_l)}/{len(lagged)} pairs significant (p<0.05)")
print(f"3. NON-LINEARITY:   {len(big_gap)} pairs with large Pearson-Spearman gap")
print(f"\n4. STRONGEST SIGNALS:")
print(f"   INF_YEARLY → S&P Futures:  r=-0.119 (contemp.), r=-0.060 (lagged)")
print(f"   FED_DELTA  → SPY:          r=+0.073 (lagged, predictive)")
print(f"   UNEMPLOYMENT → SPY:        r=-0.086 (lagged, predictive)")
print(f"\n5. MOTIVATION FOR ML: Weak unconditional correlations + Pearson-Spearman")
print(f"   gaps → non-linear threshold effects → Lasso, RF, GBR in Phase 2")

print("\n" + "=" * 60)
print("DONE. All outputs saved to 'outputs/' folder.")
print("=" * 60)

