"""
Phase 1: Descriptive Statistics & Correlation Analysis
FML Group Project 6 - Prediction Markets
 
This script runs the exploratory analysis on our Polymarket data.
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mstats
import os
import warnings
warnings.filterwarnings('ignore')
 
# Make a folder to save our charts
os.makedirs('outputs', exist_ok=True)
 
 
# =============================================
# LOAD THE DATA
# =============================================
 
# Read in the cleaned dataset
df = pd.read_csv('FEATURES_PREPARED.csv')
 
# Convert the Date column from a string to a proper datetime object
# so we can do time-based operations later 
df['Date'] = pd.to_datetime(df['Date'])
 
# Sort by date 
df = df.sort_values('Date').reset_index(drop=True)
 
print(f"Loaded {len(df)} rows")
print(f"From {df['Date'].min()} to {df['Date'].max()}")
 
 
# =============================================
# DEFINE OUR VARIABLES
# =============================================
 
# These are the 4 Polymarket signals we're using as features.
# Each one is the HOURLY CHANGE in the market-implied probability
# of a specific macro outcome.
#
# FED_DELTA:           change in expected Fed rate move (in bps)
# GDP_DELTA:           change in expected GDP growth
# UNEMPLOYMENT_DELTA:  change in expected unemployment rate
# INF_MONTHLY_DELTA:   change in expected monthly inflation
#
 
signals = ['FED_DELTA', 'GDP_DELTA', 'UNEMPLOYMENT_DELTA', 'INF_MONTHLY_DELTA']
 
# For each signal, there's a dummy variable that tells us whether
# the original data was missing (1 = missing, 0 = real data).

missing_dummies = {
    'FED_DELTA': 'FED_DELTA_is_missing',
    'GDP_DELTA': 'GDP_DELTA_is_missing',
    'UNEMPLOYMENT_DELTA': 'UNEMPLOYMENT_DELTA_is_missing',
    'INF_MONTHLY_DELTA': 'INF_MONTHLY_DELTA_is_missing',
}
 
# Define assets we will predict
assets = ['SPY_chg', 'QQQ_chg', 'Gold_chg', 'Oil_chg', 'DXY_chg',
          'BTC_chg', 'VIX_chg', 'SP500_fut_chg', 'US2Y_chg', 'US10Y_chg']
 
# Short names
signal_names = {
    'FED_DELTA': 'Fed Rate',
    'GDP_DELTA': 'GDP Growth',
    'UNEMPLOYMENT_DELTA': 'Unemployment',
    'INF_MONTHLY_DELTA': 'Monthly Inflation',
}
 
asset_names = {
    'SPY_chg': 'SPY', 'QQQ_chg': 'QQQ', 'Gold_chg': 'Gold',
    'Oil_chg': 'Oil', 'DXY_chg': 'USD (DXY)', 'BTC_chg': 'Bitcoin',
    'VIX_chg': 'VIX', 'SP500_fut_chg': 'S&P Fut.',
    'US2Y_chg': '2Y Yield', 'US10Y_chg': '10Y Yield',
}
 
 
# =============================================
# STEP 1: SUMMARY STATISTICS
# =============================================
# Goal: get basic stats (mean, std, skewness, kurtosis) for all variables.
# IMPORTANT: for the Polymarket signals, we filter OUT the artificial zeros
# using the missing dummies. Fake zeros give unreal statistics.
 
print("\n--- Summary Statistics ---")
 
rows = []
 
for sig in signals:
    # Only keep rows where the data is REAL (dummy == 0)
    dummy_col = missing_dummies[sig]
    real_data = df[df[dummy_col] == 0][sig]
 
    rows.append({
        'Variable': signal_names[sig],
        'Type': 'Polymarket',
        'N': len(real_data),
        '% liquid': round(len(real_data) / len(df) * 100, 1),
        'Mean': real_data.mean(),
        'Std': real_data.std(),
        'Min': real_data.min(),
        'Max': real_data.max(),
        'Skew': real_data.skew(),
        'Kurt': real_data.kurtosis(),
        # What % of hours close to zero movement
        '% near zero': round((real_data.abs() < 0.001).mean() * 100, 1),
    })
 
for asset in assets:
    # For assets, NaN means the market was closed - drop these
    real_data = df[asset].dropna()
 
    rows.append({
        'Variable': asset_names[asset],
        'Type': 'Asset',
        'N': len(real_data),
        '% liquid': round(len(real_data) / len(df) * 100, 1),
        'Mean': real_data.mean(),
        'Std': real_data.std(),
        'Min': real_data.min(),
        'Max': real_data.max(),
        'Skew': real_data.skew(),
        'Kurt': real_data.kurtosis(),
        '% near zero': round((real_data.abs() < 0.001).mean() * 100, 1),
    })
 
summary_table = pd.DataFrame(rows)
summary_table.to_csv('outputs/phase1_summary_statistics.tex', index=False)
print("Saved summary statistics")
 
 
# =============================================
# CHART 1: MISSINGNESS HEATMAP
# =============================================
# Goal: show WHEN each variable has data available.
# This matters because Polymarket trades 24/7 but SPY only trades
# ~6.5 hours per day. This is why we can't just do a simple 1-hour lag
# for equities.
#
# We compute the % of observations available for each (variable, hour)
# and (variable, day-of-week) combination.
 
print("\n--- Chart 1: Missingness Heatmap ---")
 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
# --- Left panel: by hour of day ---
# For each hour (0-23), compute what fraction of rows have real data
 
hour_data = pd.DataFrame()
 
for sig in signals:
    dummy_col = missing_dummies[sig]
    # Group by hour, then for each group compute the fraction where dummy == 0
    hour_data[signal_names[sig]] = df.groupby('hour_utc').apply(
        lambda g: (g[dummy_col] == 0).mean()
    )
 
# Do the same for a key assets (using notna instead of dummies)
for asset in ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg']:
    hour_data[asset_names[asset]] = df.groupby('hour_utc').apply(
        lambda g: g[asset].notna().mean()
    )
 
# Plot as a heatmap: green = available, red = missing
# imshow() takes a 2D array and colors each cell
cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
im1 = ax1.imshow(hour_data.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax1.set_xticks(range(24))
ax1.set_xticklabels(range(24), fontsize=7)
ax1.set_yticks(range(len(hour_data.columns)))
ax1.set_yticklabels(hour_data.columns, fontsize=8)
ax1.set_xlabel('Hour (UTC)')
ax1.set_title('By Hour of Day')
plt.colorbar(im1, ax=ax1, shrink=0.8, label='% available')
 
# Add the percentage text inside each cell
for i in range(hour_data.T.values.shape[0]):
    for j in range(hour_data.T.values.shape[1]):
        val = hour_data.T.values[i, j]
        # Use white text on dark cells, black on light cells
        color = 'white' if val < 0.5 else 'black'
        ax1.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=5, color=color)
 
# --- Right panel: by day of week ---
dow_data = pd.DataFrame()
dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
 
for sig in signals:
    dummy_col = missing_dummies[sig]
    dow_data[signal_names[sig]] = df.groupby('dow_utc').apply(
        lambda g: (g[dummy_col] == 0).mean()
    )
 
for asset in ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg']:
    dow_data[asset_names[asset]] = df.groupby('dow_utc').apply(
        lambda g: g[asset].notna().mean()
    )
 
im2 = ax2.imshow(dow_data.T.values, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(7))
ax2.set_xticklabels(dow_names, fontsize=8)
ax2.set_yticks(range(len(dow_data.columns)))
ax2.set_yticklabels(dow_data.columns, fontsize=8)
ax2.set_xlabel('Day of Week')
ax2.set_title('By Day of Week')
plt.colorbar(im2, ax=ax2, shrink=0.8, label='% available')
 
for i in range(dow_data.T.values.shape[0]):
    for j in range(dow_data.T.values.shape[1]):
        val = dow_data.T.values[i, j]
        color = 'white' if val < 0.5 else 'black'
        ax2.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=7, color=color)
 
fig.suptitle('Data Availability: Polymarket Signals & Asset Returns', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_missingness_heatmap.png', bbox_inches='tight')
plt.show()
print("Saved missingness heatmap")
 
 
# =============================================
# CHART 2: DISTRIBUTIONS
# =============================================
# Goal: show what the distributions of our signals and returns look like.
# We clip at the 1st and 99th percentile to remove extreme outliers.
 
print("\n--- Chart 2: Distributions ---")
 
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
 
# Top row: Polymarket signals
for i, sig in enumerate(signals):
    ax = axes[0, i]
    dummy_col = missing_dummies[sig]
    real = df[df[dummy_col] == 0][sig]
 
    # Clip at 1st/99th percentile so the plot isn't dominated by outliers
    lower = real.quantile(0.01)
    upper = real.quantile(0.99)
    clipped = real[(real >= lower) & (real <= upper)]
 
    ax.hist(clipped, bins=50, alpha=0.7, edgecolor='white', density=True)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(signal_names[sig], fontsize=10)
    ax.text(0.95, 0.95,
            f'n={len(real)}\nstd={real.std():.4f}\nskew={real.skew():.1f}\nkurt={real.kurtosis():.0f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
 
# Bottom row: asset returns
key_assets = ['SPY_chg', 'Gold_chg', 'BTC_chg', 'SP500_fut_chg']
for i, asset in enumerate(key_assets):
    ax = axes[1, i]
    real = df[asset].dropna()
 
    lower = real.quantile(0.01)
    upper = real.quantile(0.99)
    clipped = real[(real >= lower) & (real <= upper)]
 
    ax.hist(clipped, bins=50, alpha=0.7, edgecolor='white', density=True)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(asset_names[asset], fontsize=10)
    ax.text(0.95, 0.95,
            f'n={len(real)}\nstd={real.std():.4f}\nskew={real.skew():.1f}\nkurt={real.kurtosis():.0f}',
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
 
axes[0, 0].set_ylabel('Density (Polymarket)')
axes[1, 0].set_ylabel('Density (Assets)')
fig.suptitle('Distributions (1st-99th percentile, liquid obs only)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_distributions.png', bbox_inches='tight')
plt.show()
print("Saved distributions")
 
 
# =============================================
# CHARTS 3 & 4: CORRELATION HEATMAPS
# =============================================
# Goal: compute Pearson correlations between each signal-asset pair.
#
# Chart 3 = CONTEMPORANEOUS: signal and asset return in the SAME hour.
#   This tells us about co-movement (both reacting to news at once),
#   NOT about prediction.
#
# Chart 4 = LAGGED: signal at hour t, asset return at hour t+1.
#   This is closer to a predictive relationship -> does a Polymarket
#   move in one hour predict an asset move in the NEXT hour?
#
# For both: we filter out artificial zeros using the missing dummies.
#   pearsonr() returns (correlation, p-value).
 
print("\n--- Charts 3 & 4: Correlations ---")
 
 
def compute_correlations(df, signals, missing_dummies, assets, lag=0):
    """
    Compute Pearson correlations between signals and assets.
 
    If lag=0: contemporaneous (same hour).
    If lag=1: signal at hour t vs asset at hour t+1.
 
    We filter so that:
    - the signal is not an artificial zero (dummy == 0)
    - the asset return exists (not NaN)
    """
    results = []
 
    for sig in signals:
        dummy_col = missing_dummies[sig]
 
        # If lagged, shift the signal and dummy backward by 1 hour
        # shift(1) means: row i gets the value from row i-1
        # So signal_values[i] contains hour t-1's signal,
        # and we pair it with hour t's asset return
        if lag > 0:
            signal_values = df[sig].shift(lag)
            dummy_values = df[dummy_col].shift(lag)
        else:
            signal_values = df[sig]
            dummy_values = df[dummy_col]
 
        for asset in assets:
            # Build a mask: keep rows where signal is real AND asset exists
            mask = (dummy_values == 0) & df[asset].notna() & signal_values.notna()
            n = mask.sum()
 
            if n > 30:  # need enough data for a meaningful correlation
                r, p = stats.pearsonr(signal_values[mask], df.loc[mask, asset])
                results.append({
                    'signal': sig,
                    'asset': asset,
                    'r': r,
                    'p': p,
                    'n': n,
                    'significant': p < 0.05,
                })
 
    return pd.DataFrame(results)
 
 
def plot_correlation_heatmap(corr_df, title, subtitle, filename):
    """
    Plot a heatmap of correlations with significance stars.
 
    pivot() reshapes the data so signals are rows and assets are columns.
    imshow() colors each cell by the correlation value.
    """
    # Reshape into a matrix: rows = signals, columns = assets
    r_matrix = corr_df.pivot(index='signal', columns='asset', values='r')
    p_matrix = corr_df.pivot(index='signal', columns='asset', values='p')
 
    # Reorder to match our lists
    r_matrix = r_matrix.reindex(index=signals, columns=assets)
    p_matrix = p_matrix.reindex(index=signals, columns=assets)
 
    fig, ax = plt.subplots(figsize=(13, 4.5))
 
    # Color scheme: blue = negative, white = zero, red = positive
    cmap = plt.cm.RdBu_r  # Red-Blue reversed (so red = positive)
 
    # Set the color scale symmetric around zero
    # Cap at 0.15 so small correlations are still visible
    valid_values = r_matrix.values[~np.isnan(r_matrix.values)]
    vmax = min(max(abs(valid_values.min()), abs(valid_values.max())), 0.15)
 
    im = ax.imshow(r_matrix.values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
 
    # Labels
    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels([asset_names[a] for a in assets], rotation=45, ha='right')
    ax.set_yticks(range(len(signals)))
    ax.set_yticklabels([signal_names[s] for s in signals])
 
    # Add correlation values and significance stars inside each cell
    for i in range(len(signals)):
        for j in range(len(assets)):
            r_val = r_matrix.values[i, j]
            p_val = p_matrix.values[i, j]
 
            if np.isnan(r_val):
                continue
 
            # Stars: *** means p < 1%, ** means p < 5%, * means p < 10%
            if p_val < 0.01:
                star = '***'
            elif p_val < 0.05:
                star = '**'
            elif p_val < 0.10:
                star = '*'
            else:
                star = ''
 
            color = 'white' if abs(r_val) > vmax * 0.6 else 'black'
            weight = 'bold' if star else 'normal'
            ax.text(j, i, f'{r_val:.3f}{star}', ha='center', va='center',
                    fontsize=8, color=color, fontweight=weight)
 
    plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
    ax.set_title(title, pad=12)
    fig.text(0.5, 0.01, subtitle, fontsize=7, color='gray', ha='center', style='italic')
 
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', bbox_inches='tight')
    plt.show()
 
 
# Compute and plot contemporaneous correlations (lag=0)
contemp = compute_correlations(df, signals, missing_dummies, assets, lag=0)
plot_correlation_heatmap(
    contemp,
    'Contemporaneous Correlations: Polymarket Signals vs Asset Returns',
    'Caution: these reflect co-movement, not prediction. * p<0.10  ** p<0.05  *** p<0.01',
    'phase1_correlation_contemporaneous.png'
)
 
# Compute and plot lagged correlations (lag=1)
lagged = compute_correlations(df, signals, missing_dummies, assets, lag=1)
plot_correlation_heatmap(
    lagged,
    'Lagged Correlations: Polymarket Signal (t) vs Asset Return (t+1)',
    'For SPY/QQQ: only consecutive trading hours (~20% of data). * p<0.10  ** p<0.05  *** p<0.01',
    'phase1_correlation_lagged.png'
)
 
# Print which pairs are significant
sig_contemp = contemp[contemp['significant']]
sig_lagged = lagged[lagged['significant']]
print(f"Contemporaneous: {len(sig_contemp)}/40 significant at 5%")
print(f"Lagged: {len(sig_lagged)}/40 significant at 5%")
 
 
# =============================================
# CHART 5: ASYMMETRIC CORRELATIONS
# =============================================
# Goal: test whether POSITIVE and NEGATIVE signal changes have
# different effects on asset returns.
#
# Why? Diercks et al. (2025) found that positive CPI surprises
# move Fed rate expectations 4x more than negative surprises.
# We check if the same asymmetry exists in our data.
#
# Method: split each signal into positive (> 0) and negative (< 0)
# observations, compute Pearson r separately for each half.
# If r_positive is much larger than r_negative (or vice versa),
# the relationship is asymmetric.
 
print("\n--- Chart 5: Asymmetric Correlations ---")
 
asym_pairs = [
    ('FED_DELTA', 'SPY_chg', 'Fed Rate vs SPY'),
    ('FED_DELTA', 'SP500_fut_chg', 'Fed Rate vs S&P Fut.'),
    ('UNEMPLOYMENT_DELTA', 'SPY_chg', 'Unemployment vs SPY'),
    ('UNEMPLOYMENT_DELTA', 'SP500_fut_chg', 'Unemployment vs S&P Fut.'),
    ('INF_MONTHLY_DELTA', 'BTC_chg', 'Monthly Infl. vs Bitcoin'),
    ('INF_MONTHLY_DELTA', 'SP500_fut_chg', 'Monthly Infl. vs S&P Fut.'),
]
 
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
 
for idx, (sig, asset, title) in enumerate(asym_pairs):
    ax = axes[idx]
    dummy_col = missing_dummies[sig]
 
    # Filter: signal is real AND asset return exists
    mask = (df[dummy_col] == 0) & df[asset].notna()
 
    # Split into positive and negative signal changes
    pos_mask = mask & (df[sig] > 0)
    neg_mask = mask & (df[sig] < 0)
 
    if pos_mask.sum() < 30 or neg_mask.sum() < 30:
        ax.set_title(title)
        ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
        continue
 
    # Correlation for each direction separately
    r_pos, p_pos = stats.pearsonr(df.loc[pos_mask, sig], df.loc[pos_mask, asset])
    r_neg, p_neg = stats.pearsonr(df.loc[neg_mask, sig], df.loc[neg_mask, asset])
 
    # Scatter plot (returns in basis points for readability)
    ax.scatter(df.loc[pos_mask, sig], df.loc[pos_mask, asset] * 10000,
               c='indianred', s=8, alpha=0.3, label=f'Pos: r={r_pos:+.3f}')
    ax.scatter(df.loc[neg_mask, sig], df.loc[neg_mask, asset] * 10000,
               c='steelblue', s=8, alpha=0.3, label=f'Neg: r={r_neg:+.3f}')
 
    # Trend lines (linear fit)
    for m, color in [(pos_mask, 'indianred'), (neg_mask, 'steelblue')]:
        x = df.loc[m, sig].values
        y = df.loc[m, asset].values * 10000
        if len(x) > 10:
            coeffs = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, np.polyval(coeffs, x_line), color=color, linewidth=1.5)
 
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(f'{signal_names[sig]} change', fontsize=8)
    ax.set_ylabel('Asset return (bps)', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
 
    # Flag if asymmetric
    if abs(r_pos) > abs(r_neg) * 2 or abs(r_neg) > abs(r_pos) * 2:
        ax.text(0.95, 0.05, 'ASYMMETRIC', transform=ax.transAxes, fontsize=8,
                ha='right', va='bottom', color='red', fontweight='bold')
 
fig.suptitle('Asymmetric Correlations: Positive vs Negative Signal Changes\n'
             '(cf. Diercks et al. 2025, Figure 17)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('outputs/phase1_asymmetric_correlations.png', bbox_inches='tight')
plt.show()
print("Saved asymmetric correlations")
 
 
# =============================================
# CHART 6: SIGNAL CONCENTRATION ON NEWS DAYS
# =============================================
# Goal: validate that our signals actually respond to the macro
# events they're supposed to capture.
#
# Method: for each signal, compare the average absolute change
# on announcement days vs all other days. If the signal is real,
# it should be much larger on news days.
#
# We use the Mann-Whitney U test (a non-parametric test that
# doesn't assume normality) to check if the difference is
# statistically significant. We use one-sided because we expect
# news days to be LARGER, not just different.
 
print("\n--- Chart 6: Signal Concentration ---")
 
ann_pairs = [
    ('Ann_FOMC', 'FED_DELTA', 'FOMC', 'Fed Rate'),
    ('Ann_CPI', 'INF_MONTHLY_DELTA', 'CPI', 'Monthly Infl.'),
    ('Ann_Employment', 'UNEMPLOYMENT_DELTA', 'Employment', 'Unemployment'),
    ('Ann_GDP', 'GDP_DELTA', 'GDP', 'GDP Growth'),
]
 
fig, axes = plt.subplots(1, 4, figsize=(15, 4.5))
 
for idx, (ann_col, sig, ann_name, sig_name) in enumerate(ann_pairs):
    ax = axes[idx]
    dummy_col = missing_dummies[sig]
 
    # Only liquid observations
    liquid = df[dummy_col] == 0
 
    # Split into news vs other days
    news = df[liquid & (df[ann_col] == 1)][sig].abs()
    other = df[liquid & (df[ann_col] == 0)][sig].abs()
 
    if len(news) < 3:
        ax.set_visible(False)
        continue
 
    ratio = news.mean() / other.mean() if other.mean() > 0 else 0
    _, p_value = stats.mannwhitneyu(news, other, alternative='greater')
 
    ax.bar(['News\ndays', 'Other\ndays'],
           [news.mean() * 1000, other.mean() * 1000],
           color=['indianred', 'steelblue'])
    ax.set_ylabel('Mean |signal change| (x1000)', fontsize=8)
    ax.set_title(f'{ann_name} -> {sig_name}', fontsize=10)
 
    if p_value < 0.01: stars = '***'
    elif p_value < 0.05: stars = '**'
    elif p_value < 0.10: stars = '*'
    else: stars = 'n.s.'
 
    ax.text(0.95, 0.95, f'{ratio:.1f}x larger\np={p_value:.3f} {stars}\nn={len(news)} events',
            transform=ax.transAxes, fontsize=8, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.8))
 
fig.suptitle('Signal Validation: Are Signals Larger on Announcement Days?\n'
             '(Mann-Whitney U test, cf. Diercks et al. 2025)', fontsize=13, y=1.05)
plt.tight_layout()
plt.savefig('outputs/phase1_signal_concentration.png', bbox_inches='tight')
plt.show()
print("Saved signal concentration")
 
 
# =============================================
# CORRELATION SUMMARY TABLE (for appendix)
# =============================================
# Saves everything in one CSV: raw Pearson, winsorised Pearson,
# Spearman (rank), and lagged correlations.
 
print("\n--- Correlation Summary Table ---")
 
summary_rows = []
 
for sig in signals:
    dummy_col = missing_dummies[sig]
    for asset in assets:
        mask = (df[dummy_col] == 0) & df[asset].notna()
        if mask.sum() <= 30:
            continue
 
        x = df.loc[mask, sig]
        y = df.loc[mask, asset]
 
        r_raw, p_raw = stats.pearsonr(x, y)
        r_spear, _ = stats.spearmanr(x, y)
 
        # Winsorised: clip extremes before correlating
        x_wins = mstats.winsorize(x.values, limits=[0.01, 0.01])
        y_wins = mstats.winsorize(y.values, limits=[0.01, 0.01])
        r_wins, _ = stats.pearsonr(x_wins, y_wins)
 
        # Lagged
        sig_lag = df[sig].shift(1)
        dum_lag = df[dummy_col].shift(1)
        lag_mask = (dum_lag == 0) & df[asset].notna() & sig_lag.notna()
        if lag_mask.sum() > 30:
            r_lag, p_lag = stats.pearsonr(sig_lag[lag_mask], df.loc[lag_mask, asset])
        else:
            r_lag, p_lag = np.nan, np.nan
 
        summary_rows.append({
            'Signal': signal_names[sig],
            'Asset': asset_names[asset],
            'r_raw': round(r_raw, 4),
            'p_raw': round(p_raw, 4),
            'r_winsorised': round(r_wins, 4),
            'r_spearman': round(r_spear, 4),
            'r_lagged': round(r_lag, 4),
            'p_lagged': round(p_lag, 4),
            'N': mask.sum(),
        })
 
pd.DataFrame(summary_rows).to_csv('outputs/phase1_correlation_summary.tex', index=False)
print("Saved correlation summary")
 
 
# =============================================
# PRINT KEY FINDINGS
# =============================================
print("\n" + "=" * 50)
print("KEY FINDINGS")
print("=" * 50)
print(f"Contemporaneous: {len(sig_contemp)}/40 pairs significant")
print(f"Lagged:          {len(sig_lagged)}/40 pairs significant")
print(f"\nStrongest lagged predictors for SPY:")
spy_results = sig_lagged[sig_lagged['asset'] == 'SPY_chg']
for _, row in spy_results.iterrows():
    print(f"  {signal_names[row['signal']]}: r={row['r']:+.3f} (p={row['p']:.4f})")
