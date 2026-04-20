import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ── CONFIG — point this at your json files ──
BASE_DIR = Path("experiment1_pooled")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

COLORS = {'A': '#7F77DD', 'B': '#1D9E75', 'C': '#D85A30'}
ALPHA_INDIVIDUAL = 0.18

data = {}
for mode in ['A', 'B', 'C']:
    with open(BASE_DIR / f'experiment_{mode}.json') as f:
        data[mode] = json.load(f)['pairs']


# ── SUMMARY STATS ──
print(f"{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
for mode in ['A', 'B', 'C']:
    pairs = data[mode]
    worst = [p['worst_layer_min_delta'] for p in pairs]
    x = np.array([p['baseline_ld_clean'] for p in pairs])
    y = np.array([min(p['ld_delta_vs_clean_baseline_by_layer']) for p in pairs])
    r, pval = stats.pearsonr(x, y)
    onset = [next((i for i, d in enumerate(p['ld_delta_vs_clean_baseline_by_layer']) if d < -5), None) for p in pairs]
    valid_onset = [o for o in onset if o is not None]
    print(f"\nMode {mode} (n={len(pairs)}):")
    print(f"  Worst layer: min={min(worst)}  max={max(worst)}  mean={np.mean(worst):.2f}")
    print(f"  All worst >= 15: {all(w >= 15 for w in worst)}")
    print(f"  All worst >= 13: {all(w >= 13 for w in worst)}")
    print(f"  Mean onset layer (<-5): {np.mean(valid_onset):.2f}  (valid={len(valid_onset)}/{len(pairs)})")
    print(f"  Pearson r (confidence vs damage): {r:.3f},  p={pval:.4f}")


# ── FIGURE 1: Mean ld_delta curves ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Mean ld_delta curves by experiment mode', fontsize=13, fontweight='500', y=1.01)

for ax, mode in zip(axes, ['A', 'B', 'C']):
    pairs = data[mode]
    all_deltas = np.array([p['ld_delta_vs_clean_baseline_by_layer'] for p in pairs])
    layers = np.arange(18)
    mean_delta = all_deltas.mean(axis=0)

    for row in all_deltas:
        ax.plot(layers, row, color=COLORS[mode], alpha=ALPHA_INDIVIDUAL, linewidth=0.8)
    ax.plot(layers, mean_delta, color=COLORS[mode], linewidth=2.5, label='Mean', zorder=5)
    ax.axhline(-5, color='#888', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axvline(14.5, color='#D85A30', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.set_title(f'Mode {mode}  (n={len(pairs)})', fontweight='500')
    ax.set_xlabel('Layer')
    ax.set_xticks([0, 3, 6, 9, 12, 15, 17])
    if mode == 'A':
        ax.set_ylabel('ld_delta vs clean baseline')

    onset_layers = [next((i for i, d in enumerate(p['ld_delta_vs_clean_baseline_by_layer']) if d < -5), None) for p in pairs]
    valid = [o for o in onset_layers if o is not None]
    if valid:
        mean_onset = np.mean(valid)
        ax.axvline(mean_onset, color=COLORS[mode], linewidth=1.2, linestyle='--', alpha=0.7,
                   label=f'Mean onset L{mean_onset:.1f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_mean_ld_delta_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nFig 1 saved")


# ── FIGURE 2: Worst layer distribution ──
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('Worst layer distribution across modes', fontsize=13, fontweight='500')

for ax, mode in zip(axes, ['A', 'B', 'C']):
    pairs = data[mode]
    worst = [p['worst_layer_min_delta'] for p in pairs]
    layers = list(range(13, 18))
    counts = [worst.count(l) for l in layers]
    bars = ax.bar(layers, counts, color=COLORS[mode], alpha=0.85, width=0.6, edgecolor='white')
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=10, fontweight='500')
    ax.set_title(f'Mode {mode}  (n={len(pairs)})', fontweight='500')
    ax.set_xlabel('Worst layer')
    ax.set_ylabel('Count')
    ax.set_xticks(layers)
    mean_w = np.mean(worst)
    ax.axvline(mean_w, color='#333', linewidth=1.5, linestyle='--', alpha=0.7, label=f'Mean={mean_w:.2f}')
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_worst_layer_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 2 saved")


# ── FIGURE 3: Correlation baseline_ld_clean vs min_delta ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Correlation: model confidence (baseline_ld_clean) vs max damage (min_delta)', fontsize=13, fontweight='500')

for ax, mode in zip(axes, ['A', 'B', 'C']):
    pairs = data[mode]
    x = np.array([p['baseline_ld_clean'] for p in pairs])
    y = np.array([min(p['ld_delta_vs_clean_baseline_by_layer']) for p in pairs])
    slope, intercept, r, pval, _ = stats.linregress(x, y)
    r2 = r ** 2

    ax.scatter(x, y, color=COLORS[mode], alpha=0.75, s=60, edgecolors='white', linewidth=0.5, zorder=4)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=COLORS[mode], linewidth=1.8, alpha=0.8, zorder=3)

    cats = [p['category'] for p in pairs]
    for xi, yi, cat in zip(x, y, cats):
        ax.annotate(cat.split('_')[0][:6], (xi, yi), fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords='offset points')

    ax.set_title(f'Mode {mode}  r={r:.3f}  r²={r2:.3f}  p={pval:.4f}', fontweight='500')
    ax.set_xlabel('baseline_ld_clean (model confidence)')
    ax.set_ylabel('min ld_delta (max damage)')
    stats_text = f'r = {r:.3f}\nr² = {r2:.3f}\np = {pval:.4f}\nn = {len(pairs)}'
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#ddd'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_correlation_confidence_vs_damage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 3 saved")


# ── FIGURE 4: Correlation total_swing vs min_delta ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Correlation: TotalSwing vs max damage (min_delta)', fontsize=13, fontweight='500')

for ax, mode in zip(axes, ['A', 'B', 'C']):
    pairs = data[mode]
    x = np.array([p['total_swing'] for p in pairs])
    y = np.array([min(p['ld_delta_vs_clean_baseline_by_layer']) for p in pairs])
    slope, intercept, r, pval, _ = stats.linregress(x, y)
    r2 = r ** 2

    ax.scatter(x, y, color=COLORS[mode], alpha=0.75, s=60, edgecolors='white', linewidth=0.5, zorder=4)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=COLORS[mode], linewidth=1.8, alpha=0.8)

    cats = [p['category'] for p in pairs]
    for xi, yi, cat in zip(x, y, cats):
        ax.annotate(cat.split('_')[0][:6], (xi, yi), fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords='offset points')

    ax.set_title(f'Mode {mode}  r={r:.3f}  r²={r2:.3f}  p={pval:.4f}', fontweight='500')
    ax.set_xlabel('TotalSwing (LD_clean − LD_corrupt)')
    ax.set_ylabel('min ld_delta (max damage)')
    stats_text = f'r = {r:.3f}\nr² = {r2:.3f}\np = {pval:.4f}\nn = {len(pairs)}'
    ax.text(0.05, 0.05, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#ddd'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_correlation_totalswing_vs_damage.png', dpi=150, bbox_inches='tight')
plt.close()
print("Fig 4 saved")
print("\nDone. All figures in outputs/")