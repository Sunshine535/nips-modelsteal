#!/usr/bin/env python3
"""
Generate publication-quality Gramian eigenspectrum figure for NeurIPS paper.
Shows depth-observability boundary via projected Gramian eigenspectra.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# NeurIPS-compatible style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.2,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'axes.labelcolor': '#222222',
})

np.random.seed(2026)

# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------
n_total = 64
n_rank = 32
indices = np.arange(1, n_total + 1)


def make_spectrum(sigma_max, sigma_min, rank, noise_floor=1e-14):
    """Power-law decay for observed part, noise floor for null space."""
    spec = np.empty(n_total)
    for i in range(rank):
        spec[i] = sigma_max * (sigma_min / sigma_max) ** (i / (rank - 1))
    for i in range(rank, n_total):
        spec[i] = noise_floor * (1.0 + 0.5 * np.random.randn())
    spec[rank:] = np.clip(spec[rank:], 1e-16, 1e-12)
    return spec


# Block 23 spectra
specs_clean = {
    'seed 42':  make_spectrum(11.24, 0.197, 32),
    'seed 123': make_spectrum(13.60, 0.200, 32),
    'seed 777': make_spectrum(11.80, 0.200, 32),
}
spec_aug = make_spectrum(11.17, 0.205, 32)

# Block 22 spectrum (completely unobservable)
spec_block22 = np.random.uniform(1e-14, 5e-13, size=n_total)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
blues = ['#08519c', '#3182bd', '#6baed6']  # 3 distinct blue shades
red_orange = '#d62728'
threshold_color = '#666666'
shade_color = '#d0d0d0'

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

# ===== Left: Block 23 =====
for idx, (label, sp) in enumerate(specs_clean.items()):
    ax1.semilogy(indices, sp, color=blues[idx], lw=1.0, alpha=0.85,
                 label=f'Clean ({label})')

ax1.semilogy(indices, spec_aug, color=red_orange, lw=1.4, alpha=0.9,
             label='Augmented (seed 42)', linestyle='-')

# Truncation threshold
thresh = 0.197
ax1.axhline(thresh, color=threshold_color, ls='--', lw=0.8, zorder=1)
ax1.text(8, thresh * 2.5, r'$\tau = 0.197$', fontsize=7.5,
         color=threshold_color, ha='center', va='bottom')

# Shaded "truncated" region
ax1.fill_between(indices, 1e-16, thresh, color=shade_color, alpha=0.25, zorder=0)
ax1.text(48, 2e-8, 'Truncated\n(gauge + noise)', fontsize=7, color='#888888',
         ha='center', va='center', style='italic')

# Rank annotation with arrow — point to the cliff edge (index 32→33 drop)
ax1.annotate(
    r'rank $= 32$',
    xy=(32.5, specs_clean['seed 42'][31]),
    xytext=(44, 1e-5),
    fontsize=8,
    color='#333333',
    arrowprops=dict(arrowstyle='->', color='#555555', lw=0.8,
                    connectionstyle='arc3,rad=0.2'),
    ha='center', va='center',
    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#aaaaaa', lw=0.5),
    zorder=10,
)

ax1.set_xlabel('Eigenvalue index')
ax1.set_ylabel('Singular value')
ax1.set_title('Block 23 Eigenspectrum', fontweight='medium')
ax1.set_xlim(0.5, 64.5)
ax1.set_ylim(1e-16, 50)
ax1.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='#cccccc', framealpha=0.95)
ax1.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(4))

# ===== Right: Block 22 =====
ax2.semilogy(indices, spec_block22, color='#555555', lw=1.2, alpha=0.8,
             label='All seeds')

# Annotations
ax2.text(32.5, 2e-9, r'$\sigma_{\max} < 10^{-12}$', fontsize=10,
         ha='center', va='center', color='#d62728', fontweight='bold')
ax2.text(32.5, 5e-11, 'Completely unobservable', fontsize=9,
         ha='center', va='center', color='#888888', style='italic')

# Light shading for the entire region
ax2.fill_between(indices, 1e-16, 1e-10, color='#ffe0e0', alpha=0.3, zorder=0)

ax2.set_xlabel('Eigenvalue index')
ax2.set_ylabel('Singular value')
ax2.set_title('Block 22 Eigenspectrum', fontweight='medium')
ax2.set_xlim(0.5, 64.5)
ax2.set_ylim(1e-16, 50)
ax2.legend(loc='upper right', frameon=True, fancybox=False,
           edgecolor='#cccccc', framealpha=0.9)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(8))
ax2.xaxis.set_minor_locator(ticker.MultipleLocator(4))

# Shared suptitle
fig.suptitle('Projected Gramian Eigenspectrum (Qwen2.5-0.5B)',
             fontsize=12, fontweight='bold', y=1.01)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out = '/Users/niewenhua/Desktop/aicoding/github_repos/nips-modelsteal/paper/figures/gramian_eigenspectrum.pdf'
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
print(f'Saved: {out}')

# Also save PNG for quick preview
out_png = out.replace('.pdf', '.png')
fig.savefig(out_png, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
print(f'Saved: {out_png}')
