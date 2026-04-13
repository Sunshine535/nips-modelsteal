#!/usr/bin/env python3
"""Generate the Transformer Tomography framework overview figure for NeurIPS."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe
import numpy as np

# ---------- Global style ----------
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.linewidth": 0,
    "figure.dpi": 300,
})

# ---------- Color palette ----------
C_DARK      = "#3b4252"   # nord polar night – unobservable blocks
C_MID       = "#81a1c1"   # nord frost – partially observable
C_BRIGHT    = "#d08770"   # nord aurora orange – recoverable / highlight
C_BRIGHT2   = "#a3be8c"   # nord green – recovered annotation
C_FADED     = "#d8dee9"   # nord snow – faded / unrecoverable in student
C_ARROW     = "#4c566a"   # arrows
C_TEXT      = "#2e3440"   # text
C_BG        = "#ffffff"
C_DASHED    = "#bf616a"   # dashed divider line
C_EIGEN_HI  = "#5e81ac"   # eigenvalue bars nonzero
C_EIGEN_LO  = "#d8dee9"   # eigenvalue bars zero

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-0.5, 10.5)
ax.set_ylim(-0.3, 5.5)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(C_BG)

# ============================================================
# Helper: draw a rounded box with label
# ============================================================
def draw_block(ax, x, y, w, h, label, facecolor, edgecolor="#4c566a",
               fontsize=7.5, fontcolor="white", alpha=1.0, linestyle="-",
               hatch=None, linewidth=0.8, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                         facecolor=facecolor, edgecolor=edgecolor,
                         linewidth=linewidth, alpha=alpha, linestyle=linestyle,
                         zorder=zorder, mutation_scale=0.3)
    if hatch:
        box.set_hatch(hatch)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fontsize, color=fontcolor, zorder=zorder + 1,
            fontweight="medium")
    return box

# ============================================================
# SECTION 1 – Teacher (Black Box)  x ∈ [0.3, 2.5]
# ============================================================
sec1_x = 0.5
sec1_w = 1.8
block_w = 1.4
block_h = 0.38
block_x = sec1_x + (sec1_w - block_w) / 2
gap = 0.06

# Section title
ax.text(sec1_x + sec1_w / 2, 5.25, r"$\bf{Teacher\ (Black\ Box)}$",
        ha="center", va="center", fontsize=10, color=C_TEXT)

# Background panel
panel1 = FancyBboxPatch((sec1_x - 0.15, 0.15), sec1_w + 0.3, 4.95,
                         boxstyle="round,pad=0.08", facecolor="#eceff4",
                         edgecolor="#b0b8c8", linewidth=0.6, alpha=0.5, zorder=0)
ax.add_patch(panel1)

# Transformer blocks (bottom to top)
block_labels = ["Block 1", "Block 2", r"$\cdots$", "Block 22", "Block 23", "lm_head"]
block_colors = [C_DARK, C_DARK, C_DARK, C_DARK, C_MID, C_BRIGHT]
block_fontcolors = ["white", "white", C_TEXT, "white", "white", "white"]

y_start = 0.4
positions = []
for i, (lbl, col, fc) in enumerate(zip(block_labels, block_colors, block_fontcolors)):
    y = y_start + i * (block_h + gap)
    if lbl == r"$\cdots$":
        ax.text(block_x + block_w / 2, y + block_h / 2, lbl,
                ha="center", va="center", fontsize=9, color=C_TEXT, zorder=3)
        positions.append((block_x, y, block_w, block_h))
        continue
    draw_block(ax, block_x, y, block_w, block_h, lbl,
               facecolor=col, fontcolor=fc)
    positions.append((block_x, y, block_w, block_h))

# Dashed separator between Block 22 and Block 23
sep_y = positions[3][1] + block_h + gap / 2
ax.plot([sec1_x - 0.05, sec1_x + sec1_w + 0.05], [sep_y, sep_y],
        linestyle="--", color=C_DASHED, linewidth=1.0, zorder=1)
ax.text(sec1_x + sec1_w + 0.12, sep_y + 0.15, "Suffix\n(target)",
        fontsize=6, color=C_DASHED, ha="left", va="center")
ax.text(sec1_x + sec1_w + 0.12, sep_y - 0.18, "Prefix\n(frozen)",
        fontsize=6, color=C_DASHED, ha="left", va="center")

# Input arrow
ax.annotate("", xy=(block_x + block_w / 2, positions[0][1]),
            xytext=(block_x + block_w / 2, positions[0][1] - 0.45),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))
ax.text(block_x + block_w / 2, positions[0][1] - 0.55,
        r"$\mathbf{x}$", ha="center", va="top", fontsize=10, color=C_TEXT)

# Output arrow
top_y = positions[-1][1] + block_h
ax.annotate("", xy=(block_x + block_w / 2, top_y + 0.45),
            xytext=(block_x + block_w / 2, top_y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2))
ax.text(block_x + block_w / 2, top_y + 0.55,
        r"$z^T(\mathbf{x})\in\mathbb{R}^V$",
        ha="center", va="bottom", fontsize=8, color=C_TEXT)

# ============================================================
# SECTION 2 – Observability Analysis   x ∈ [3.3, 6.5]
# ============================================================
sec2_x = 3.6
sec2_w = 3.0

ax.text(sec2_x + sec2_w / 2, 5.25, r"$\bf{Observability\ Analysis}$",
        ha="center", va="center", fontsize=10, color=C_TEXT)

panel2 = FancyBboxPatch((sec2_x - 0.15, 0.15), sec2_w + 0.3, 4.95,
                         boxstyle="round,pad=0.08", facecolor="#eceff4",
                         edgecolor="#b0b8c8", linewidth=0.6, alpha=0.5, zorder=0)
ax.add_patch(panel2)

# Arrow from teacher to middle
arr_start_x = sec1_x + sec1_w + 0.15
arr_end_x = sec2_x - 0.15
arr_y = 3.5
ax.annotate("", xy=(arr_end_x, arr_y), xytext=(arr_start_x, arr_y),
            arrowprops=dict(arrowstyle="-|>", color=C_BRIGHT, lw=1.5,
                            connectionstyle="arc3,rad=0"))
ax.text((arr_start_x + arr_end_x) / 2, arr_y + 0.18,
        r"Query logits $z^T(\mathbf{x})$",
        ha="center", va="bottom", fontsize=7, color=C_TEXT,
        style="italic")

# Gramian equation box
gram_cx = sec2_x + sec2_w / 2
gram_y = 3.8
gram_box = FancyBboxPatch((sec2_x + 0.2, gram_y - 0.35), sec2_w - 0.4, 0.9,
                           boxstyle="round,pad=0.06", facecolor="white",
                           edgecolor=C_EIGEN_HI, linewidth=1.0, zorder=2)
ax.add_patch(gram_box)
ax.text(gram_cx, gram_y + 0.22, r"$\bf{Fisher\ Information\ Gramian}$",
        ha="center", va="center", fontsize=7.5, color=C_EIGEN_HI, zorder=3)
ax.text(gram_cx, gram_y - 0.08,
        r"$G(\mathcal{Q}) = \frac{1}{N}\sum_{i} J_i^\top J_i$",
        ha="center", va="center", fontsize=9, color=C_TEXT, zorder=3)

# --- Eigenspectrum: Block 23 ---
eigen_title_y = 3.05
bar_base_y = 2.15
bar_h_max = 0.7
n_bars_nonzero = 8
n_bars_zero = 6
bar_w = 0.12
bar_gap = 0.04
total_bars = n_bars_nonzero + n_bars_zero
bar_total_w = total_bars * (bar_w + bar_gap)
bar_start_x = sec2_x + 0.35

ax.text(bar_start_x + bar_total_w / 2, eigen_title_y,
        r"$\bf{Block\ 23}$  eigenspectrum",
        ha="center", va="center", fontsize=7, color=C_TEXT)

# nonzero bars (decreasing)
for i in range(n_bars_nonzero):
    h = bar_h_max * np.exp(-0.25 * i)
    bx = bar_start_x + i * (bar_w + bar_gap)
    rect = Rectangle((bx, bar_base_y), bar_w, h, facecolor=C_EIGEN_HI,
                      edgecolor="white", linewidth=0.3, zorder=2)
    ax.add_patch(rect)

# zero bars
for i in range(n_bars_zero):
    bx = bar_start_x + (n_bars_nonzero + i) * (bar_w + bar_gap)
    rect = Rectangle((bx, bar_base_y), bar_w, 0.03, facecolor=C_EIGEN_LO,
                      edgecolor="#b0b8c8", linewidth=0.3, zorder=2)
    ax.add_patch(rect)

ax.text(bar_start_x + bar_total_w / 2, bar_base_y - 0.18,
        r"rank 32 / 14.9M params",
        ha="center", va="top", fontsize=6.5, color=C_EIGEN_HI,
        fontweight="bold")

# --- Eigenspectrum: Block 22 ---
eigen2_title_y = 1.7
bar2_base_y = 1.05

ax.text(bar_start_x + bar_total_w / 2, eigen2_title_y,
        r"$\bf{Block\ 22}$  eigenspectrum",
        ha="center", va="center", fontsize=7, color=C_TEXT)

for i in range(total_bars):
    bx = bar_start_x + i * (bar_w + bar_gap)
    rect = Rectangle((bx, bar2_base_y), bar_w, 0.03, facecolor=C_EIGEN_LO,
                      edgecolor="#b0b8c8", linewidth=0.3, zorder=2)
    ax.add_patch(rect)

ax.text(bar_start_x + bar_total_w / 2, bar2_base_y - 0.18,
        r"rank 0 (zero Gramian)",
        ha="center", va="top", fontsize=6.5, color="#999999",
        fontweight="bold")

# Axis labels for bar charts
ax.text(bar_start_x - 0.08, bar_base_y + bar_h_max / 2,
        r"$\lambda_i$", ha="right", va="center", fontsize=7, color=C_TEXT)
ax.text(bar_start_x - 0.08, bar2_base_y + 0.2,
        r"$\lambda_i$", ha="right", va="center", fontsize=7, color=C_TEXT)

# Small arrow connecting the two bar chart sections
ax.annotate("", xy=(bar_start_x + bar_total_w / 2, eigen2_title_y + 0.12),
            xytext=(bar_start_x + bar_total_w / 2, bar_base_y - 0.28),
            arrowprops=dict(arrowstyle="-", color="#b0b8c8", lw=0.5,
                            linestyle="--"))

# ============================================================
# SECTION 3 – Student (Recovered)   x ∈ [7.2, 9.5]
# ============================================================
sec3_x = 7.6
sec3_w = 1.8

ax.text(sec3_x + sec3_w / 2, 5.25, r"$\bf{Student\ (Recovered)}$",
        ha="center", va="center", fontsize=10, color=C_TEXT)

panel3 = FancyBboxPatch((sec3_x - 0.15, 0.15), sec3_w + 0.3, 4.95,
                         boxstyle="round,pad=0.08", facecolor="#eceff4",
                         edgecolor="#b0b8c8", linewidth=0.6, alpha=0.5, zorder=0)
ax.add_patch(panel3)

# Arrow from middle to right
arr2_start_x = sec2_x + sec2_w + 0.15
arr2_end_x = sec3_x - 0.15
arr2_y = 3.5
ax.annotate("", xy=(arr2_end_x, arr2_y), xytext=(arr2_start_x, arr2_y),
            arrowprops=dict(arrowstyle="-|>", color=C_BRIGHT, lw=1.5))
ax.text((arr2_start_x + arr2_end_x) / 2, arr2_y + 0.18,
        r"S-PSI",
        ha="center", va="bottom", fontsize=7, color=C_TEXT,
        fontweight="bold")
ax.text((arr2_start_x + arr2_end_x) / 2, arr2_y - 0.15,
        "optimization",
        ha="center", va="top", fontsize=6.5, color=C_TEXT,
        style="italic")

# Student blocks
s_block_x = sec3_x + (sec3_w - block_w) / 2
s_labels = ["Block 1", "Block 2", r"$\cdots$", "Block 22", "Block 23", "lm_head"]
s_colors = [C_FADED, C_FADED, C_FADED, C_FADED, C_FADED, C_BRIGHT]
s_fontcolors = ["#999999", "#999999", "#999999", "#999999", "#999999", "white"]
s_annotations = [None, None, None, r"cos $\approx$ 0", r"cos $\approx$ 0", r"cos $\approx$ 0.54"]
s_ann_colors = [None, None, None, "#999999", "#999999", C_BRIGHT2]

s_positions = []
for i, (lbl, col, fc) in enumerate(zip(s_labels, s_colors, s_fontcolors)):
    y = y_start + i * (block_h + gap)
    if lbl == r"$\cdots$":
        ax.text(s_block_x + block_w / 2, y + block_h / 2, lbl,
                ha="center", va="center", fontsize=9, color="#999999", zorder=3)
        s_positions.append((s_block_x, y, block_w, block_h))
        continue

    ls = "-" if col != C_FADED else (0, (3, 3))
    edge = "#b0b8c8" if col == C_FADED else "#4c566a"
    draw_block(ax, s_block_x, y, block_w, block_h, lbl,
               facecolor=col, fontcolor=fc, edgecolor=edge,
               linestyle=ls if isinstance(ls, str) else ls)
    s_positions.append((s_block_x, y, block_w, block_h))

    # Cross-out for unrecoverable
    if col == C_FADED and lbl not in [r"$\cdots$"]:
        cx = s_block_x + block_w / 2
        cy = y + block_h / 2
        d = 0.12
        ax.plot([cx - d, cx + d], [cy - d, cy + d], color="#bf616a",
                lw=0.8, zorder=4, alpha=0.6)
        ax.plot([cx - d, cx + d], [cy + d, cy - d], color="#bf616a",
                lw=0.8, zorder=4, alpha=0.6)

# Annotations
for i, (ann, anncol) in enumerate(zip(s_annotations, s_ann_colors)):
    if ann is None:
        continue
    bx, by, bw, bh = s_positions[i]
    ax.text(bx + bw + 0.08, by + bh / 2, ann,
            ha="left", va="center", fontsize=6.5, color=anncol,
            fontweight="bold", zorder=5)

# ============================================================
# BOTTOM LEGEND
# ============================================================
legend_y = -0.15
leg_items = [
    (C_DARK,   "Unobservable"),
    (C_MID,    "Partially observable"),
    (C_BRIGHT, "Recoverable"),
    (C_FADED,  "Unrecoverable (student)"),
]
leg_start_x = 1.5
leg_spacing = 2.3

for i, (col, label) in enumerate(leg_items):
    lx = leg_start_x + i * leg_spacing
    rect = FancyBboxPatch((lx, legend_y - 0.1), 0.28, 0.22,
                           boxstyle="round,pad=0.02", facecolor=col,
                           edgecolor="#4c566a", linewidth=0.5, zorder=2)
    ax.add_patch(rect)
    ax.text(lx + 0.38, legend_y + 0.01, label,
            ha="left", va="center", fontsize=7, color=C_TEXT)

# ============================================================
# Save
# ============================================================
out = "/Users/niewenhua/Desktop/aicoding/github_repos/nips-modelsteal/paper/figures/framework_overview.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.08, dpi=300)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.08, dpi=300)
print(f"Saved → {out}")
print(f"Saved → {out.replace('.pdf', '.png')}")
plt.close(fig)
