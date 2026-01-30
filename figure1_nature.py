"""
Figure 1 (Nature style): Source vs Top data-driven targets

Exports: Figure1.pdf (vector), Figure1.svg (vector), Figure1.png (600 dpi)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Nature-style global settings
# -----------------------------

mpl.rcParams.update({
    "font.family": "DejaVu Sans",      # swap to Arial if installed: "Arial"
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,                # embed fonts cleanly
    "ps.fonttype": 42,
})

# -----------------------------
# INPUT DATA (edit these)
# -----------------------------

# Panel A: counts
counts = {
    "Validated\n(both)": 4,
    "Literature\nonly": 40,
    "Data-driven\nonly": 96
}

# Panel B: Top targets (Nature main figure usually top 10–15; move full top 30 to Supplementary)
genes = [
    "MGP", "CXCL8", "MAOA", "APOD", "PPP1R14A",
    "STMN2", "SERPINA3", "ALDH1A3", "SAA1", "RGCC",
    "ACTG2", "DPT", "SLC40A1", "INMT", "ADH1A"
]
scores = [
    37.5, 24.8, 22.6, 21.5, 20.4,
    19.8, 19.2, 18.6, 17.9, 17.4,
    16.9, 16.6, 16.3, 16.0, 15.7
]
known = {"CXCL8", "ACTG2"}  # edit to your "Known (literature)" genes

# -----------------------------
# Styling (print-safe)
# -----------------------------

COL_KNOWN = "#B3B3B3"   # neutral gray
COL_NOVEL = "#4C72B0"   # muted blue (Nature-safe accent)
COL_BOTH  = "#6E6E6E"   # darker gray

# Figure size: Nature double-column width ~183 mm
mm_to_in = 1 / 25.4
fig_w = 183 * mm_to_in
fig_h = 70  * mm_to_in   # ~70 mm tall; tweak if needed

fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
gs = fig.add_gridspec(
    1, 2, width_ratios=[1.0, 1.8],
    left=0.06, right=0.99, top=0.92, bottom=0.18, wspace=0.28
)

# =============================
# Panel A
# =============================

axA = fig.add_subplot(gs[0, 0])

cats = list(counts.keys())
vals = [counts[c] for c in cats]
colorsA = [COL_BOTH, COL_KNOWN, COL_NOVEL]

bars = axA.bar(np.arange(len(cats)), vals, color=colorsA, width=0.62)

# Counts only (Nature: keep it clean)
for b in bars:
    axA.text(
        b.get_x() + b.get_width()/2, b.get_height() + max(vals)*0.02,
        f"{int(b.get_height())}",
        ha="center", va="bottom", fontsize=7
    )

axA.set_xticks(np.arange(len(cats)))
axA.set_xticklabels(cats)
axA.set_ylabel("Number of genes")
axA.set_title("Source of identified targets", pad=6)

# Minimal spines (Nature style)
axA.spines["top"].set_visible(False)
axA.spines["right"].set_visible(False)
axA.grid(False)

# Panel label
axA.text(-0.20, 1.05, "A", transform=axA.transAxes, fontweight="bold", fontsize=10, va="top")

# =============================
# Panel B
# =============================

axB = fig.add_subplot(gs[0, 1])

# Sort (descending)
order = np.argsort(scores)[::-1]
genes_s = [genes[i] for i in order]
scores_s = [scores[i] for i in order]
colorsB = [COL_KNOWN if g in known else COL_NOVEL for g in genes_s]

y = np.arange(len(genes_s))
axB.barh(y, scores_s, color=colorsB, height=0.72)

axB.set_yticks(y)
axB.set_yticklabels(genes_s)
axB.invert_yaxis()
axB.set_xlabel("Multi-omics dysregulation score")
axB.set_title("Top data-driven targets", pad=6)

# Nature-style axes
axB.spines["top"].set_visible(False)
axB.spines["right"].set_visible(False)
axB.spines["left"].set_visible(False)
axB.grid(axis="x", linestyle="-", linewidth=0.6, alpha=0.18)
axB.tick_params(axis="y", length=0)  # remove y tick marks for cleaner look

# Legend (simple, no frame)
legend = [
    Patch(facecolor=COL_NOVEL, edgecolor="none", label="Novel (data-driven)"),
    Patch(facecolor=COL_KNOWN, edgecolor="none", label="Known (literature)")
]
axB.legend(handles=legend, frameon=False, loc="lower right", fontsize=7, handlelength=1.2)

# Panel label
axB.text(-0.10, 1.05, "B", transform=axB.transAxes, fontweight="bold", fontsize=10, va="top")

# -----------------------------
# Export (Nature-friendly)
# -----------------------------

fig.savefig("Figure1.pdf")                 # vector
fig.savefig("Figure1.svg")                 # vector
fig.savefig("Figure1.png", dpi=600)        # high-res raster for submission systems

print("Exported: Figure1.pdf, Figure1.svg, Figure1.png (600 dpi)")
