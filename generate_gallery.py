"""
generate_gallery.py — SciVizKit showcase figure generator
Generates 8 publication-quality example figures for the README gallery.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

os.makedirs("assets/gallery", exist_ok=True)

# Nature journal color palette
NATURE = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"]

np.random.seed(42)

# ── 1. Violin + box: Gene expression ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

cell_types = ['CD4+ T cells', 'CD8+ T cells', 'B cells', 'NK cells']
data = [np.random.lognormal(mean, 0.5, 120) for mean in [2.1, 2.8, 1.6, 3.2]]

parts = ax.violinplot(data, positions=range(1, 5), showmedians=False, showextrema=False)
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(NATURE[i])
    pc.set_alpha(0.7)
    pc.set_edgecolor('white')

bp = ax.boxplot(data, positions=range(1, 5), widths=0.12,
                patch_artist=True, medianprops=dict(color='white', linewidth=2))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor(NATURE[i])
    patch.set_alpha(0.9)
for element in ['whiskers', 'caps', 'fliers']:
    for item in bp[element]:
        item.set(color='#555555', linewidth=1)

ax.set_xticks(range(1, 5))
ax.set_xticklabels(cell_types, fontsize=12)
ax.set_ylabel('Expression (log₂ CPM)', fontsize=13)
ax.set_title('Gene Expression by Cell Type', fontsize=14, fontweight='bold', pad=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#dddddd')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig('assets/gallery/01_violin_distribution.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/01_violin_distribution.png")

# ── 2. Clustered heatmap: material properties ────────────────────────────────
props = ['Conductivity', 'Hardness', 'Density', 'Melting Pt', 'Ductility',
         'Tensile Str', 'Yield Str', 'Corrosion R', 'Thermal Exp', 'Elasticity']
corr = np.random.uniform(-1, 1, (10, 10))
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, xticklabels=props, yticklabels=props,
            cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', annot_kws={'size': 8},
            linewidths=0.5, linecolor='white', ax=ax,
            cbar_kws={'shrink': 0.8, 'label': 'Pearson r'})
ax.set_title('Material Property Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('assets/gallery/02_heatmap_correlation.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/02_heatmap_correlation.png")

# ── 3. Volcano plot: DEG ────────────────────────────────────────────────────
n = 800
log2fc = np.random.normal(0, 1.2, n)
pvals = np.random.uniform(0, 1, n)
pvals[np.abs(log2fc) > 2] *= 0.001
neg_log_p = -np.log10(pvals + 1e-10)

up   = (log2fc > 1) & (neg_log_p > 2)
down = (log2fc < -1) & (neg_log_p > 2)
ns   = ~(up | down)

fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.scatter(log2fc[ns],   neg_log_p[ns],   c='#AAAAAA', alpha=0.4, s=15, linewidths=0)
ax.scatter(log2fc[up],   neg_log_p[up],   c=NATURE[0], alpha=0.8, s=20, linewidths=0, label=f'Up ({up.sum()})')
ax.scatter(log2fc[down], neg_log_p[down], c=NATURE[1], alpha=0.8, s=20, linewidths=0, label=f'Down ({down.sum()})')
ax.axhline(2, color='#666', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(1,  color='#666', linestyle='--', linewidth=1, alpha=0.7)
ax.axvline(-1, color='#666', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('log₂ Fold Change', fontsize=13)
ax.set_ylabel('−log₁₀(p-value)', fontsize=13)
ax.set_title('Volcano Plot — Differential Gene Expression', fontsize=14, fontweight='bold', pad=12)
ax.legend(fontsize=11, framealpha=0.9)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#dddddd')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig('assets/gallery/03_volcano_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/03_volcano_plot.png")

# ── 4. Radar chart: material comparison ─────────────────────────────────────
categories = ['Strength', 'Ductility', 'Corrosion\nResistance', 'Conductivity', 'Cost\nEfficiency', 'Weldability']
materials  = ['Steel 316L', 'Titanium', 'Aluminum', 'Inconel 718']
values_all = [
    [8, 6, 7, 4, 8, 7],
    [7, 8, 9, 3, 4, 5],
    [5, 9, 6, 8, 9, 8],
    [9, 5, 8, 3, 3, 4],
]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f9f9f9')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8, color='grey')
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.xaxis.grid(True, linestyle='--', alpha=0.5)

for i, (mat, vals) in enumerate(zip(materials, values_all)):
    v = vals + vals[:1]
    ax.plot(angles, v, 'o-', linewidth=2, color=NATURE[i], label=mat)
    ax.fill(angles, v, alpha=0.12, color=NATURE[i])

ax.set_title('Multi-Property Material Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
plt.tight_layout()
plt.savefig('assets/gallery/04_radar_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/04_radar_chart.png")

# ── 5. Kaplan-Meier survival curves ─────────────────────────────────────────
def km_curve(n, hazard, max_t=60):
    times = np.sort(np.random.exponential(1/hazard, n))
    t_pts = np.linspace(0, max_t, 200)
    surv  = np.array([np.mean(times > t) for t in t_pts])
    return t_pts, surv

groups   = ['Control', 'Low Dose', 'Mid Dose', 'High Dose']
hazards  = [0.04, 0.03, 0.02, 0.012]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
for i, (grp, hz) in enumerate(zip(groups, hazards)):
    t, s = km_curve(100, hz)
    ax.step(t, s, where='post', color=NATURE[i], linewidth=2, label=grp)
    noise = s + np.random.normal(0, 0.015, len(s))
    ax.fill_between(t, np.clip(noise - 0.06, 0, 1), np.clip(noise + 0.06, 0, 1),
                    alpha=0.1, color=NATURE[i], step='post')

ax.axhline(0.5, color='#888', linestyle=':', linewidth=1)
ax.set_xlabel('Time (months)', fontsize=13)
ax.set_ylabel('Survival Probability', fontsize=13)
ax.set_title('Kaplan-Meier Survival Analysis', fontsize=14, fontweight='bold', pad=12)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, framealpha=0.9)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#dddddd')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig('assets/gallery/05_survival_curve.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/05_survival_curve.png")

# ── 6. Network graph: protein interactions ───────────────────────────────────
G = nx.barabasi_albert_graph(22, 2, seed=42)
hubs = sorted(G.degree, key=lambda x: x[1], reverse=True)[:4]
hub_nodes = {n for n, _ in hubs}

node_colors = []
node_sizes  = []
for n in G.nodes():
    deg = G.degree(n)
    if n in hub_nodes:
        node_colors.append(NATURE[0])
        node_sizes.append(600 + deg * 60)
    else:
        node_colors.append(NATURE[1])
        node_sizes.append(200 + deg * 30)

pos = nx.spring_layout(G, seed=7, k=1.2)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color='#888888', width=1.2)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)
labels = {n: f'P{n+1}' for n in G.nodes()}
nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_color='white', font_weight='bold')

legend_els = [
    mpatches.Patch(facecolor=NATURE[0], label='Hub proteins'),
    mpatches.Patch(facecolor=NATURE[1], label='Interactors'),
]
ax.legend(handles=legend_els, fontsize=11, loc='upper left', framealpha=0.9)
ax.set_title('Protein–Protein Interaction Network', fontsize=14, fontweight='bold', pad=12)
ax.axis('off')
plt.tight_layout()
plt.savefig('assets/gallery/06_network_graph.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/06_network_graph.png")

# ── 7. 3D scatter: PCA visualization ────────────────────────────────────────
class_labels = ['Class A', 'Class B', 'Class C']
centers = [(0, 0, 0), (4, 4, 2), (-3, 3, 4)]
all_pts, all_cls = [], []
for i, (cx, cy, cz) in enumerate(centers):
    pts = np.random.randn(60, 3) + np.array([cx, cy, cz])
    all_pts.append(pts)
    all_cls.extend([i]*60)

fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')
for i, (pts, lbl) in enumerate(zip(all_pts, class_labels)):
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=NATURE[i], s=30, alpha=0.8, label=lbl, linewidths=0)

ax.set_xlabel('PC1 (38.2%)', fontsize=11, labelpad=8)
ax.set_ylabel('PC2 (24.7%)', fontsize=11, labelpad=8)
ax.set_zlabel('PC3 (14.1%)', fontsize=11, labelpad=8)
ax.set_title('PCA — 3D Score Plot', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('assets/gallery/07_3d_scatter.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/07_3d_scatter.png")

# ── 8. Multi-panel publication figure ───────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.patch.set_facecolor('white')
fig.suptitle('Multi-Panel Publication Figure', fontsize=15, fontweight='bold', y=1.01)

# (a) Bar chart
ax = axes[0, 0]
ax.set_facecolor('white')
cats = ['Sample A', 'Sample B', 'Sample C', 'Sample D']
vals = [3.2, 5.8, 4.1, 7.3]
errs = [0.3, 0.4, 0.25, 0.5]
bars = ax.bar(cats, vals, yerr=errs, capsize=5, color=NATURE[:4], alpha=0.85, edgecolor='white')
ax.set_ylabel('Activity (μmol/min)', fontsize=11)
ax.set_title('(a) Enzyme Activity', fontsize=12, fontweight='bold')
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#dddddd')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# (b) Line plot
ax = axes[0, 1]
ax.set_facecolor('white')
x = np.linspace(0, 10, 100)
for i, (lbl, phase) in enumerate(zip(['Treated', 'Control', 'Inhibited'], [0, np.pi/3, np.pi*0.7])):
    ax.plot(x, np.sin(x - phase) * np.exp(-x * 0.05) + i * 0.2,
            color=NATURE[i], linewidth=2, label=lbl)
ax.set_xlabel('Time (h)', fontsize=11)
ax.set_ylabel('Signal (a.u.)', fontsize=11)
ax.set_title('(b) Time-Course Signal', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.9)
ax.yaxis.grid(True, linestyle='--', alpha=0.4)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# (c) Scatter with regression
ax = axes[1, 0]
ax.set_facecolor('white')
x = np.random.uniform(1, 10, 80)
y = 2.3 * x + np.random.normal(0, 1.5, 80)
ax.scatter(x, y, c=NATURE[3], alpha=0.6, s=25, linewidths=0)
m, b = np.polyfit(x, y, 1)
xfit = np.linspace(1, 10, 100)
ax.plot(xfit, m * xfit + b, color=NATURE[0], linewidth=2, linestyle='--')
ax.text(0.05, 0.92, f'r² = 0.91', transform=ax.transAxes, fontsize=11,
        bbox=dict(facecolor='white', edgecolor='#ccc', boxstyle='round,pad=0.3'))
ax.set_xlabel('Concentration (mM)', fontsize=11)
ax.set_ylabel('Response (mV)', fontsize=11)
ax.set_title('(c) Dose–Response Regression', fontsize=12, fontweight='bold')
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# (d) Stacked bar
ax = axes[1, 1]
ax.set_facecolor('white')
samples = ['S1', 'S2', 'S3', 'S4', 'S5']
comp = np.array([
    [40, 30, 20, 25, 35],
    [25, 35, 30, 20, 25],
    [20, 20, 35, 30, 20],
    [15, 15, 15, 25, 20],
])
bottom = np.zeros(5)
lbls = ['Phase α', 'Phase β', 'Phase γ', 'Amorphous']
for i, (row, lbl) in enumerate(zip(comp, lbls)):
    ax.bar(samples, row, bottom=bottom, label=lbl, color=NATURE[i], alpha=0.85, edgecolor='white')
    bottom += row
ax.set_ylabel('Composition (%)', fontsize=11)
ax.set_title('(d) Phase Composition', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('assets/gallery/08_publication_panel.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/08_publication_panel.png")

print("\n✅ All 8 gallery figures generated successfully!")
