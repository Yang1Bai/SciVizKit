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

# ── 9. Radial Bar + Significance (环形柱状图 + 误差线) — NGplot-inspired ───────
np.random.seed(3)
rb_groups = ['WT', 'KO-1', 'KO-2', 'OE-1', 'OE-2', 'KD']
rb_data   = [np.random.normal(m, 0.6, 20)
             for m in [5.2, 3.8, 3.5, 7.1, 6.8, 4.4]]
rb_means  = np.array([d.mean() for d in rb_data])
rb_sems   = np.array([d.std() / np.sqrt(len(d)) for d in rb_data])
n_rb      = len(rb_groups)

angles_rb  = np.linspace(0, 2 * np.pi, n_rb, endpoint=False)
bar_w_rb   = (2 * np.pi) / n_rb * 0.65
rb_palette = [NATURE[i % len(NATURE)] for i in range(n_rb)]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'projection': 'polar'})
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafafa')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

max_rb = (rb_means + rb_sems).max() * 1.25
ax.set_ylim(0, max_rb)

for i, (angle, mean_v, sem_v, grp) in enumerate(
        zip(angles_rb, rb_means, rb_sems, rb_groups)):
    ax.bar(angle, mean_v, width=bar_w_rb,
           color=rb_palette[i], alpha=0.85, zorder=2)
    # Error bar (SEM)
    ax.plot([angle, angle], [mean_v - sem_v, mean_v + sem_v],
            color='#333333', lw=2, zorder=4)
    ax.plot([angle - bar_w_rb*0.15, angle + bar_w_rb*0.15],
            [mean_v + sem_v, mean_v + sem_v],
            color='#333333', lw=2, zorder=4)

ax.set_xticks(angles_rb)
ax.set_xticklabels(rb_groups, fontsize=11, fontweight='bold')
ax.set_yticks(np.arange(0, max_rb, 2))
ax.set_yticklabels([f'{v:.0f}' for v in np.arange(0, max_rb, 2)],
                    fontsize=8, color='grey')
ax.yaxis.set_tick_params(labelleft=True)
ax.grid(True, linestyle='--', alpha=0.4, color='grey')

# ANOVA annotation
try:
    from scipy.stats import f_oneway as _fow
    _, p_rb = _fow(*rb_data)
    stars_rb = '***' if p_rb < 0.001 else '**' if p_rb < 0.01 else '*' if p_rb < 0.05 else 'ns'
    sig_note = f'ANOVA: p={p_rb:.2e}  {stars_rb}'
except Exception:
    sig_note = ''

ax.set_title(f'Gene Expression by Genotype\nRadial Bar + Significance  {sig_note}',
             pad=20, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/gallery/09_radial_bar_sig.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: assets/gallery/09_radial_bar_sig.png')

# ── Jade Ring (kept, separate file not in README) ───────────────────────────
jr_data = [
    ('Proteobacteria',  38.4),
    ('Firmicutes',      27.1),
    ('Bacteroidetes',   18.6),
    ('Actinobacteria',  10.2),
    ('Verrucomicrobia',  3.9),
    ('Fusobacteria',     1.8),
]
# Sort ascending so largest value is on outermost ring
jr_data_sorted = sorted(jr_data, key=lambda x: x[1])
categories_jr = [x[0] for x in jr_data_sorted]
abundances_jr = [x[1] for x in jr_data_sorted]
n_jr = len(categories_jr)
max_val_jr = max(abundances_jr)

# Jade-green gradient palette
jr_colors = ['#A8D8B9', '#7FBF7F', '#50C878', '#2E8B57', '#3CB371', '#00A86B']

fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={'projection': 'polar'})
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_axis_off()

ring_width_jr = 0.50 / (n_jr + 1)
base_r_jr = 0.20

for i, (cat, val) in enumerate(zip(categories_jr, abundances_jr)):
    radius = base_r_jr + i * (ring_width_jr + 0.05)
    fraction = val / max_val_jr   # arc = relative to max (jade ring convention)
    theta_end = fraction * 2 * np.pi

    theta_fill = np.linspace(0, theta_end, 500)
    ax.fill_between(theta_fill, radius, radius + ring_width_jr,
                    color=jr_colors[i], alpha=0.90)
    theta_bg = np.linspace(theta_end, 2 * np.pi, 100)
    ax.fill_between(theta_bg, radius, radius + ring_width_jr,
                    color='#e8f5e9', alpha=0.7)

    # % label at arc end-point
    label_r = radius + ring_width_jr / 2
    if fraction > 0.08:
        mid_a = theta_end / 2
        ax.text(mid_a, label_r,
                f'{val:.1f}%', ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

# Legend with sorted order (largest first for readability)
leg_sorted = sorted(zip(categories_jr, abundances_jr, jr_colors), key=lambda x: -x[1])
leg_handles = [mpatches.Patch(color=c, label=f'{cat}  {val:.1f}%')
               for cat, val, c in leg_sorted]
ax.legend(handles=leg_handles, loc='lower center', bbox_to_anchor=(0.5, -0.08),
          ncol=2, fontsize=9.5, frameon=False, handlelength=1.2)

ax.set_title('Gut Microbiome Composition\nJade Ring Chart',
             pad=22, fontsize=13, fontweight='bold', color='#1a5c35')
plt.tight_layout()
plt.savefig('assets/gallery/09b_jade_ring.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/09b_jade_ring.png")

# ── 10. Bar + Significance Brackets ─────────────────────────────────────────
np.random.seed(7)
groups_sig = ['Control', 'Drug A', 'Drug B', 'Drug C']
data_sig = [np.random.normal(m, 1.2, 25) for m in [5.0, 7.8, 9.2, 6.4]]

means_s = [d.mean() for d in data_sig]
sems_s  = [d.std() / np.sqrt(len(d)) for d in data_sig]

palette_s = [NATURE[i] for i in range(4)]
fig, ax = plt.subplots(figsize=(7, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

bars_s = ax.bar(range(4), means_s, yerr=sems_s, capsize=6,
                color=palette_s, alpha=0.85, width=0.55,
                error_kw=dict(ecolor='#333333', lw=1.8, capthick=1.8),
                edgecolor='white', linewidth=1.5)

# Scatter individual points
for i, d in enumerate(data_sig):
    jitter = np.random.uniform(-0.15, 0.15, len(d))
    ax.scatter(i + jitter, d, color=palette_s[i], alpha=0.35, s=18, zorder=3, linewidths=0)

# Significance brackets
def draw_bracket(ax, x1, x2, y, label, color='black'):
    h = 0.18
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], color=color, lw=1.4)
    ax.text((x1+x2)/2, y+h+0.05, label, ha='center', va='bottom',
            fontsize=13, color='red' if label != 'ns' else 'grey')

y_top = max(means_s[i] + sems_s[i] for i in range(4))
draw_bracket(ax, 0, 2, y_top + 0.6, '***')
draw_bracket(ax, 1, 2, y_top + 1.5, '*')
draw_bracket(ax, 0, 3, y_top + 2.5, 'ns')

ax.set_xticks(range(4))
ax.set_xticklabels(groups_sig, fontsize=12)
ax.set_ylabel('Tumor Volume (mm³)', fontsize=12)
ax.set_title('Drug Treatment Effect\nBar + Significance Brackets', fontsize=13, fontweight='bold', pad=12)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#dddddd')
ax.set_axisbelow(True)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
plt.tight_layout()
plt.savefig('assets/gallery/10_bar_significance.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/10_bar_significance.png")

# ── 11. Ridgeline Plot (Joy Plot) ────────────────────────────────────────────
from scipy.stats import gaussian_kde

cell_lines = ['MCF-7', 'HeLa', 'A549', 'HCT116', 'PC-3', 'U87-MG']
ridgecolors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']

np.random.seed(15)
ridge_data = [np.random.normal(loc, scale, 500)
              for loc, scale in [(4.2, 1.1), (5.8, 0.9), (3.5, 1.4),
                                  (6.2, 0.8), (4.9, 1.3), (7.1, 0.7)]]

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

overlap = 1.5
x_range = np.linspace(-1, 12, 400)

for i, (cell, d, col) in enumerate(zip(cell_lines, ridge_data, ridgecolors)):
    kde = gaussian_kde(d, bw_method=0.35)
    y = kde(x_range)
    y_scaled = y / y.max() * overlap
    baseline = i * 1.0

    ax.fill_between(x_range, baseline, baseline + y_scaled,
                    color=col, alpha=0.75)
    ax.plot(x_range, baseline + y_scaled, color=col, lw=1.5, alpha=0.9)
    ax.axhline(baseline, color='white', lw=0.8, alpha=0.6)
    ax.text(-0.8, baseline + 0.15, cell, ha='right', va='bottom',
            fontsize=10.5, fontweight='bold', color='#333333')

ax.set_xlabel('mRNA Expression (log₂ TPM)', fontsize=12)
ax.set_xlim(-1, 11)
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_visible(True)
ax.set_title('Expression Distribution Across Cell Lines\nRidgeline Plot', fontsize=13, fontweight='bold', pad=14)
plt.tight_layout()
plt.savefig('assets/gallery/11_ridgeline.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: assets/gallery/11_ridgeline.png")

# ── 12. Sankey / Alluvial Diagram ────────────────────────────────────────────
try:
    import plotly.graph_objects as go_s

    label_s = ['Stage I', 'Stage II', 'Stage III', 'Stage IV',
               'Complete\nResponse', 'Partial\nResponse', 'Stable\nDisease', 'Progression']
    source_s = [0, 0, 1, 1, 2, 2, 3, 3]
    target_s = [4, 5, 5, 6, 6, 7, 7, 4]
    value_s  = [45, 15, 30, 25, 20, 35, 40, 10]
    color_links = ['rgba(230,75,53,0.4)', 'rgba(77,187,213,0.4)',
                   'rgba(77,187,213,0.4)', 'rgba(0,160,135,0.4)',
                   'rgba(0,160,135,0.4)', 'rgba(60,84,136,0.4)',
                   'rgba(60,84,136,0.4)', 'rgba(230,75,53,0.4)']

    fig_s2 = go_s.Figure(go_s.Sankey(
        node=dict(
            pad=15, thickness=20, line=dict(color='white', width=0.5),
            label=label_s,
            color=['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
                   '#91D1C2', '#F39B7F', '#8491B4', '#DC0000'],
        ),
        link=dict(source=source_s, target=target_s, value=value_s, color=color_links),
    ))
    fig_s2.update_layout(
        title_text='Cancer Stage → Treatment Response<br>Sankey Diagram',
        title_font_size=14,
        font_size=12,
        width=800, height=500,
        paper_bgcolor='white',
    )
    fig_s2.write_image('assets/gallery/12_sankey.png', scale=1.5)
    print("Saved: assets/gallery/12_sankey.png")
except Exception:
    # Fallback: proper Bezier-ribbon Sankey with matplotlib
    from matplotlib.path import Path
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Left nodes (stages) and right nodes (outcomes)
    left_lbls  = ['Stage I',  'Stage II', 'Stage III', 'Stage IV']
    right_lbls = ['Complete\nResponse', 'Partial\nResponse', 'Stable\nDisease', 'Progression']
    # flows[i][j] = n patients from stage i to outcome j
    flows_mat = np.array([
        [45, 10,  5,  0],
        [10, 25, 15,  5],
        [ 5, 10, 20, 20],
        [ 0,  5, 10, 35],
    ], dtype=float)

    left_totals  = flows_mat.sum(axis=1)
    right_totals = flows_mat.sum(axis=0)
    total = flows_mat.sum()

    margin, node_w = 0.02, 0.03
    left_x, right_x = 0.18, 0.78
    gap = 0.025

    def node_ys(totals, margin=0.05):
        """Return (bottom_y, top_y) for each node stacked vertically."""
        scale = (1 - 2*margin - gap*(len(totals)-1)) / totals.sum()
        ys = []
        y = margin
        for t in totals:
            h = t * scale
            ys.append((y, y + h))
            y += h + gap
        return ys

    left_ys  = node_ys(left_totals)
    right_ys = node_ys(right_totals)

    lc = [NATURE[i] for i in range(4)]
    rc = ['#91D1C2', '#F39B7F', '#8491B4', '#DC0000']

    # Draw Bezier ribbons
    left_cursors  = [y0 for y0, _ in left_ys]
    right_cursors = [y0 for y0, _ in right_ys]
    scale = (1 - 2*0.05 - gap*3) / total

    for i in range(4):
        for j in range(4):
            val = flows_mat[i, j]
            if val == 0:
                continue
            h = val * scale
            y0_l, y1_l = left_cursors[i], left_cursors[i] + h
            y0_r, y1_r = right_cursors[j], right_cursors[j] + h
            left_cursors[i]  += h
            right_cursors[j] += h

            verts = [
                (left_x + node_w, y0_l), (0.5, y0_l),
                (0.5, y0_r), (right_x, y0_r),
                (right_x, y1_r), (0.5, y1_r),
                (0.5, y1_l), (left_x + node_w, y1_l),
                (left_x + node_w, y0_l),
            ]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
                     Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.CLOSEPOLY]
            path  = Path(verts, codes)
            patch = mpatches.FancyArrowPatch.__new__(mpatches.FancyArrowPatch)
            patch = mpatches.PathPatch(path, facecolor=lc[i], alpha=0.35, linewidth=0)
            ax.add_patch(patch)

    # Draw nodes
    for i, ((y0, y1), lbl) in enumerate(zip(left_ys, left_lbls)):
        ax.add_patch(plt.Rectangle((left_x, y0), node_w, y1-y0, color=lc[i], zorder=3))
        ax.text(left_x - 0.02, (y0+y1)/2, lbl, ha='right', va='center',
                fontsize=10, fontweight='bold', color='#333333')
    for j, ((y0, y1), lbl) in enumerate(zip(right_ys, right_lbls)):
        ax.add_patch(plt.Rectangle((right_x, y0), node_w, y1-y0, color=rc[j], zorder=3))
        ax.text(right_x + node_w + 0.02, (y0+y1)/2, lbl, ha='left', va='center',
                fontsize=10, fontweight='bold', color='#333333')

    ax.set_title('Cancer Stage to Treatment Response\nSankey Flow Diagram',
                 fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('assets/gallery/12_sankey.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print('Saved: assets/gallery/12_sankey.png (bezier fallback)')

print("\n✅ All new gallery figures (09–12) generated!")
