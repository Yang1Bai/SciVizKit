"""
SciVizKit — Specialty generators (NGplot-inspired)
Charts unique to bioinformatics workflows, inspired by bioinforw.com/sciZ.
Each function returns (fig_static, fig_plotly, code_str).
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Wedge

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ── Jade Ring Chart (玉珏图) ──────────────────────────────────────────
# Concentric arc rings where each ring represents a group and the arc
# length encodes the numeric value. Commonly used in bioinformatics
# to compare relative abundances or feature values across samples.

def jade_ring_chart(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Jade Ring Chart (玉珏图): each group gets a concentric ring whose
    arc length is proportional to its mean value. The gap in each ring
    encodes the remaining proportion visually.
    """
    try:
        agg = df.groupby(group_col)[value_col].mean().reset_index()
        agg = agg.sort_values(value_col, ascending=False).reset_index(drop=True)
        n = len(agg)

        max_val = agg[value_col].max()
        if max_val == 0:
            return None, None, "# Error: all values are zero"

        # Color palette
        cmap = cm.get_cmap("tab20", n)
        colors = [cmap(i) for i in range(n)]

        fig_s, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_axis_off()

        ring_width = 0.6 / (n + 1)  # width of each arc ring
        base_radius = 0.3

        for i, row in agg.iterrows():
            radius = base_radius + i * (ring_width + 0.05)
            fraction = row[value_col] / max_val
            theta_end = fraction * 2 * np.pi

            # Draw arc (filled wedge via bar in polar)
            theta = np.linspace(0, theta_end, 300)
            ax.fill_between(theta,
                             radius,
                             radius + ring_width,
                             color=colors[i],
                             alpha=0.85,
                             label=str(row[group_col]))

            # Background arc (gap)
            theta_bg = np.linspace(theta_end, 2 * np.pi, 100)
            ax.fill_between(theta_bg,
                             radius,
                             radius + ring_width,
                             color="lightgrey",
                             alpha=0.3)

            # Label inside ring
            label_angle = theta_end / 2
            label_r = radius + ring_width / 2
            ax.text(label_angle, label_r,
                    f"{row[group_col]}\n{row[value_col]:.2f}",
                    ha="center", va="center", fontsize=7.5, fontweight="bold",
                    color="white" if fraction > 0.2 else "black")

        ax.set_title(f"Jade Ring Chart\n({value_col} by {group_col})",
                     pad=20, fontsize=13, fontweight="bold")
        fig_s.tight_layout()

        code = f"""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -- Data preparation --
agg = df.groupby('{group_col}')['{value_col}'].mean().reset_index()
agg = agg.sort_values('{value_col}', ascending=False).reset_index(drop=True)
n = len(agg)
max_val = agg['{value_col}'].max()

cmap = cm.get_cmap('tab20', n)
colors = [cmap(i) for i in range(n)]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={{'projection': 'polar'}})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_axis_off()

ring_width = 0.6 / (n + 1)
base_radius = 0.3

for i, row in agg.iterrows():
    radius = base_radius + i * (ring_width + 0.05)
    fraction = row['{value_col}'] / max_val
    theta_end = fraction * 2 * np.pi
    theta = np.linspace(0, theta_end, 300)
    ax.fill_between(theta, radius, radius + ring_width,
                    color=colors[i], alpha=0.85, label=str(row['{group_col}']))
    theta_bg = np.linspace(theta_end, 2 * np.pi, 100)
    ax.fill_between(theta_bg, radius, radius + ring_width,
                    color='lightgrey', alpha=0.3)

ax.set_title('Jade Ring Chart', pad=20, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('jade_ring.png', dpi=300, bbox_inches='tight')
plt.show()
"""
        return fig_s, None, code

    except Exception as e:
        return None, None, f"# Error: {e}"
