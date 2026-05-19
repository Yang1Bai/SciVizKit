"""
SciVizKit — 3D visualization generators
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.express as px
import plotly.graph_objects as go


# ── 3D Scatter Plot ───────────────────────────────────────────────────

def scatter_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
               color_col: str = None):
    """3D scatter plot using plotly and matplotlib."""
    try:
        clean = df[[x_col, y_col, z_col] +
                   ([color_col] if color_col and color_col in df.columns else [])].dropna()

        color_arg = color_col if (color_col and color_col in df.columns) else None

        # Plotly interactive
        fig_p = px.scatter_3d(clean, x=x_col, y=y_col, z=z_col, color=color_arg,
                              opacity=0.7, title=f"3D Scatter: {x_col}/{y_col}/{z_col}")
        fig_p.update_traces(marker=dict(size=4))

        # Matplotlib static
        fig_s = plt.figure(figsize=(9, 7))
        ax = fig_s.add_subplot(111, projection="3d")
        if color_arg:
            groups = clean[color_arg].unique()
            colors = plt.cm.tab20.colors
            for i, g in enumerate(groups):
                sub = clean[clean[color_arg] == g]
                ax.scatter(sub[x_col], sub[y_col], sub[z_col],
                           label=str(g), s=20, alpha=0.7,
                           color=colors[i % len(colors)])
            ax.legend(fontsize=7)
        else:
            ax.scatter(clean[x_col], clean[y_col], clean[z_col],
                       color="steelblue", s=20, alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Scatter: {x_col}/{y_col}/{z_col}")
        fig_s.tight_layout()

        code = f"""# 3D Scatter Plot
import plotly.express as px
fig = px.scatter_3d(df, x='{x_col}', y='{y_col}', z='{z_col}',
                    opacity=0.7, title='3D Scatter')
fig.update_traces(marker=dict(size=4))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── 3D Surface Plot ───────────────────────────────────────────────────

def surface_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str):
    """3D surface plot using plotly and matplotlib."""
    try:
        clean = df[[x_col, y_col, z_col]].dropna()

        # Try to create a grid for surface
        x_uniq = np.sort(clean[x_col].unique())
        y_uniq = np.sort(clean[y_col].unique())

        if len(x_uniq) >= 5 and len(y_uniq) >= 5:
            xi = np.linspace(clean[x_col].min(), clean[x_col].max(), 30)
            yi = np.linspace(clean[y_col].min(), clean[y_col].max(), 30)
            Xi, Yi = np.meshgrid(xi, yi)

            from scipy.interpolate import griddata
            Zi = griddata((clean[x_col], clean[y_col]), clean[z_col],
                          (Xi, Yi), method="linear")
        else:
            # Fallback: pivot table
            pivot = clean.pivot_table(index=y_col, columns=x_col,
                                      values=z_col, aggfunc="mean")
            Xi, Yi = np.meshgrid(pivot.columns, pivot.index)
            Zi = pivot.values

        # Plotly
        fig_p = go.Figure(go.Surface(
            x=Xi, y=Yi, z=Zi,
            colorscale="Viridis", showscale=True,
        ))
        fig_p.update_layout(
            title=f"3D Surface: {z_col}",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=z_col,
            ),
        )

        # Matplotlib
        fig_s = plt.figure(figsize=(9, 7))
        ax = fig_s.add_subplot(111, projection="3d")
        surf = ax.plot_surface(Xi, Yi, Zi, cmap="viridis", alpha=0.8)
        fig_s.colorbar(surf, ax=ax, label=z_col, shrink=0.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Surface: {z_col}")
        fig_s.tight_layout()

        code = f"""# 3D Surface Plot
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import griddata
xi = np.linspace(df['{x_col}'].min(), df['{x_col}'].max(), 30)
yi = np.linspace(df['{y_col}'].min(), df['{y_col}'].max(), 30)
Xi, Yi = np.meshgrid(xi, yi)
Zi = griddata((df['{x_col}'], df['{y_col}']), df['{z_col}'], (Xi, Yi))
fig = go.Figure(go.Surface(x=Xi, y=Yi, z=Zi, colorscale='Viridis'))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── 3D Bar Chart ──────────────────────────────────────────────────────

def bar_3d(df: pd.DataFrame, x_col: str, y_col: str, z_col: str):
    """3D bar chart using plotly and matplotlib."""
    try:
        agg = df.groupby([x_col, y_col])[z_col].mean().reset_index().head(100)

        x_cats = agg[x_col].astype(str).unique()
        y_cats = agg[y_col].astype(str).unique()
        x_idx = {c: i for i, c in enumerate(x_cats)}
        y_idx = {c: i for i, c in enumerate(y_cats)}

        # Plotly 3D bar (using scatter3d with vertical lines)
        fig_p = go.Figure()
        colors = plt.cm.viridis(np.linspace(0, 1, len(agg)))
        for i, (_, row) in enumerate(agg.iterrows()):
            xi = x_idx[str(row[x_col])]
            yi = y_idx[str(row[y_col])]
            zi = float(row[z_col])
            fig_p.add_trace(go.Scatter3d(
                x=[xi, xi], y=[yi, yi], z=[0, zi],
                mode="lines",
                line=dict(color=f"rgba({int(colors[i][0]*255)},{int(colors[i][1]*255)},{int(colors[i][2]*255)},0.8)",
                          width=8),
                showlegend=False,
            ))
        fig_p.update_layout(
            title=f"3D Bar: {z_col}",
            scene=dict(
                xaxis=dict(ticktext=list(x_cats),
                           tickvals=list(range(len(x_cats))),
                           title=x_col),
                yaxis=dict(ticktext=list(y_cats),
                           tickvals=list(range(len(y_cats))),
                           title=y_col),
                zaxis_title=z_col,
            ),
        )

        # Matplotlib
        fig_s = plt.figure(figsize=(10, 7))
        ax = fig_s.add_subplot(111, projection="3d")
        xs = [x_idx[str(r[x_col])] for _, r in agg.iterrows()]
        ys = [y_idx[str(r[y_col])] for _, r in agg.iterrows()]
        zs = agg[z_col].values
        dx = dy = 0.7
        ax.bar3d(xs, ys, np.zeros_like(zs), dx, dy, zs,
                 shade=True, alpha=0.8, color="steelblue")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(f"3D Bar Chart: {z_col}")
        fig_s.tight_layout()

        code = f"""# 3D Bar Chart
import plotly.graph_objects as go
# Aggregate data
agg = df.groupby(['{x_col}', '{y_col}'])['{z_col}'].mean().reset_index()
fig = go.Figure()
for _, row in agg.iterrows():
    fig.add_trace(go.Scatter3d(
        x=[row['{x_col}'], row['{x_col}']],
        y=[row['{y_col}'], row['{y_col}']],
        z=[0, row['{z_col}']],
        mode='lines', line=dict(width=8)
    ))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
