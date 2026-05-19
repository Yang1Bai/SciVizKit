"""
SciVizKit — Proportional generators
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
import plotly.express as px
import plotly.graph_objects as go

try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False

from ..code_generator import get_code


def _agg(df, label_col, value_col):
    return df.groupby(label_col)[value_col].sum().reset_index()


# ── Pie Chart ─────────────────────────────────────────────────────────

def pie_chart(df: pd.DataFrame, label_col: str, value_col: str):
    try:
        agg = _agg(df, label_col, value_col)

        fig_s, ax = plt.subplots(figsize=(7, 7))
        ax.pie(agg[value_col], labels=agg[label_col].astype(str), autopct="%1.1f%%")
        ax.set_title("Pie Chart")
        fig_s.tight_layout()

        fig_p = px.pie(agg, names=label_col, values=value_col, title="Pie Chart")
        code = get_code("pie", label_col=label_col, value_col=value_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Donut Chart ───────────────────────────────────────────────────────

def donut_chart(df: pd.DataFrame, label_col: str, value_col: str):
    try:
        agg = _agg(df, label_col, value_col)

        fig_s, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            agg[value_col], labels=agg[label_col].astype(str),
            autopct="%1.1f%%", pctdistance=0.8,
            wedgeprops=dict(width=0.6)
        )
        ax.set_title("Donut Chart")
        fig_s.tight_layout()

        fig_p = px.pie(agg, names=label_col, values=value_col,
                       hole=0.4, title="Donut Chart")
        code = get_code("donut", label_col=label_col, value_col=value_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Treemap ───────────────────────────────────────────────────────────

def treemap(df: pd.DataFrame, label_col: str, value_col: str):
    try:
        agg = _agg(df, label_col, value_col)
        agg = agg[agg[value_col] > 0].sort_values(value_col, ascending=False).head(30)

        if HAS_SQUARIFY:
            fig_s, ax = plt.subplots(figsize=(10, 7))
            squarify.plot(sizes=agg[value_col].tolist(),
                          label=agg[label_col].astype(str).tolist(),
                          alpha=0.8, ax=ax)
            ax.axis("off")
            ax.set_title("Treemap")
            fig_s.tight_layout()
        else:
            fig_s = None

        fig_p = px.treemap(agg, path=[label_col], values=value_col, title="Treemap")
        code = get_code("treemap", label_col=label_col, value_col=value_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Sunburst ──────────────────────────────────────────────────────────

def sunburst(df: pd.DataFrame, label_col: str, parent_col: str, value_col: str):
    try:
        fig_p = px.sunburst(df, names=label_col, parents=parent_col,
                            values=value_col, title="Sunburst Chart")
        code = get_code("sunburst", label_col=label_col, parent_col=parent_col, value_col=value_col)
        return None, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Nightingale Rose ──────────────────────────────────────────────────

def waffle_chart(df: pd.DataFrame, label_col: str, value_col: str):
    """Waffle chart using matplotlib grid of squares"""
    try:
        agg = _agg(df, label_col, value_col)
        agg = agg[agg[value_col] > 0].head(10)
        total = agg[value_col].sum()
        shares = (agg[value_col] / total * 100).round().astype(int)
        # Adjust to sum to 100
        diff = 100 - shares.sum()
        shares.iloc[0] += diff

        grid_size = 10  # 10x10 = 100 cells
        cells = []
        for i, (_, row) in enumerate(agg.iterrows()):
            cells.extend([i] * shares.iloc[i])
        cells = cells[:100]
        grid = np.array(cells[::-1]).reshape(grid_size, grid_size)

        colors = plt.cm.tab10.colors
        cmap = plt.matplotlib.colors.ListedColormap(
            [colors[i % len(colors)] for i in range(len(agg))])

        fig_s, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=len(agg) - 1, aspect="equal")
        # Grid lines
        for x in range(grid_size + 1):
            ax.axvline(x - 0.5, color="white", lw=2)
        for y in range(grid_size + 1):
            ax.axhline(y - 0.5, color="white", lw=2)
        ax.set_xticks([])
        ax.set_yticks([])
        # Legend
        patches = [plt.matplotlib.patches.Patch(
            color=colors[i % len(colors)],
            label=f"{str(row[label_col])} ({shares.iloc[i]}%)")
            for i, (_, row) in enumerate(agg.iterrows())]
        ax.legend(handles=patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
        ax.set_title("Waffle Chart")
        fig_s.tight_layout()

        # Plotly: use treemap as proxy
        fig_p = px.treemap(agg, path=[label_col], values=value_col,
                           title="Waffle Chart (Treemap view)")

        code = f"""# Waffle Chart
import numpy as np
import matplotlib.pyplot as plt
agg = df.groupby('{label_col}')['{value_col}'].sum().reset_index()
# Build 10x10 grid proportional to values
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Marimekko Chart ───────────────────────────────────────────────────

def marimekko_chart(df: pd.DataFrame, x_col: str, y_col: str, value_col: str):
    """Mosaic/Marimekko chart"""
    try:
        agg = df.groupby([x_col, y_col])[value_col].sum().reset_index()
        x_cats = agg[x_col].unique()
        y_cats = agg[y_col].unique()

        # Column widths proportional to x totals
        x_totals = agg.groupby(x_col)[value_col].sum()
        grand_total = x_totals.sum()
        col_widths = x_totals / grand_total

        fig_s, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.tab20.colors
        x_pos = 0
        for xi, xcat in enumerate(x_cats):
            col_data = agg[agg[x_col] == xcat]
            col_total = col_data[value_col].sum()
            width = float(col_widths.get(xcat, 0))
            y_pos = 0
            for yi, ycat in enumerate(y_cats):
                row = col_data[col_data[y_col] == ycat]
                if len(row):
                    h = float(row[value_col].values[0]) / col_total if col_total > 0 else 0
                    ax.bar(x_pos + width / 2, h, width=width, bottom=y_pos,
                           color=colors[yi % len(colors)], edgecolor="white")
                    if h > 0.05:
                        ax.text(x_pos + width / 2, y_pos + h / 2,
                                str(ycat), ha="center", va="center", fontsize=7)
                    y_pos += h
            ax.text(x_pos + width / 2, -0.05, str(xcat),
                    ha="center", va="top", fontsize=8, rotation=30)
            x_pos += width

        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.05)
        ax.set_ylabel("Proportion")
        ax.set_xticks([])
        ax.set_title("Marimekko Chart")
        fig_s.tight_layout()

        fig_p = px.bar(agg, x=x_col, y=value_col, color=y_col,
                       barmode="stack", title="Marimekko Chart (Stacked Bar)")

        code = f"""# Marimekko / Mosaic Chart
# Column width = x_col total share; height = y_col proportion within column
import matplotlib.pyplot as plt
agg = df.groupby(['{x_col}', '{y_col}'])['{value_col}'].sum().reset_index()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Circle Packing ────────────────────────────────────────────────────

def circle_packing(df: pd.DataFrame, label_col: str, value_col: str,
                   parent_col: str = None):
    """Circle packing chart using plotly"""
    try:
        agg = _agg(df, label_col, value_col)
        agg = agg[agg[value_col] > 0].sort_values(value_col, ascending=False).head(30)

        if parent_col and parent_col in df.columns:
            path = [parent_col, label_col]
        else:
            path = [label_col]

        fig_p = px.treemap(agg, path=path, values=value_col,
                           title="Circle Packing")
        fig_p.update_traces(marker=dict(line=dict(width=2, color="white")))

        # Static matplotlib circles using a simple packing layout
        fig_s, ax = plt.subplots(figsize=(10, 10))
        values = agg[value_col].values.astype(float)
        radii = np.sqrt(values / values.max()) * 4
        colors = plt.cm.tab20(np.linspace(0, 1, len(agg)))

        # Simple grid placement (not true packing, but illustrative)
        n = len(agg)
        cols_n = int(np.ceil(np.sqrt(n)))
        for i, (radius, color, (_, row)) in enumerate(
                zip(radii, colors, agg.iterrows())):
            xi = (i % cols_n) * 2.5 - cols_n * 1.25
            yi = (i // cols_n) * 2.5 - (n // cols_n) * 1.25
            circle = plt.Circle((xi, yi), radius, color=color, alpha=0.8)
            ax.add_patch(circle)
            if radius > 0.5:
                ax.text(xi, yi, str(row[label_col])[:12],
                        ha="center", va="center", fontsize=7, wrap=True)

        ax.set_xlim(-cols_n * 1.5, cols_n * 1.5)
        ax.set_ylim(-cols_n * 1.5, cols_n * 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Circle Packing")
        fig_s.tight_layout()

        code = f"""# Circle Packing
import plotly.express as px
fig = px.treemap(df, path=['{label_col}'], values='{value_col}',
                 title='Circle Packing')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


def nightingale_rose(df: pd.DataFrame, label_col: str, value_col: str):
    try:
        agg = _agg(df, label_col, value_col)
        N = len(agg)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = agg[value_col].values.astype(float)
        width = 2 * np.pi / N

        fig_s, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, N))
        bars = ax.bar(theta, radii, width=width * 0.9, bottom=0.0, color=colors, alpha=0.85)
        ax.set_xticks(theta)
        ax.set_xticklabels(agg[label_col].astype(str), fontsize=8)
        ax.set_title("Nightingale Rose Chart", pad=20)
        fig_s.tight_layout()

        fig_p = go.Figure(go.Barpolar(
            r=radii.tolist(),
            theta=agg[label_col].astype(str).tolist(),
            name="value",
        ))
        fig_p.update_layout(title="Nightingale Rose Chart",
                            polar=dict(radialaxis=dict(visible=True)))
        code = get_code("nightingale", label_col=label_col, value_col=value_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
