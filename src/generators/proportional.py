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
