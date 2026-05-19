"""
SciVizKit — Comparison generators
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
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from ..code_generator import get_code


# ── Bar Chart ─────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        agg = df.groupby(x_col)[y_col].mean().reset_index()

        fig_s, ax = plt.subplots(figsize=(9, 5))
        ax.bar(agg[x_col].astype(str), agg[y_col], color="steelblue")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Bar Chart: {y_col} by {x_col}")
        plt.xticks(rotation=45, ha="right")
        fig_s.tight_layout()

        fig_p = px.bar(agg, x=x_col, y=y_col, title=f"Bar Chart: {y_col} by {x_col}")
        code = get_code("bar", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Grouped Bar ───────────────────────────────────────────────────────

def grouped_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str):
    try:
        agg = df.groupby([x_col, color_col])[y_col].mean().reset_index()

        fig_p = px.bar(agg, x=x_col, y=y_col, color=color_col,
                       barmode="group", title="Grouped Bar Chart")

        fig_s, ax = plt.subplots(figsize=(10, 6))
        groups = agg[color_col].unique()
        cats = agg[x_col].unique()
        x = np.arange(len(cats))
        width = 0.8 / len(groups)
        for i, g in enumerate(groups):
            sub = agg[agg[color_col] == g]
            vals = [sub[sub[x_col] == c][y_col].values[0] if len(sub[sub[x_col] == c]) else 0 for c in cats]
            ax.bar(x + i * width, vals, width, label=str(g))
        ax.set_xticks(x + width * (len(groups) - 1) / 2)
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_title("Grouped Bar Chart")
        ax.legend()
        fig_s.tight_layout()

        code = get_code("grouped_bar", x_col=x_col, y_col=y_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Stacked Bar ───────────────────────────────────────────────────────

def stacked_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str):
    try:
        agg = df.groupby([x_col, color_col])[y_col].sum().reset_index()
        pivot = agg.pivot(index=x_col, columns=color_col, values=y_col).fillna(0)

        fig_s, ax = plt.subplots(figsize=(10, 6))
        pivot.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Stacked Bar Chart")
        plt.xticks(rotation=45, ha="right")
        fig_s.tight_layout()

        fig_p = px.bar(agg, x=x_col, y=y_col, color=color_col,
                       barmode="stack", title="Stacked Bar Chart")
        code = get_code("stacked_bar", x_col=x_col, y_col=y_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Lollipop ──────────────────────────────────────────────────────────

def lollipop_chart(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        agg = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col)

        fig_s, ax = plt.subplots(figsize=(8, max(5, len(agg) * 0.4)))
        ax.hlines(agg[x_col].astype(str), 0, agg[y_col], colors="steelblue", linewidth=2)
        ax.plot(agg[y_col], agg[x_col].astype(str), "o", color="steelblue", markersize=8)
        ax.set_title("Lollipop Chart")
        fig_s.tight_layout()

        fig_p = go.Figure()
        for _, row in agg.iterrows():
            fig_p.add_trace(go.Scatter(x=[0, row[y_col]], y=[str(row[x_col]), str(row[x_col])],
                                       mode="lines+markers", line=dict(color="steelblue"),
                                       marker=dict(size=10), showlegend=False))
        fig_p.update_layout(title="Lollipop Chart")
        code = get_code("lollipop", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Dumbbell ─────────────────────────────────────────────────────────

def dumbbell_chart(df: pd.DataFrame, label_col: str, val1_col: str, val2_col: str):
    try:
        sample = df[[label_col, val1_col, val2_col]].dropna().head(20)

        fig_s, ax = plt.subplots(figsize=(9, max(5, len(sample) * 0.5)))
        for _, row in sample.iterrows():
            lbl = str(row[label_col])
            ax.plot([row[val1_col], row[val2_col]], [lbl, lbl], "o-",
                    color="gray", linewidth=2)
            ax.plot(row[val1_col], lbl, "o", color="#4e79a7", markersize=10)
            ax.plot(row[val2_col], lbl, "o", color="#f28e2b", markersize=10)
        ax.set_title(f"Dumbbell: {val1_col} vs {val2_col}")
        fig_s.tight_layout()

        fig_p = go.Figure()
        for _, row in sample.iterrows():
            lbl = str(row[label_col])
            fig_p.add_trace(go.Scatter(x=[row[val1_col], row[val2_col]],
                                       y=[lbl, lbl], mode="lines+markers",
                                       marker=dict(size=12, color=["#4e79a7", "#f28e2b"]),
                                       showlegend=False))
        fig_p.update_layout(title="Dumbbell Chart")
        code = get_code("dumbbell", label_col=label_col, val1_col=val1_col, val2_col=val2_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Dot Plot ──────────────────────────────────────────────────────────

def dot_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        agg = df.groupby(x_col)[y_col].mean().reset_index().sort_values(y_col)

        fig_s, ax = plt.subplots(figsize=(8, max(5, len(agg) * 0.4)))
        ax.plot(agg[y_col], agg[x_col].astype(str), "o", color="steelblue", markersize=10)
        ax.set_title("Dot Plot")
        ax.grid(axis="x", alpha=0.3)
        fig_s.tight_layout()

        fig_p = px.scatter(agg, x=y_col, y=x_col, title="Dot Plot")
        code = get_code("dotplot", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Slope Chart ───────────────────────────────────────────────────────

def slope_chart(df: pd.DataFrame, label_col: str, val1_col: str, val2_col: str):
    try:
        sample = df[[label_col, val1_col, val2_col]].dropna().head(15)
        colors = plt.cm.tab20.colors

        fig_s, ax = plt.subplots(figsize=(6, max(6, len(sample) * 0.5)))
        for i, (_, row) in enumerate(sample.iterrows()):
            c = colors[i % len(colors)]
            ax.plot([1, 2], [row[val1_col], row[val2_col]], "o-", color=c, linewidth=2)
            ax.text(0.95, row[val1_col], str(row[label_col]), ha="right", fontsize=8, color=c)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([val1_col, val2_col])
        ax.set_title("Slope Chart")
        fig_s.tight_layout()

        fig_p = go.Figure()
        for _, row in sample.iterrows():
            fig_p.add_trace(go.Scatter(x=[val1_col, val2_col],
                                       y=[row[val1_col], row[val2_col]],
                                       mode="lines+markers+text",
                                       name=str(row[label_col]),
                                       text=[str(row[label_col]), ""],
                                       textposition="middle left"))
        fig_p.update_layout(title="Slope Chart")
        code = get_code("slope", label_col=label_col, val1_col=val1_col, val2_col=val2_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Waterfall ─────────────────────────────────────────────────────────

def waterfall_chart(df: pd.DataFrame, label_col: str, value_col: str):
    try:
        sample = df[[label_col, value_col]].dropna().head(20)
        labels = sample[label_col].astype(str).tolist()
        values = sample[value_col].tolist()

        # Static
        running = 0
        bottoms = []
        for v in values:
            bottoms.append(running)
            running += v
        colors = ["green" if v >= 0 else "red" for v in values]

        fig_s, ax = plt.subplots(figsize=(10, 6))
        ax.bar(labels, values, bottom=bottoms, color=colors, edgecolor="white")
        ax.set_title("Waterfall Chart")
        plt.xticks(rotation=45, ha="right")
        fig_s.tight_layout()

        fig_p = go.Figure(go.Waterfall(x=labels, y=values, orientation="v"))
        fig_p.update_layout(title="Waterfall Chart")
        code = get_code("waterfall", label_col=label_col, value_col=value_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Error Bar ─────────────────────────────────────────────────────────

def error_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, err_col: str):
    try:
        sample = df[[x_col, y_col, err_col]].dropna()

        fig_s, ax = plt.subplots(figsize=(9, 5))
        ax.bar(sample[x_col].astype(str), sample[y_col], yerr=sample[err_col],
               capsize=5, color="steelblue", error_kw=dict(ecolor="black"))
        ax.set_title("Error Bar Plot")
        plt.xticks(rotation=45, ha="right")
        fig_s.tight_layout()

        fig_p = go.Figure(go.Bar(
            x=sample[x_col].astype(str),
            y=sample[y_col],
            error_y=dict(type="data", array=sample[err_col].tolist()),
        ))
        fig_p.update_layout(title="Error Bar Plot")
        code = get_code("errorbar", x_col=x_col, y_col=y_col, err_col=err_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
