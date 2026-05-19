"""
SciVizKit — Time Series generators
Each function returns (fig_static, fig_plotly, code_str).
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from ..code_generator import get_code


def _sort_by_x(df, x_col):
    try:
        df2 = df.copy()
        df2[x_col] = pd.to_datetime(df2[x_col], infer_datetime_format=True, errors="coerce")
        return df2.sort_values(x_col)
    except Exception:
        return df.sort_values(x_col)


# ── Line Chart ────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x_col: str, y_cols: list):
    try:
        df2 = _sort_by_x(df, x_col)

        fig_s, ax = plt.subplots(figsize=(10, 5))
        for y in y_cols:
            ax.plot(df2[x_col], df2[y], lw=2, label=y)
        ax.set_xlabel(x_col)
        ax.legend()
        ax.set_title("Line Chart")
        fig_s.tight_layout()

        fig_p = px.line(df2, x=x_col, y=y_cols, title="Line Chart")
        code = get_code("line", x_col=x_col, y_cols=y_cols)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Area Chart ────────────────────────────────────────────────────────

def area_chart(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        df2 = _sort_by_x(df, x_col)

        fig_s, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(df2[x_col], df2[y_col], alpha=0.5, color="steelblue")
        ax.plot(df2[x_col], df2[y_col], lw=2, color="steelblue")
        ax.set_title("Area Chart")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig_s.tight_layout()

        fig_p = px.area(df2, x=x_col, y=y_col, title="Area Chart")
        code = get_code("area", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Stacked Area ──────────────────────────────────────────────────────

def stacked_area(df: pd.DataFrame, x_col: str, y_cols: list):
    try:
        df2 = _sort_by_x(df, x_col)

        fig_s, ax = plt.subplots(figsize=(10, 5))
        ax.stackplot(df2[x_col], [df2[y] for y in y_cols], labels=y_cols, alpha=0.7)
        ax.set_title("Stacked Area Chart")
        ax.set_xlabel(x_col)
        ax.legend(loc="upper left")
        fig_s.tight_layout()

        fig_p = px.area(df2, x=x_col, y=y_cols, title="Stacked Area Chart")
        code = get_code("stacked_area", x_col=x_col, y_cols=y_cols)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Step Line ─────────────────────────────────────────────────────────

def step_line(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        df2 = _sort_by_x(df, x_col)

        fig_s, ax = plt.subplots(figsize=(10, 5))
        ax.step(df2[x_col], df2[y_col], lw=2, color="steelblue", where="post")
        ax.set_title("Step Line Chart")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig_s.tight_layout()

        fig_p = px.line(df2, x=x_col, y=y_col, line_shape="hv",
                        title="Step Line Chart")
        code = get_code("step_line", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
