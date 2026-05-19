"""
SciVizKit — Correlation generators
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
from scipy import stats

from ..code_generator import get_code


# ── Scatter Plot ──────────────────────────────────────────────────────

def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                 color_col: str = None, add_regression: bool = True):
    try:
        fig_s, ax = plt.subplots(figsize=(8, 6))
        if color_col and color_col in df.columns:
            groups = df[color_col].unique()
            for g in groups:
                sub = df[df[color_col] == g]
                ax.scatter(sub[x_col], sub[y_col], alpha=0.6, label=str(g))
            ax.legend()
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, color="steelblue")
        if add_regression:
            clean = df[[x_col, y_col]].dropna()
            if len(clean) > 2:
                slope, intercept, r, p, se = stats.linregress(clean[x_col], clean[y_col])
                xr = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
                ax.plot(xr, slope * xr + intercept, "r--", lw=2,
                        label=f"R²={r**2:.3f}")
                ax.legend()
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter: {x_col} vs {y_col}")
        fig_s.tight_layout()

        color_arg = color_col if (color_col and color_col in df.columns) else None
        trendline = "ols" if add_regression else None
        fig_p = px.scatter(df, x=x_col, y=y_col, color=color_arg,
                           trendline=trendline,
                           title=f"Scatter: {x_col} vs {y_col}")
        code = get_code("scatter", x_col=x_col, y_col=y_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Bubble Chart ──────────────────────────────────────────────────────

def bubble_chart(df: pd.DataFrame, x_col: str, y_col: str,
                 size_col: str, color_col: str = None):
    try:
        color_arg = color_col if (color_col and color_col in df.columns) else None
        df2 = df.dropna(subset=[x_col, y_col, size_col])
        sizes = (df2[size_col] - df2[size_col].min()) / (df2[size_col].max() - df2[size_col].min() + 1e-9)
        sizes = sizes * 500 + 20

        fig_s, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df2[x_col], df2[y_col], s=sizes, alpha=0.6, color="steelblue")
        ax.set_title("Bubble Chart")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig_s.tight_layout()

        fig_p = px.scatter(df2, x=x_col, y=y_col, size=size_col, color=color_arg,
                           size_max=50, title="Bubble Chart")
        code = get_code("bubble", x_col=x_col, y_col=y_col, size_col=size_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Hexbin ────────────────────────────────────────────────────────────

def hexbin_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        clean = df[[x_col, y_col]].dropna()

        fig_s, ax = plt.subplots(figsize=(8, 6))
        hb = ax.hexbin(clean[x_col], clean[y_col], gridsize=30, cmap="Blues")
        fig_s.colorbar(hb, ax=ax, label="count")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Hexbin: {x_col} vs {y_col}")
        fig_s.tight_layout()

        code = get_code("hexbin", x_col=x_col, y_col=y_col)
        return fig_s, None, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Correlation Heatmap ───────────────────────────────────────────────

def corr_heatmap(df: pd.DataFrame):
    try:
        num_df = df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return None, None, "# Need at least 2 numeric columns"

        corr = num_df.corr()

        fig_s, ax = plt.subplots(figsize=(max(6, len(corr) * 0.8), max(5, len(corr) * 0.7)))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap")
        fig_s.tight_layout()

        fig_p = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Correlation Heatmap")
        code = get_code("corr_heatmap")
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Pair Plot ─────────────────────────────────────────────────────────

def pairplot(df: pd.DataFrame, cols: list):
    try:
        sub = df[cols].dropna() if cols else df.select_dtypes("number").dropna()
        sub = sub.head(500)  # cap for performance

        g = sns.pairplot(sub, diag_kind="kde", plot_kws={"alpha": 0.4})
        g.figure.suptitle("Pair Plot", y=1.02)
        g.figure.tight_layout()

        code = get_code("pairplot", cols=cols)
        return g.figure, None, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Parallel Coordinates ──────────────────────────────────────────────

def parallel_coords(df: pd.DataFrame, cols: list, color_col: str = None):
    try:
        num_df = df[cols].select_dtypes(include="number") if cols else df.select_dtypes(include="number")
        if num_df.empty:
            return None, None, "# No numeric columns"

        color_col_final = color_col if (color_col and color_col in df.columns) else num_df.columns[0]
        color_vals = df[color_col_final] if color_col_final in df.columns else None

        fig_p = px.parallel_coordinates(df, dimensions=num_df.columns.tolist(),
                                         color=color_col_final,
                                         title="Parallel Coordinates")

        fig_s, ax = plt.subplots(figsize=(12, 6))
        pd.plotting.parallel_coordinates(
            num_df.assign(_color_=df[color_col_final].astype(str) if color_vals is not None else "all"),
            "_color_", ax=ax, alpha=0.4
        )
        ax.set_title("Parallel Coordinates")
        ax.get_legend().remove()
        fig_s.tight_layout()

        code = get_code("parallel_coords", cols=num_df.columns.tolist(), color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
