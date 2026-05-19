"""
SciVizKit — Distribution generators
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
import scipy.stats as stats

from ..code_generator import get_code


def _safe_close(fig):
    try:
        plt.close(fig)
    except Exception:
        pass


# ── Histogram ─────────────────────────────────────────────────────────

def histogram(df: pd.DataFrame, x_col: str, color_col: str = None):
    try:
        fig_s, ax = plt.subplots(figsize=(8, 5))
        if color_col and color_col in df.columns:
            for grp, sub in df.groupby(color_col):
                ax.hist(sub[x_col].dropna(), bins=30, alpha=0.6, label=str(grp))
            ax.legend()
        else:
            ax.hist(df[x_col].dropna(), bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel(x_col)
        ax.set_ylabel("Count")
        ax.set_title(f"Histogram of {x_col}")
        fig_s.tight_layout()

        color_arg = color_col if (color_col and color_col in df.columns) else None
        fig_p = px.histogram(df, x=x_col, color=color_arg, nbins=30,
                             marginal="box", title=f"Histogram of {x_col}")
        code = get_code("histogram", x_col=x_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── KDE ───────────────────────────────────────────────────────────────

def kde_plot(df: pd.DataFrame, x_col: str, color_col: str = None):
    try:
        fig_s, ax = plt.subplots(figsize=(8, 5))
        hue = color_col if (color_col and color_col in df.columns) else None
        sns.kdeplot(data=df, x=x_col, hue=hue, fill=True, ax=ax)
        ax.set_title(f"KDE Plot of {x_col}")
        fig_s.tight_layout()

        color_arg = color_col if hue else None
        fig_p = px.histogram(df, x=x_col, color=color_arg,
                             histnorm="density", marginal="rug",
                             title=f"KDE of {x_col}")
        code = get_code("kde", x_col=x_col, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Violin ────────────────────────────────────────────────────────────

def violin_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        fig_s, ax = plt.subplots(figsize=(9, 6))
        sns.violinplot(data=df, x=x_col, y=y_col, ax=ax, inner="box")
        ax.set_title("Violin Plot")
        fig_s.tight_layout()

        fig_p = px.violin(df, x=x_col, y=y_col, box=True, points="outliers",
                          title="Violin Plot")
        code = get_code("violin", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Box Plot ──────────────────────────────────────────────────────────

def box_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        fig_s, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
        ax.set_title("Box Plot")
        fig_s.tight_layout()

        fig_p = px.box(df, x=x_col, y=y_col, points="outliers", title="Box Plot")
        code = get_code("boxplot", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Strip Plot ────────────────────────────────────────────────────────

def strip_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        fig_s, ax = plt.subplots(figsize=(9, 6))
        sns.stripplot(data=df, x=x_col, y=y_col, jitter=True, ax=ax, alpha=0.7)
        ax.set_title("Strip Plot")
        fig_s.tight_layout()

        fig_p = px.strip(df, x=x_col, y=y_col, title="Strip Plot")
        code = get_code("stripplot", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── ECDF ──────────────────────────────────────────────────────────────

def ecdf_plot(df: pd.DataFrame, x_col: str):
    try:
        data = df[x_col].dropna().sort_values()
        y = np.arange(1, len(data) + 1) / len(data)

        fig_s, ax = plt.subplots(figsize=(8, 5))
        ax.plot(data, y, lw=2, color="steelblue")
        ax.set_xlabel(x_col)
        ax.set_ylabel("ECDF")
        ax.set_title(f"ECDF of {x_col}")
        fig_s.tight_layout()

        fig_p = px.ecdf(df, x=x_col, title=f"ECDF of {x_col}")
        code = get_code("ecdf", x_col=x_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Q-Q Plot ──────────────────────────────────────────────────────────

def qq_plot(df: pd.DataFrame, x_col: str):
    try:
        fig_s, ax = plt.subplots(figsize=(6, 6))
        (osm, osr), (slope, intercept, r) = stats.probplot(df[x_col].dropna(), dist="norm")
        ax.plot(osm, osr, "o", alpha=0.5, color="steelblue")
        ax.plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel("Sample Quantiles")
        ax.set_title(f"Q-Q Plot of {x_col}")
        fig_s.tight_layout()

        code = get_code("qqplot", x_col=x_col)
        return fig_s, None, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Beeswarm ──────────────────────────────────────────────────────────

def beeswarm_plot(df: pd.DataFrame, x_col: str, y_col: str):
    try:
        fig_s, ax = plt.subplots(figsize=(9, 6))
        sns.swarmplot(data=df.sample(min(300, len(df))), x=x_col, y=y_col, ax=ax)
        ax.set_title("Beeswarm Plot")
        fig_s.tight_layout()

        fig_p = px.strip(df, x=x_col, y=y_col, title="Beeswarm Plot")
        code = get_code("beeswarm", x_col=x_col, y_col=y_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
