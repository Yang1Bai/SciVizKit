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


# ── Ridgeline Plot ────────────────────────────────────────────────────

def ridgeline_plot(df: pd.DataFrame, x_col: str, group_col: str):
    """Joy/Ridgeline plot using matplotlib (manual KDE stacking)"""
    try:
        groups = df[group_col].dropna().unique()
        n = len(groups)
        if n > 15:
            groups = groups[:15]
            n = 15

        fig_s, axes = plt.subplots(n, 1, figsize=(10, max(6, n * 1.2)),
                                   sharex=True)
        if n == 1:
            axes = [axes]

        colors = plt.cm.viridis(np.linspace(0.2, 0.85, n))
        x_min = df[x_col].quantile(0.01)
        x_max = df[x_col].quantile(0.99)
        xs = np.linspace(x_min, x_max, 300)

        for i, (grp, ax, color) in enumerate(zip(groups, axes, colors)):
            sub = df[df[group_col] == grp][x_col].dropna()
            if len(sub) < 5:
                continue
            kde = stats.gaussian_kde(sub)
            ys = kde(xs)
            ax.fill_between(xs, ys, alpha=0.7, color=color)
            ax.plot(xs, ys, color=color, lw=1.5)
            ax.set_ylabel(str(grp), rotation=0, ha="right", va="center", fontsize=8)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        axes[-1].set_xlabel(x_col)
        fig_s.suptitle(f"Ridgeline Plot of {x_col} by {group_col}", y=1.01)
        fig_s.tight_layout()

        # Plotly version using violin
        fig_p = go.Figure()
        for grp in groups:
            sub = df[df[group_col] == grp][x_col].dropna()
            fig_p.add_trace(go.Violin(x=sub, name=str(grp), orientation="h",
                                      side="positive", width=2, points=False))
        fig_p.update_layout(title=f"Ridgeline Plot of {x_col} by {group_col}",
                            xaxis_title=x_col)

        code = f"""# Ridgeline Plot
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

groups = df['{group_col}'].unique()
n = len(groups)
fig, axes = plt.subplots(n, 1, figsize=(10, n*1.2), sharex=True)
xs = np.linspace(df['{x_col}'].min(), df['{x_col}'].max(), 300)
for ax, grp in zip(axes, groups):
    sub = df[df['{group_col}']==grp]['{x_col}'].dropna()
    kde = stats.gaussian_kde(sub)
    ax.fill_between(xs, kde(xs), alpha=0.7)
    ax.set_ylabel(str(grp), rotation=0, ha='right')
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Marginal Plot ─────────────────────────────────────────────────────

def marginal_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """Scatter plot with marginal histograms on top and right axes"""
    try:
        clean = df[[x_col, y_col] + ([color_col] if color_col and color_col in df.columns else [])].dropna()

        fig_s = plt.figure(figsize=(8, 8))
        gs = fig_s.add_gridspec(4, 4, hspace=0.05, wspace=0.05)
        ax_main = fig_s.add_subplot(gs[1:, :-1])
        ax_top = fig_s.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig_s.add_subplot(gs[1:, -1], sharey=ax_main)

        if color_col and color_col in df.columns:
            for grp, sub in clean.groupby(color_col):
                ax_main.scatter(sub[x_col], sub[y_col], alpha=0.5, s=20, label=str(grp))
            ax_main.legend(fontsize=7)
        else:
            ax_main.scatter(clean[x_col], clean[y_col], alpha=0.5, color="steelblue", s=20)

        ax_top.hist(clean[x_col], bins=30, color="steelblue", alpha=0.7)
        ax_right.hist(clean[y_col], bins=30, orientation="horizontal",
                      color="steelblue", alpha=0.7)
        ax_top.axis("off")
        ax_right.axis("off")
        ax_main.set_xlabel(x_col)
        ax_main.set_ylabel(y_col)
        fig_s.suptitle(f"Marginal Plot: {x_col} vs {y_col}")
        fig_s.tight_layout()

        color_arg = color_col if (color_col and color_col in df.columns) else None
        fig_p = px.scatter(clean, x=x_col, y=y_col, color=color_arg,
                           marginal_x="histogram", marginal_y="histogram",
                           title=f"Marginal Plot: {x_col} vs {y_col}")

        code = f"""# Marginal Plot
import plotly.express as px
fig = px.scatter(df, x='{x_col}', y='{y_col}',
                 marginal_x='histogram', marginal_y='histogram',
                 title='Marginal Plot')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Raincloud Plot ────────────────────────────────────────────────────

def raincloud_plot(df: pd.DataFrame, x_col: str, y_col: str):
    """Combined violin + box + strip plot (raincloud style)"""
    try:
        groups = df[x_col].dropna().unique()[:10]
        fig_s, ax = plt.subplots(figsize=(max(8, len(groups) * 1.5), 6))

        positions = np.arange(len(groups))
        colors = plt.cm.tab10.colors

        for i, grp in enumerate(groups):
            sub = df[df[x_col] == grp][y_col].dropna().values
            if len(sub) < 3:
                continue
            c = colors[i % len(colors)]
            pos = positions[i]

            # Violin (half)
            vp = ax.violinplot([sub], positions=[pos], widths=0.6,
                               showmeans=False, showmedians=False, showextrema=False)
            for body in vp["bodies"]:
                body.set_facecolor(c)
                body.set_alpha(0.5)
                # Clip to left half only
                m = np.mean(body.get_paths()[0].vertices[:, 0])
                body.get_paths()[0].vertices[:, 0] = np.clip(
                    body.get_paths()[0].vertices[:, 0], -np.inf, m)

            # Box
            bp = ax.boxplot([sub], positions=[pos + 0.15], widths=0.1,
                            patch_artist=True,
                            boxprops=dict(facecolor=c, alpha=0.7),
                            whiskerprops=dict(color=c),
                            capprops=dict(color=c),
                            medianprops=dict(color="white", linewidth=2),
                            flierprops=dict(marker="o", markersize=3, alpha=0.5))

            # Strip (jittered)
            jitter = np.random.uniform(-0.05, 0.05, len(sub))
            ax.scatter(np.full(len(sub), pos + 0.3) + jitter, sub,
                       alpha=0.5, s=10, color=c)

        ax.set_xticks(positions)
        ax.set_xticklabels([str(g) for g in groups], rotation=45, ha="right")
        ax.set_ylabel(y_col)
        ax.set_title(f"Raincloud Plot: {y_col} by {x_col}")
        fig_s.tight_layout()

        # Plotly: violin with box and points
        fig_p = px.violin(df[df[x_col].isin(groups)], x=x_col, y=y_col,
                          box=True, points="all",
                          title=f"Raincloud Plot: {y_col} by {x_col}")

        code = f"""# Raincloud Plot (violin + box + strip)
import plotly.express as px
fig = px.violin(df, x='{x_col}', y='{y_col}', box=True, points='all',
                title='Raincloud Plot')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
