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


# ── Diverging Bar ─────────────────────────────────────────────────────

def diverging_bar(df: pd.DataFrame, label_col: str, value_col: str):
    """Diverging bar chart centered at 0"""
    try:
        agg = df.groupby(label_col)[value_col].mean().reset_index().sort_values(value_col)
        colors = ["#d73027" if v < 0 else "#4575b4" for v in agg[value_col]]

        fig_s, ax = plt.subplots(figsize=(9, max(5, len(agg) * 0.5)))
        ax.barh(agg[label_col].astype(str), agg[value_col], color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel(value_col)
        ax.set_title(f"Diverging Bar: {value_col}")
        fig_s.tight_layout()

        fig_p = go.Figure(go.Bar(
            x=agg[value_col].tolist(),
            y=agg[label_col].astype(str).tolist(),
            orientation="h",
            marker_color=["#d73027" if v < 0 else "#4575b4" for v in agg[value_col]],
        ))
        fig_p.update_layout(title=f"Diverging Bar: {value_col}",
                            xaxis_title=value_col)
        code = f"""# Diverging Bar Chart
import plotly.graph_objects as go
colors = ['#d73027' if v < 0 else '#4575b4' for v in df['{value_col}']]
fig = go.Figure(go.Bar(x=df['{value_col}'], y=df['{label_col}'],
                       orientation='h', marker_color=colors))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Population Pyramid ─────────────────────────────────────────────────

def population_pyramid(df: pd.DataFrame, age_col: str, male_col: str, female_col: str):
    """Back-to-back horizontal bar chart"""
    try:
        sample = df[[age_col, male_col, female_col]].dropna().head(20)
        ages = sample[age_col].astype(str)
        males = sample[male_col].abs()
        females = sample[female_col].abs()

        fig_s, ax = plt.subplots(figsize=(10, max(6, len(sample) * 0.5)))
        ax.barh(ages, -males, color="#4575b4", label=male_col)
        ax.barh(ages, females, color="#d73027", label=female_col)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("← Male | Female →")
        ax.set_title("Population Pyramid")
        ax.legend()
        # Fix x-tick labels to show absolute values
        ticks = ax.get_xticks()
        ax.set_xticklabels([str(abs(int(t))) for t in ticks])
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(y=ages, x=-males, name=male_col,
                               orientation="h", marker_color="#4575b4"))
        fig_p.add_trace(go.Bar(y=ages, x=females, name=female_col,
                               orientation="h", marker_color="#d73027"))
        fig_p.update_layout(title="Population Pyramid", barmode="overlay",
                            xaxis_title="Population")
        code = f"""# Population Pyramid
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(y=df['{age_col}'], x=-df['{male_col}'], name='{male_col}',
                     orientation='h', marker_color='#4575b4'))
fig.add_trace(go.Bar(y=df['{age_col}'], x=df['{female_col}'], name='{female_col}',
                     orientation='h', marker_color='#d73027'))
fig.update_layout(barmode='overlay')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Percent Stacked Bar ───────────────────────────────────────────────

def percent_stacked_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str):
    """100% stacked bar showing proportions"""
    try:
        agg = df.groupby([x_col, color_col])[y_col].sum().reset_index()
        pivot = agg.pivot(index=x_col, columns=color_col, values=y_col).fillna(0)
        pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        fig_s, ax = plt.subplots(figsize=(10, 6))
        pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("Percentage (%)")
        ax.set_title("100% Stacked Bar Chart")
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha="right")
        fig_s.tight_layout()

        agg_pct = pct.reset_index().melt(id_vars=x_col, value_name="pct")
        fig_p = px.bar(agg_pct, x=x_col, y="pct", color=color_col,
                       barmode="stack", title="100% Stacked Bar Chart",
                       labels={"pct": "Percentage (%)"})
        code = f"""# 100% Stacked Bar Chart
import plotly.express as px
agg = df.groupby(['{x_col}', '{color_col}'])['{y_col}'].sum().reset_index()
pivot = agg.pivot(index='{x_col}', columns='{color_col}', values='{y_col}').fillna(0)
pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
pct_df = pct.reset_index().melt(id_vars='{x_col}', value_name='pct')
fig = px.bar(pct_df, x='{x_col}', y='pct', color='{color_col}', barmode='stack')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Radial Bar Chart ──────────────────────────────────────────────────

def radial_bar_chart(df: pd.DataFrame, label_col: str, value_col: str):
    """Circular/radial bar chart using polar axes"""
    try:
        agg = df.groupby(label_col)[value_col].mean().reset_index().head(20)
        N = len(agg)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = agg[value_col].values.astype(float)
        width = 2 * np.pi / N * 0.8
        colors = plt.cm.viridis(radii / (radii.max() + 1e-9))

        fig_s, ax = plt.subplots(subplot_kw={"polar": True}, figsize=(8, 8))
        bars = ax.bar(theta, radii, width=width, bottom=0.0,
                      color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(theta)
        ax.set_xticklabels(agg[label_col].astype(str), fontsize=8)
        ax.set_title("Radial Bar Chart", pad=20)
        fig_s.tight_layout()

        fig_p = go.Figure(go.Barpolar(
            r=radii.tolist(),
            theta=agg[label_col].astype(str).tolist(),
            marker_colorscale="Viridis",
            marker_color=radii.tolist(),
        ))
        fig_p.update_layout(title="Radial Bar Chart")
        code = f"""# Radial Bar Chart
import plotly.graph_objects as go
fig = go.Figure(go.Barpolar(
    r=df['{value_col}'], theta=df['{label_col}'],
    marker_colorscale='Viridis'
))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Bullet Chart ──────────────────────────────────────────────────────

def bullet_chart(df: pd.DataFrame, label_col: str, actual_col: str,
                 target_col: str, range_col: str = None):
    """Bullet chart showing actual vs target"""
    try:
        sample = df[[label_col, actual_col, target_col] +
                    ([range_col] if range_col and range_col in df.columns else [])].dropna().head(10)

        fig_s, ax = plt.subplots(figsize=(10, max(4, len(sample) * 0.8)))
        y_pos = np.arange(len(sample))
        height = 0.4

        for i, (_, row) in enumerate(sample.iterrows()):
            if range_col and range_col in df.columns:
                ax.barh(i, row[range_col], height=height * 1.5,
                        color="#cccccc", zorder=1)
            ax.barh(i, row[actual_col], height=height,
                    color="#4575b4", zorder=2)
            ax.plot([row[target_col], row[target_col]],
                    [i - height * 0.8, i + height * 0.8],
                    color="#d73027", lw=3, zorder=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(sample[label_col].astype(str))
        ax.set_xlabel(actual_col)
        ax.set_title("Bullet Chart")
        fig_s.tight_layout()

        fig_p = go.Figure()
        for _, row in sample.iterrows():
            fig_p.add_trace(go.Bar(
                name=str(row[label_col]),
                x=[row[actual_col]],
                y=[str(row[label_col])],
                orientation="h",
                marker_color="#4575b4",
            ))
            fig_p.add_shape(type="line",
                            x0=row[target_col], x1=row[target_col],
                            y0=str(row[label_col]),
                            y1=str(row[label_col]),
                            line=dict(color="red", width=3))
        fig_p.update_layout(title="Bullet Chart", showlegend=False)
        code = f"""# Bullet Chart
# Actual vs Target comparison
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
for i, (_, row) in enumerate(df.iterrows()):
    ax.barh(i, row['{actual_col}'], color='#4575b4', height=0.4)
    ax.plot([row['{target_col}']]*2, [i-0.25, i+0.25], color='red', lw=3)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['{label_col}'])
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Tornado Chart ─────────────────────────────────────────────────────

def tornado_chart(df: pd.DataFrame, label_col: str, low_col: str, high_col: str):
    """Tornado/butterfly chart for sensitivity analysis"""
    try:
        sample = df[[label_col, low_col, high_col]].dropna().copy()
        sample["range"] = sample[high_col] - sample[low_col]
        sample = sample.sort_values("range", ascending=True).head(15)

        fig_s, ax = plt.subplots(figsize=(10, max(5, len(sample) * 0.6)))
        y_pos = np.arange(len(sample))
        base = (sample[low_col] + sample[high_col]) / 2
        ax.barh(y_pos, sample[high_col] - base, left=base,
                color="#d73027", alpha=0.8, label="High")
        ax.barh(y_pos, sample[low_col] - base, left=base,
                color="#4575b4", alpha=0.8, label="Low")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sample[label_col].astype(str))
        ax.axvline(base.mean(), color="black", lw=0.8, linestyle="--")
        ax.set_title("Tornado Chart")
        ax.legend()
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Bar(
            name="Low",
            y=sample[label_col].astype(str).tolist(),
            x=sample[low_col].tolist(),
            orientation="h",
            marker_color="#4575b4",
        ))
        fig_p.add_trace(go.Bar(
            name="High",
            y=sample[label_col].astype(str).tolist(),
            x=sample[high_col].tolist(),
            orientation="h",
            marker_color="#d73027",
        ))
        fig_p.update_layout(title="Tornado Chart", barmode="overlay")
        code = f"""# Tornado Chart
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Bar(y=df['{label_col}'], x=df['{low_col}'],
                     orientation='h', name='Low', marker_color='#4575b4'))
fig.add_trace(go.Bar(y=df['{label_col}'], x=df['{high_col}'],
                     orientation='h', name='High', marker_color='#d73027'))
fig.update_layout(barmode='overlay')
fig.show()
"""
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
