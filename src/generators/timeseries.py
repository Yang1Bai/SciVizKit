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


# ── Streamgraph ────────────────────────────────────────────────────────

def streamgraph(df, x_col: str, y_cols: list):
    """Streamgraph (stacked area centered at 0)"""
    try:
        import numpy as np
        df2 = _sort_by_x(df, x_col)
        numeric_y = [c for c in y_cols if c in df2.columns and
                     pd.api.types.is_numeric_dtype(df2[c])]
        if not numeric_y:
            return None, None, "# No numeric y columns found"

        data = df2[numeric_y].fillna(0).values.T
        # Center: subtract half of sum
        totals = data.sum(axis=0)
        baseline = -totals / 2

        fig_s, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab20.colors
        ax.stackplot(df2[x_col], data, labels=numeric_y,
                     baseline="sym",
                     colors=[colors[i % len(colors)] for i in range(len(numeric_y))],
                     alpha=0.8)
        ax.set_xlabel(x_col)
        ax.set_title("Streamgraph")
        ax.legend(loc="upper left", fontsize=8)
        fig_s.tight_layout()

        fig_p = px.area(df2, x=x_col, y=numeric_y, title="Streamgraph")

        code = f"""# Streamgraph
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
ax.stackplot(df['{x_col}'], [df[c] for c in {numeric_y}],
             labels={numeric_y}, baseline='sym', alpha=0.8)
ax.legend()
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Bump Chart ──────────────────────────────────────────────────────────

def bump_chart(df, x_col: str, label_col: str, rank_col: str):
    """Bump/ranking chart showing rank changes over time"""
    try:
        import numpy as np
        df2 = df[[x_col, label_col, rank_col]].dropna().copy()
        df2 = _sort_by_x(df2, x_col)
        labels = df2[label_col].unique()[:12]
        colors = plt.cm.tab20.colors

        fig_s, ax = plt.subplots(figsize=(12, 6))
        for i, lbl in enumerate(labels):
            sub = df2[df2[label_col] == lbl].sort_values(x_col)
            ax.plot(sub[x_col], sub[rank_col], "o-", lw=2.5,
                    color=colors[i % len(colors)], label=str(lbl))

        ax.set_ylabel("Rank")
        ax.set_xlabel(x_col)
        ax.invert_yaxis()
        ax.set_title("Bump Chart (Rank Over Time)")
        ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, 0.5))
        fig_s.tight_layout()

        fig_p = px.line(df2[df2[label_col].isin(labels)],
                        x=x_col, y=rank_col, color=label_col,
                        markers=True, title="Bump Chart")
        fig_p.update_yaxes(autorange="reversed")

        code = f"""# Bump Chart
import plotly.express as px
fig = px.line(df, x='{x_col}', y='{rank_col}', color='{label_col}',
              markers=True, title='Bump Chart')
fig.update_yaxes(autorange='reversed')
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Calendar Heatmap ───────────────────────────────────────────────────

def calendar_heatmap(df, date_col: str, value_col: str):
    """GitHub-style calendar heatmap using matplotlib"""
    try:
        import numpy as np
        df2 = df[[date_col, value_col]].copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna(subset=[date_col])
        df2 = df2.groupby(date_col)[value_col].mean().reset_index()

        # Use the most recent year
        if df2.empty:
            return None, None, "# No valid dates found"

        latest_year = df2[date_col].dt.year.max()
        year_df = df2[df2[date_col].dt.year == latest_year].copy()
        year_df["week"] = year_df[date_col].dt.isocalendar().week.astype(int)
        year_df["dow"] = year_df[date_col].dt.dayofweek  # 0=Mon

        # Build 7 x 53 grid
        grid = np.full((7, 54), np.nan)
        for _, row in year_df.iterrows():
            w = min(int(row["week"]) - 1, 53)
            d = int(row["dow"])
            grid[d, w] = row[value_col]

        fig_s, ax = plt.subplots(figsize=(14, 3))
        im = ax.imshow(grid, aspect="auto", cmap="YlGn",
                       interpolation="nearest")
        plt.colorbar(im, ax=ax, label=value_col)
        ax.set_yticks(range(7))
        ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_xlabel("Week of year")
        ax.set_title(f"Calendar Heatmap: {value_col} ({latest_year})")
        fig_s.tight_layout()

        # Plotly version
        fig_p = go.Figure(go.Heatmap(
            z=grid,
            colorscale="YlGn",
            showscale=True,
        ))
        fig_p.update_layout(title=f"Calendar Heatmap ({latest_year})",
                            xaxis_title="Week",
                            yaxis=dict(ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                                       tickvals=list(range(7))))

        code = f"""# Calendar Heatmap
# Requires date column and value column
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df['{date_col}'] = pd.to_datetime(df['{date_col}'])
year_df = df[df['{date_col}'].dt.year == df['{date_col}'].dt.year.max()].copy()
year_df['week'] = year_df['{date_col}'].dt.isocalendar().week.astype(int)
year_df['dow'] = year_df['{date_col}'].dt.dayofweek
# Build 7x54 grid and plot with imshow
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Candlestick Chart ──────────────────────────────────────────────────

def candlestick_chart(df, date_col: str, open_col: str, high_col: str,
                      low_col: str, close_col: str):
    """OHLC candlestick chart using plotly"""
    try:
        df2 = df[[date_col, open_col, high_col, low_col, close_col]].dropna()
        df2 = _sort_by_x(df2, date_col)

        fig_p = go.Figure(go.Candlestick(
            x=df2[date_col],
            open=df2[open_col],
            high=df2[high_col],
            low=df2[low_col],
            close=df2[close_col],
            name="OHLC",
        ))
        fig_p.update_layout(title="Candlestick Chart",
                            xaxis_title=date_col,
                            yaxis_title="Price")

        # Static matplotlib version
        import matplotlib.patches as mpatches
        fig_s, ax = plt.subplots(figsize=(12, 5))
        x_idx = range(len(df2))
        for i, (_, row) in enumerate(df2.iterrows()):
            color = "green" if row[close_col] >= row[open_col] else "red"
            # Wick
            ax.plot([i, i], [row[low_col], row[high_col]], color=color, lw=1)
            # Body
            body_h = abs(row[close_col] - row[open_col])
            body_b = min(row[open_col], row[close_col])
            ax.add_patch(mpatches.Rectangle(
                (i - 0.3, body_b), 0.6, body_h,
                color=color, alpha=0.8
            ))
        ax.set_xticks(range(0, len(df2), max(1, len(df2) // 10)))
        ax.set_xticklabels(
            [str(df2.iloc[i][date_col])[:10]
             for i in range(0, len(df2), max(1, len(df2) // 10))],
            rotation=45, ha="right"
        )
        ax.set_title("Candlestick Chart")
        fig_s.tight_layout()

        code = f"""# Candlestick Chart
import plotly.graph_objects as go
fig = go.Figure(go.Candlestick(
    x=df['{date_col}'],
    open=df['{open_col}'],
    high=df['{high_col}'],
    low=df['{low_col}'],
    close=df['{close_col}'],
))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Spiral Chart ────────────────────────────────────────────────────────

def spiral_chart(df, date_col: str, value_col: str):
    """Spiral/polar time series chart"""
    try:
        import numpy as np
        df2 = df[[date_col, value_col]].dropna().copy()
        df2[date_col] = pd.to_datetime(df2[date_col], errors="coerce")
        df2 = df2.dropna(subset=[date_col]).sort_values(date_col)

        n = len(df2)
        if n < 5:
            return None, None, "# Need at least 5 data points"

        # Map time to angle (full rotation per year or per cycle)
        t = np.linspace(0, 4 * np.pi, n)  # 2 full rotations
        r = df2[value_col].values.astype(float)
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9)
        radii = 1 + r_norm  # radius 1-2

        theta = t
        x = radii * np.cos(theta)
        y = radii * np.sin(theta)

        fig_s, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
        sc = ax.scatter(theta, radii, c=r_norm, cmap="viridis", s=10, alpha=0.8)
        ax.plot(theta, radii, lw=0.5, alpha=0.4, color="gray")
        plt.colorbar(sc, ax=ax, label=value_col)
        ax.set_title(f"Spiral Chart: {value_col} over time", pad=20)
        fig_s.tight_layout()

        fig_p = go.Figure(go.Scatterpolar(
            r=radii.tolist(),
            theta=np.degrees(theta).tolist(),
            mode="markers",
            marker=dict(color=r_norm.tolist(), colorscale="Viridis",
                        size=5, showscale=True),
        ))
        fig_p.update_layout(title=f"Spiral Chart: {value_col} over time")

        code = f"""# Spiral Chart
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(0, 4*np.pi, len(df))
r = df['{value_col}'].values
r_norm = (r - r.min()) / (r.max() - r.min() + 1e-9) + 1
fig, ax = plt.subplots(subplot_kw={{'polar': True}}, figsize=(8, 8))
ax.scatter(t, r_norm, c=r_norm, cmap='viridis', s=10)
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
