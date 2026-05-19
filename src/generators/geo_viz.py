"""
SciVizKit — Geographic visualization generators
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


# ── Choropleth Map ────────────────────────────────────────────────────

def choropleth_map(df: pd.DataFrame, location_col: str, value_col: str,
                   location_type: str = "country"):
    """Choropleth map using plotly express."""
    try:
        agg = df.groupby(location_col)[value_col].mean().reset_index()

        # Try ISO-3 first, then country names
        locationmode = "ISO-3"
        if location_type in ("country", "name"):
            locationmode = "country names"
        elif location_type in ("us-states", "USA-states"):
            locationmode = "USA-states"

        fig_p = px.choropleth(
            agg,
            locations=location_col,
            color=value_col,
            locationmode=locationmode,
            color_continuous_scale="Viridis",
            title=f"Choropleth Map: {value_col}",
        )
        fig_p.update_geos(showframe=False, showcoastlines=True)

        # Static: simple bar chart as fallback (choropleth needs geo data)
        fig_s, ax = plt.subplots(figsize=(12, 5))
        agg_sorted = agg.sort_values(value_col, ascending=False).head(20)
        ax.barh(agg_sorted[location_col].astype(str),
                agg_sorted[value_col], color="steelblue")
        ax.set_xlabel(value_col)
        ax.set_title(f"Choropleth Data: {value_col} by {location_col}")
        fig_s.tight_layout()

        code = f"""# Choropleth Map
import plotly.express as px
fig = px.choropleth(
    df,
    locations='{location_col}',
    color='{value_col}',
    locationmode='country names',
    color_continuous_scale='Viridis',
    title='Choropleth Map'
)
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Bubble Map ────────────────────────────────────────────────────────

def bubble_map(df: pd.DataFrame, lat_col: str, lon_col: str, size_col: str,
               color_col: str = None):
    """Bubble map using plotly express."""
    try:
        clean = df[[lat_col, lon_col, size_col] +
                   ([color_col] if color_col and color_col in df.columns else [])].dropna()

        color_arg = color_col if (color_col and color_col in df.columns) else None

        fig_p = px.scatter_geo(
            clean,
            lat=lat_col,
            lon=lon_col,
            size=size_col,
            color=color_arg,
            projection="natural earth",
            size_max=30,
            title=f"Bubble Map: {size_col}",
        )
        fig_p.update_geos(showcoastlines=True, showland=True, landcolor="lightgray")

        # Static: scatter plot as proxy
        fig_s, ax = plt.subplots(figsize=(12, 6))
        sizes = (clean[size_col] - clean[size_col].min()) / \
                (clean[size_col].max() - clean[size_col].min() + 1e-9) * 300 + 20
        sc = ax.scatter(clean[lon_col], clean[lat_col], s=sizes,
                        alpha=0.6, c=sizes, cmap="viridis")
        plt.colorbar(sc, ax=ax, label=size_col)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Bubble Map: {size_col}")
        ax.grid(alpha=0.3)
        fig_s.tight_layout()

        code = f"""# Bubble Map
import plotly.express as px
fig = px.scatter_geo(
    df,
    lat='{lat_col}',
    lon='{lon_col}',
    size='{size_col}',
    projection='natural earth',
    size_max=30,
    title='Bubble Map'
)
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
