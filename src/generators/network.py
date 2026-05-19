"""
SciVizKit — Network generators
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
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from ..code_generator import get_code


# ── Sankey Diagram ────────────────────────────────────────────────────

def sankey_diagram(df: pd.DataFrame, source_col: str, target_col: str, value_col: str):
    try:
        all_nodes = list(pd.concat([df[source_col], df[target_col]]).astype(str).unique())
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        fig_p = go.Figure(go.Sankey(
            node=dict(
                label=all_nodes,
                pad=15,
                thickness=20,
            ),
            link=dict(
                source=[node_idx[str(s)] for s in df[source_col]],
                target=[node_idx[str(t)] for t in df[target_col]],
                value=df[value_col].tolist(),
            )
        ))
        fig_p.update_layout(title="Sankey Diagram", font_size=12)
        code = get_code("sankey", source_col=source_col, target_col=target_col, value_col=value_col)
        return None, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Network Graph ─────────────────────────────────────────────────────

def network_graph(df: pd.DataFrame, source_col: str, target_col: str):
    try:
        if not HAS_NX:
            return None, None, "# networkx not installed"

        G = nx.from_pandas_edgelist(df.head(200), source=source_col, target=target_col)
        pos = nx.spring_layout(G, seed=42)

        fig_s, ax = plt.subplots(figsize=(10, 8))
        nx.draw_networkx(G, pos=pos, ax=ax,
                         node_color="#667eea", node_size=300,
                         font_size=7, edge_color="gray", alpha=0.8)
        ax.set_title("Network Graph")
        ax.axis("off")
        fig_s.tight_layout()

        # Plotly version
        edge_x, edge_y = [], []
        for e0, e1 in G.edges():
            x0, y0 = pos[e0]
            x1, y1 = pos[e1]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G.nodes()]
        node_y = [pos[n][1] for n in G.nodes()]
        node_labels = [str(n) for n in G.nodes()]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                   line=dict(color="gray", width=1), hoverinfo="none"))
        fig_p.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                   text=node_labels, textposition="top center",
                                   marker=dict(size=10, color="#667eea"),
                                   hoverinfo="text"))
        fig_p.update_layout(title="Network Graph",
                            showlegend=False,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        code = get_code("network_graph", source_col=source_col, target_col=target_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Dendrogram ────────────────────────────────────────────────────────

def dendrogram_plot(df: pd.DataFrame, cols: list):
    try:
        num_df = df[cols].select_dtypes("number") if cols else df.select_dtypes("number")
        num_df = num_df.dropna(axis=1).head(200)
        if num_df.shape[1] < 2:
            return None, None, "# Need at least 2 numeric columns"

        Z = sch.linkage(num_df.T, method="ward")

        fig_s, ax = plt.subplots(figsize=(max(8, num_df.shape[1] * 0.6), 6))
        sch.dendrogram(Z, labels=num_df.columns.tolist(), ax=ax, leaf_rotation=45)
        ax.set_title("Dendrogram (feature clustering)")
        fig_s.tight_layout()

        code = get_code("dendrogram", cols=cols)
        return fig_s, None, code
    except Exception as e:
        return None, None, f"# Error: {e}"
