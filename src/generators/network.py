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
        src, tgt, val = source_col, target_col, value_col
        df_agg = df.groupby([src, tgt], as_index=False)[val].sum()
        all_nodes = list(pd.concat([df_agg[src], df_agg[tgt]]).astype(str).unique())
        node_idx = {n: i for i, n in enumerate(all_nodes)}

        fig_p = go.Figure(go.Sankey(
            node=dict(
                label=all_nodes,
                pad=15,
                thickness=20,
            ),
            link=dict(
                source=[node_idx[str(s)] for s in df_agg[src]],
                target=[node_idx[str(t)] for t in df_agg[tgt]],
                value=df_agg[val].tolist(),
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


# ── Chord Diagram ─────────────────────────────────────────────────────

def chord_diagram(df: pd.DataFrame, source_col: str, target_col: str, value_col: str):
    """Chord diagram using plotly Sankey as chord-like layout"""
    try:
        all_nodes = list(pd.concat([df[source_col], df[target_col]]).astype(str).unique())
        n_nodes = len(all_nodes)
        node_idx = {nd: i for i, nd in enumerate(all_nodes)}

        # Build adjacency matrix
        matrix = np.zeros((n_nodes, n_nodes))
        for _, row in df.iterrows():
            s = node_idx[str(row[source_col])]
            t = node_idx[str(row[target_col])]
            matrix[s][t] += float(row[value_col])

        # Plotly Sankey
        fig_p = go.Figure(go.Sankey(
            node=dict(label=all_nodes, pad=15, thickness=20),
            link=dict(
                source=[node_idx[str(s)] for s in df[source_col]],
                target=[node_idx[str(t)] for t in df[target_col]],
                value=df[value_col].tolist(),
            )
        ))
        fig_p.update_layout(title="Chord Diagram")

        # Static: nodes on a circle with chord lines
        fig_s, ax = plt.subplots(figsize=(9, 9))
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        colors = plt.cm.tab20.colors
        import matplotlib.patches as mpatches

        for i, (nd, ang) in enumerate(zip(all_nodes, angles)):
            xi, yi = np.cos(ang), np.sin(ang)
            ax.plot(xi * 1.05, yi * 1.05, "o",
                    color=colors[i % len(colors)], markersize=12)
            ax.text(xi * 1.18, yi * 1.18, str(nd)[:10],
                    ha="center", va="center", fontsize=7)

        max_val = matrix.max() + 1e-9
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if matrix[i][j] + matrix[j][i] > 0:
                    xi, yi = np.cos(angles[i]), np.sin(angles[i])
                    xj, yj = np.cos(angles[j]), np.sin(angles[j])
                    lw = (matrix[i][j] + matrix[j][i]) / max_val * 4
                    ax.plot([xi, xj], [yi, yj],
                            color=colors[i % len(colors)],
                            alpha=0.4, lw=max(0.5, lw))

        circle = plt.Circle((0, 0), 1, fill=False, edgecolor="gray", lw=0.5)
        ax.add_patch(circle)
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Chord Diagram")
        fig_s.tight_layout()

        code = """# Chord Diagram
import plotly.graph_objects as go
fig = go.Figure(go.Sankey(
    node=dict(label=nodes),
    link=dict(source=sources, target=targets, value=values)
))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Arc Diagram ───────────────────────────────────────────────────────

def arc_diagram(df: pd.DataFrame, source_col: str, target_col: str,
                value_col: str = None):
    """Arc diagram — nodes on a line, arcs above"""
    try:
        import matplotlib.patches as mpatches
        all_nodes = list(pd.concat([df[source_col], df[target_col]]).astype(str).unique())
        node_idx = {nd: i for i, nd in enumerate(all_nodes)}
        n_nodes = len(all_nodes)
        colors = plt.cm.tab10.colors

        fig_s, ax = plt.subplots(figsize=(max(10, n_nodes * 0.8), 6))
        for i, nd in enumerate(all_nodes):
            ax.plot(i, 0, "o", color="steelblue", markersize=12, zorder=5)
            ax.text(i, -0.15, str(nd)[:10], ha="center", va="top", fontsize=8)

        max_val = df[value_col].max() if (value_col and value_col in df.columns) else 1
        for _, row in df.iterrows():
            s = node_idx[str(row[source_col])]
            t = node_idx[str(row[target_col])]
            if s == t:
                continue
            lw = 1.5
            if value_col and value_col in df.columns:
                lw = min(float(row[value_col]) / (max_val + 1e-9) * 5, 5)
            arc = mpatches.Arc(((s + t) / 2, 0), abs(t - s), abs(t - s),
                               angle=0, theta1=0, theta2=180,
                               color=colors[s % len(colors)], lw=lw, alpha=0.7)
            ax.add_patch(arc)

        ax.set_xlim(-0.5, n_nodes - 0.5)
        ax.set_ylim(-0.3, max(1.5, n_nodes / 2))
        ax.axis("off")
        ax.set_title("Arc Diagram")
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=list(range(n_nodes)),
            y=[0] * n_nodes,
            mode="markers+text",
            text=all_nodes,
            textposition="bottom center",
            marker=dict(size=15, color="steelblue"),
        ))
        fig_p.update_layout(title="Arc Diagram",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        code = """# Arc Diagram
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(12, 6))
for s, t in zip(sources, targets):
    arc = mpatches.Arc(((s+t)/2, 0), abs(t-s), abs(t-s),
                       theta1=0, theta2=180, color='steelblue', alpha=0.7)
    ax.add_patch(arc)
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Alluvial Diagram ──────────────────────────────────────────────────

def alluvial_diagram(df: pd.DataFrame, stage_cols: list, value_col: str = None):
    """Alluvial/parallel sets diagram using plotly"""
    try:
        valid_cols = [c for c in stage_cols if c in df.columns]
        if len(valid_cols) < 2:
            return None, None, "# Need at least 2 valid stage columns"

        plot_df = df[valid_cols].astype(str).copy()
        dimensions = [dict(values=plot_df[c], label=c) for c in valid_cols]

        fig_p = go.Figure(go.Parcats(
            dimensions=dimensions,
        ))
        fig_p.update_layout(title="Alluvial Diagram")

        fig_s = None  # Best shown interactively

        code = f"""# Alluvial Diagram
import plotly.graph_objects as go
dimensions = [dict(values=df[c].astype(str), label=c) for c in {valid_cols}]
fig = go.Figure(go.Parcats(dimensions=dimensions))
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
