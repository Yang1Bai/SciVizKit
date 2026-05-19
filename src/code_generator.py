"""
SciVizKit — Code Generator
Returns ready-to-copy Python code snippets for each chart type.
"""

from __future__ import annotations
from typing import Optional


def get_code(chart_id: str, **kwargs) -> str:
    """Return a code string for the given chart_id with column names filled in."""
    generators = {
        "histogram": _histogram,
        "kde": _kde,
        "violin": _violin,
        "boxplot": _boxplot,
        "stripplot": _stripplot,
        "beeswarm": _beeswarm,
        "ecdf": _ecdf,
        "qqplot": _qqplot,
        "bar": _bar,
        "grouped_bar": _grouped_bar,
        "stacked_bar": _stacked_bar,
        "lollipop": _lollipop,
        "dumbbell": _dumbbell,
        "dotplot": _dotplot,
        "slope": _slope,
        "waterfall": _waterfall,
        "errorbar": _errorbar,
        "scatter": _scatter,
        "bubble": _bubble,
        "hexbin": _hexbin,
        "corr_heatmap": _corr_heatmap,
        "pairplot": _pairplot,
        "parallel_coords": _parallel_coords,
        "line": _line,
        "area": _area,
        "stacked_area": _stacked_area,
        "step_line": _step_line,
        "pie": _pie,
        "donut": _donut,
        "treemap": _treemap,
        "sunburst": _sunburst,
        "nightingale": _nightingale,
        "sankey": _sankey,
        "network_graph": _network_graph,
        "dendrogram": _dendrogram,
        "volcano": _volcano,
        "pca_plot": _pca_plot,
        "roc_curve": _roc_curve,
        "radar": _radar,
        "parity_plot": _parity_plot,
        "bland_altman": _bland_altman,
        "kaplan_meier": _kaplan_meier,
    }
    fn = generators.get(chart_id)
    if fn:
        try:
            return fn(**kwargs)
        except Exception:
            return f"# Code generation failed for chart_id={chart_id}"
    return f"# No code template for chart_id={chart_id}"


# ── Distribution ────────────────────────────────────────────────────────

def _histogram(x_col="value", color_col=None, **kw):
    color_arg = f', color="{color_col}"' if color_col else ""
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.histogram(df, x="{x_col}"{color_arg},
                   nbins=30, marginal="box",
                   title="Histogram of {x_col}")
fig.update_layout(bargap=0.1)
fig.show()
"""

def _kde(x_col="value", color_col=None, **kw):
    hue = f', hue="{color_col}"' if color_col else ""
    return f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.kdeplot(data=df, x="{x_col}"{hue}, fill=True, ax=ax)
ax.set_title("KDE Plot of {x_col}")
plt.tight_layout()
plt.show()
"""

def _violin(x_col="group", y_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.violin(df, x="{x_col}", y="{y_col}",
                box=True, points="all",
                title="Violin Plot")
fig.show()
"""

def _boxplot(x_col="group", y_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.box(df, x="{x_col}", y="{y_col}",
             points="outliers", title="Box Plot")
fig.show()
"""

def _stripplot(x_col="group", y_col="value", **kw):
    return f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.stripplot(data=df, x="{x_col}", y="{y_col}", jitter=True, ax=ax)
ax.set_title("Strip Plot")
plt.tight_layout()
plt.show()
"""

def _beeswarm(x_col="group", y_col="value", **kw):
    return f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 5))
sns.swarmplot(data=df, x="{x_col}", y="{y_col}", ax=ax)
ax.set_title("Beeswarm Plot")
plt.tight_layout()
plt.show()
"""

def _ecdf(x_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.ecdf(df, x="{x_col}", title="ECDF of {x_col}")
fig.show()
"""

def _qqplot(x_col="value", **kw):
    return f"""\
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(df["{x_col}"].dropna(), dist="norm", plot=ax)
ax.set_title("Q-Q Plot of {x_col}")
plt.tight_layout()
plt.show()
"""

# ── Comparison ────────────────────────────────────────────────────────

def _bar(x_col="category", y_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.bar(df, x="{x_col}", y="{y_col}", title="Bar Chart")
fig.show()
"""

def _grouped_bar(x_col="category", y_col="value", color_col="group", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.bar(df, x="{x_col}", y="{y_col}", color="{color_col}",
             barmode="group", title="Grouped Bar Chart")
fig.show()
"""

def _stacked_bar(x_col="category", y_col="value", color_col="group", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.bar(df, x="{x_col}", y="{y_col}", color="{color_col}",
             barmode="stack", title="Stacked Bar Chart")
fig.show()
"""

def _lollipop(x_col="category", y_col="value", **kw):
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
df = df.sort_values("{y_col}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.hlines(df["{x_col}"], 0, df["{y_col}"], colors="steelblue", linewidth=2)
ax.plot(df["{y_col}"], df["{x_col}"], "o", color="steelblue", markersize=8)
ax.set_title("Lollipop Chart")
plt.tight_layout()
plt.show()
"""

def _dumbbell(label_col="label", val1_col="value1", val2_col="value2", **kw):
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 6))
for _, row in df.iterrows():
    ax.plot([row["{val1_col}"], row["{val2_col}"]], [row["{label_col}"], row["{label_col}"]],
            "o-", color="gray", linewidth=2)
ax.set_title("Dumbbell Chart")
plt.tight_layout()
plt.show()
"""

def _dotplot(x_col="category", y_col="value", **kw):
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
df = df.sort_values("{y_col}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df["{y_col}"], df["{x_col}"], "o", color="steelblue", markersize=10)
ax.set_title("Dot Plot")
plt.tight_layout()
plt.show()
"""

def _slope(label_col="label", val1_col="value1", val2_col="value2", **kw):
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(6, 8))
for _, row in df.iterrows():
    ax.plot([1, 2], [row["{val1_col}"], row["{val2_col}"]], "o-", linewidth=2)
    ax.text(1 - 0.05, row["{val1_col}"], row["{label_col}"], ha="right", fontsize=9)
ax.set_xticks([1, 2])
ax.set_xticklabels(["{val1_col}", "{val2_col}"])
ax.set_title("Slope Chart")
plt.tight_layout()
plt.show()
"""

def _waterfall(label_col="label", value_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("your_data.csv")

fig = go.Figure(go.Waterfall(
    name="Waterfall",
    orientation="v",
    x=df["{label_col}"].tolist(),
    y=df["{value_col}"].tolist(),
))
fig.update_layout(title="Waterfall Chart")
fig.show()
"""

def _errorbar(x_col="category", y_col="value", err_col="error", **kw):
    return f"""\
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("your_data.csv")

fig = go.Figure(go.Bar(
    x=df["{x_col}"],
    y=df["{y_col}"],
    error_y=dict(type="data", array=df["{err_col}"].tolist()),
))
fig.update_layout(title="Error Bar Plot")
fig.show()
"""

# ── Correlation ────────────────────────────────────────────────────────

def _scatter(x_col="x", y_col="y", color_col=None, **kw):
    color_arg = f', color="{color_col}"' if color_col else ""
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.scatter(df, x="{x_col}", y="{y_col}"{color_arg},
                 trendline="ols", title="Scatter Plot")
fig.show()
"""

def _bubble(x_col="x", y_col="y", size_col="size", color_col=None, **kw):
    color_arg = f', color="{color_col}"' if color_col else ""
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.scatter(df, x="{x_col}", y="{y_col}", size="{size_col}"{color_arg},
                 title="Bubble Chart")
fig.show()
"""

def _hexbin(x_col="x", y_col="y", **kw):
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 6))
hb = ax.hexbin(df["{x_col}"], df["{y_col}"], gridsize=30, cmap="Blues")
fig.colorbar(hb, ax=ax, label="count")
ax.set_xlabel("{x_col}")
ax.set_ylabel("{y_col}")
ax.set_title("Hexbin Plot")
plt.tight_layout()
plt.show()
"""

def _corr_heatmap(**kw):
    return """\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
numeric_df = df.select_dtypes(include="number")

fig, ax = plt.subplots(figsize=(10, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.show()
"""

def _pairplot(cols=None, **kw):
    cols_str = str(cols) if cols else "None  # list of column names"
    return f"""\
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
cols = {cols_str}

g = sns.pairplot(df[cols] if cols else df.select_dtypes("number"),
                 diag_kind="kde", plot_kws={{"alpha": 0.5}})
g.figure.suptitle("Pair Plot", y=1.02)
plt.show()
"""

def _parallel_coords(cols=None, color_col=None, **kw):
    return """\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")
numeric_cols = df.select_dtypes(include="number").columns.tolist()

fig = px.parallel_coordinates(df, dimensions=numeric_cols,
                               color=numeric_cols[0],
                               title="Parallel Coordinates")
fig.show()
"""

# ── Time Series ────────────────────────────────────────────────────────

def _line(x_col="date", y_cols=None, **kw):
    y = y_cols[0] if (y_cols and len(y_cols) > 0) else "value"
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv", parse_dates=["{x_col}"])
df = df.sort_values("{x_col}")

fig = px.line(df, x="{x_col}", y="{y}", title="Line Chart")
fig.show()
"""

def _area(x_col="date", y_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv", parse_dates=["{x_col}"])
df = df.sort_values("{x_col}")

fig = px.area(df, x="{x_col}", y="{y_col}", title="Area Chart")
fig.show()
"""

def _stacked_area(x_col="date", y_cols=None, **kw):
    y = str(y_cols) if y_cols else '["value1", "value2"]'
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv", parse_dates=["{x_col}"])
df = df.sort_values("{x_col}")
y_cols = {y}

fig = px.area(df, x="{x_col}", y=y_cols, title="Stacked Area Chart")
fig.show()
"""

def _step_line(x_col="date", y_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv", parse_dates=["{x_col}"])
df = df.sort_values("{x_col}")

fig = px.line(df, x="{x_col}", y="{y_col}", line_shape="hv",
              title="Step Line Chart")
fig.show()
"""

# ── Proportional ────────────────────────────────────────────────────────

def _pie(label_col="label", value_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.pie(df, names="{label_col}", values="{value_col}", title="Pie Chart")
fig.show()
"""

def _donut(label_col="label", value_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.pie(df, names="{label_col}", values="{value_col}",
             hole=0.4, title="Donut Chart")
fig.show()
"""

def _treemap(label_col="label", value_col="value", **kw):
    return f"""\
import pandas as pd
import squarify
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
agg = df.groupby("{label_col}")["{value_col}"].sum().reset_index()

fig, ax = plt.subplots(figsize=(10, 7))
squarify.plot(sizes=agg["{value_col}"], label=agg["{label_col}"],
              alpha=0.8, ax=ax)
ax.axis("off")
ax.set_title("Treemap")
plt.tight_layout()
plt.show()
"""

def _sunburst(label_col="label", parent_col="parent", value_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.sunburst(df, names="{label_col}", parents="{parent_col}",
                  values="{value_col}", title="Sunburst Chart")
fig.show()
"""

def _nightingale(label_col="label", value_col="value", **kw):
    return f"""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
agg = df.groupby("{label_col}")["{value_col}"].sum().reset_index()

N = len(agg)
theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
radii = agg["{value_col}"].values
width = 2 * np.pi / N

fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
bars = ax.bar(theta, radii, width=width * 0.9, bottom=0.0)
ax.set_xticks(theta)
ax.set_xticklabels(agg["{label_col}"], fontsize=9)
ax.set_title("Nightingale Rose Chart", pad=20)
plt.tight_layout()
plt.show()
"""

# ── Network ────────────────────────────────────────────────────────────

def _sankey(source_col="source", target_col="target", value_col="value", **kw):
    return f"""\
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("your_data.csv")

all_nodes = list(pd.concat([df["{source_col}"], df["{target_col}"]]).unique())
node_idx = {{n: i for i, n in enumerate(all_nodes)}}

fig = go.Figure(go.Sankey(
    node=dict(label=all_nodes),
    link=dict(
        source=[node_idx[s] for s in df["{source_col}"]],
        target=[node_idx[t] for t in df["{target_col}"]],
        value=df["{value_col}"].tolist(),
    )
))
fig.update_layout(title="Sankey Diagram")
fig.show()
"""

def _network_graph(source_col="source", target_col="target", **kw):
    return f"""\
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

G = nx.from_pandas_edgelist(df, source="{source_col}", target="{target_col}")
pos = nx.spring_layout(G, seed=42)

fig, ax = plt.subplots(figsize=(10, 8))
nx.draw_networkx(G, pos=pos, ax=ax, node_color="steelblue",
                 node_size=500, font_size=8, edge_color="gray")
ax.set_title("Network Graph")
ax.axis("off")
plt.tight_layout()
plt.show()
"""

def _dendrogram(cols=None, **kw):
    return """\
import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
numeric_df = df.select_dtypes(include="number")

Z = sch.linkage(numeric_df.T, method="ward")

fig, ax = plt.subplots(figsize=(12, 6))
sch.dendrogram(Z, labels=numeric_df.columns.tolist(), ax=ax,
               leaf_rotation=45)
ax.set_title("Dendrogram")
plt.tight_layout()
plt.show()
"""

# ── Scientific ──────────────────────────────────────────────────────────

def _volcano(fc_col="log2FoldChange", pval_col="pvalue", **kw):
    return f"""\
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("your_data.csv")
df["-log10p"] = -np.log10(df["{pval_col}"].clip(lower=1e-300))

# Label significance
def label(row):
    if abs(row["{fc_col}"]) > 1 and row["-log10p"] > -np.log10(0.05):
        return "Significant"
    return "Not significant"

df["sig"] = df.apply(label, axis=1)

fig = px.scatter(df, x="{fc_col}", y="-log10p", color="sig",
                 hover_name=df.columns[0],
                 color_discrete_map={{"Significant": "red", "Not significant": "gray"}},
                 title="Volcano Plot")
fig.add_vline(x=1, line_dash="dash", line_color="blue")
fig.add_vline(x=-1, line_dash="dash", line_color="blue")
fig.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="green")
fig.show()
"""

def _pca_plot(feature_cols=None, color_col=None, **kw):
    color_arg = f', color=df["{color_col}"]' if color_col else ""
    return f"""\
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px

df = pd.read_csv("your_data.csv")
numeric_df = df.select_dtypes(include="number")

scaler = StandardScaler()
X = scaler.fit_transform(numeric_df.fillna(0))

pca = PCA(n_components=2)
components = pca.fit_transform(X)
ev = pca.explained_variance_ratio_

pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
{f'pca_df["group"] = df["{color_col}"].values' if color_col else ""}

fig = px.scatter(pca_df, x="PC1", y="PC2"{'color="group"' if color_col else ""},
                 labels={{"PC1": f"PC1 ({{ev[0]*100:.1f}}%)", "PC2": f"PC2 ({{ev[1]*100:.1f}}%)"}},
                 title="PCA Plot")
fig.show()
"""

def _roc_curve(y_true_col="y_true", y_score_col="y_score", **kw):
    return f"""\
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go

df = pd.read_csv("your_data.csv")

fpr, tpr, _ = roc_curve(df["{y_true_col}"], df["{y_score_col}"])
roc_auc = auc(fpr, tpr)

fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                         name=f"ROC curve (AUC = {{roc_auc:.3f}})"))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(dash="dash"), name="Random"))
fig.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    title="ROC Curve"
)
fig.show()
"""

def _radar(label_col="label", value_cols=None, **kw):
    return f"""\
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("your_data.csv")
value_cols = {str(value_cols) if value_cols else "df.select_dtypes('number').columns.tolist()"}

fig = go.Figure()
for _, row in df.iterrows():
    vals = [row[c] for c in value_cols]
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=value_cols + [value_cols[0]],
        fill="toself",
        name=str(row["{label_col}"]),
    ))
fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                  title="Radar Chart")
fig.show()
"""

def _parity_plot(actual_col="actual", predicted_col="predicted", **kw):
    return f"""\
import pandas as pd
import numpy as np
import plotly.express as px

df = pd.read_csv("your_data.csv")

fig = px.scatter(df, x="{actual_col}", y="{predicted_col}",
                 title="Parity Plot (Actual vs Predicted)")
mn = min(df["{actual_col}"].min(), df["{predicted_col}"].min())
mx = max(df["{actual_col}"].max(), df["{predicted_col}"].max())
fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
              line=dict(color="red", dash="dash"))
fig.show()
"""

def _bland_altman(method1_col="method1", method2_col="method2", **kw):
    return f"""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

mean = (df["{method1_col}"] + df["{method2_col}"]) / 2
diff = df["{method1_col}"] - df["{method2_col}"]
md = diff.mean()
sd = diff.std()

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(mean, diff, alpha=0.6)
ax.axhline(md, color="blue", label=f"Mean diff: {{md:.2f}}")
ax.axhline(md + 1.96 * sd, color="red", linestyle="--", label="+1.96 SD")
ax.axhline(md - 1.96 * sd, color="red", linestyle="--", label="-1.96 SD")
ax.set_xlabel("Mean of methods")
ax.set_ylabel("Difference")
ax.set_title("Bland-Altman Plot")
ax.legend()
plt.tight_layout()
plt.show()
"""

def _kaplan_meier(duration_col="duration", event_col="event", group_col=None, **kw):
    group_code = ""
    if group_col:
        group_code = f"""\
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
for name, grp in df.groupby("{group_col}"):
    kmf.fit(grp["{duration_col}"], grp["{event_col}"], label=str(name))
    kmf.plot_survival_function(ax=ax)
"""
    else:
        group_code = f"""\
from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()
kmf.fit(df["{duration_col}"], df["{event_col}"], label="All")
kmf.plot_survival_function(ax=ax)
"""
    return f"""\
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

fig, ax = plt.subplots(figsize=(8, 6))
{group_code}
ax.set_title("Kaplan-Meier Survival Curve")
ax.set_xlabel("Time")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
plt.show()
"""
