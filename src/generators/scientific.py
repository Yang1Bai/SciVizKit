"""
SciVizKit — Scientific generators
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

from ..code_generator import get_code

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


# ── Volcano Plot ──────────────────────────────────────────────────────

def volcano_plot(df: pd.DataFrame, fc_col: str, pval_col: str,
                 fc_thresh: float = 1.0, pval_thresh: float = 0.05):
    try:
        df2 = df[[fc_col, pval_col]].dropna().copy()
        df2["neg_log10p"] = -np.log10(df2[pval_col].clip(lower=1e-300))
        log_thresh = -np.log10(pval_thresh)

        def classify(row):
            if abs(row[fc_col]) > fc_thresh and row["neg_log10p"] > log_thresh:
                return "Significant"
            return "Not significant"

        df2["sig"] = df2.apply(classify, axis=1)
        palette = {"Significant": "#e41a1c", "Not significant": "#aaaaaa"}

        fig_s, ax = plt.subplots(figsize=(9, 7))
        for sig, sub in df2.groupby("sig"):
            ax.scatter(sub[fc_col], sub["neg_log10p"],
                       c=palette[sig], alpha=0.5, s=20, label=sig)
        ax.axvline(fc_thresh, color="blue", linestyle="--", lw=1)
        ax.axvline(-fc_thresh, color="blue", linestyle="--", lw=1)
        ax.axhline(log_thresh, color="green", linestyle="--", lw=1)
        ax.set_xlabel(f"log₂ Fold Change ({fc_col})")
        ax.set_ylabel(f"-log₁₀(p-value) ({pval_col})")
        ax.set_title("Volcano Plot")
        ax.legend()
        fig_s.tight_layout()

        color_map = {"Significant": "#e41a1c", "Not significant": "#aaaaaa"}
        fig_p = px.scatter(df2, x=fc_col, y="neg_log10p", color="sig",
                           color_discrete_map=color_map,
                           labels={"neg_log10p": "-log₁₀(p-value)"},
                           title="Volcano Plot",
                           opacity=0.6)
        fig_p.add_vline(x=fc_thresh, line_dash="dash", line_color="blue")
        fig_p.add_vline(x=-fc_thresh, line_dash="dash", line_color="blue")
        fig_p.add_hline(y=log_thresh, line_dash="dash", line_color="green")

        code = get_code("volcano", fc_col=fc_col, pval_col=pval_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── PCA Plot ──────────────────────────────────────────────────────────

def pca_plot(df: pd.DataFrame, feature_cols: list, color_col: str = None):
    try:
        num_df = df[feature_cols].select_dtypes("number").dropna()
        if num_df.shape[1] < 2 or num_df.shape[0] < 5:
            return None, None, "# Need at least 2 numeric features and 5 rows"

        X = StandardScaler().fit_transform(num_df.fillna(0))
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        ev = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=num_df.index)
        pc1_label = f"PC1 ({ev[0]*100:.1f}%)"
        pc2_label = f"PC2 ({ev[1]*100:.1f}%)"

        color_vals = None
        if color_col and color_col in df.columns:
            pca_df["group"] = df.loc[num_df.index, color_col].values
            color_vals = "group"

        fig_s, ax = plt.subplots(figsize=(8, 6))
        if color_vals:
            groups = pca_df["group"].unique()
            for g in groups:
                sub = pca_df[pca_df["group"] == g]
                ax.scatter(sub["PC1"], sub["PC2"], alpha=0.7, label=str(g), s=40)
            ax.legend()
        else:
            ax.scatter(pca_df["PC1"], pca_df["PC2"], alpha=0.7, color="steelblue", s=40)
        ax.set_xlabel(pc1_label)
        ax.set_ylabel(pc2_label)
        ax.set_title("PCA Plot")
        fig_s.tight_layout()

        fig_p = px.scatter(pca_df, x="PC1", y="PC2", color=color_vals,
                           labels={"PC1": pc1_label, "PC2": pc2_label},
                           title="PCA Plot")
        code = get_code("pca_plot", feature_cols=feature_cols, color_col=color_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── ROC Curve ─────────────────────────────────────────────────────────

def roc_curve_plot(df: pd.DataFrame, y_true_col: str, y_score_col: str):
    try:
        clean = df[[y_true_col, y_score_col]].dropna()
        y_true = clean[y_true_col].values
        y_score = clean[y_score_col].values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        fig_s, ax = plt.subplots(figsize=(7, 7))
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                   name=f"AUC = {roc_auc:.3f}",
                                   line=dict(color="darkorange", width=2)))
        fig_p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(dash="dash", color="navy"), name="Random"))
        fig_p.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curve"
        )
        code = get_code("roc_curve", y_true_col=y_true_col, y_score_col=y_score_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Radar Chart ───────────────────────────────────────────────────────

def radar_chart(df: pd.DataFrame, label_col: str, value_cols: list):
    try:
        num_cols = [c for c in value_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) < 3:
            return None, None, "# Need at least 3 numeric value columns"

        sample = df[[label_col] + num_cols].dropna().head(6)
        angles = np.linspace(0, 2 * np.pi, len(num_cols), endpoint=False).tolist()
        angles += angles[:1]

        fig_s, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        colors = plt.cm.tab10.colors
        for i, (_, row) in enumerate(sample.iterrows()):
            vals = [row[c] for c in num_cols]
            v_max = max(vals) if max(vals) != 0 else 1
            norm_vals = [v / v_max for v in vals]
            norm_vals += norm_vals[:1]
            ax.plot(angles, norm_vals, lw=2, color=colors[i % len(colors)],
                    label=str(row[label_col]))
            ax.fill(angles, norm_vals, alpha=0.15, color=colors[i % len(colors)])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(num_cols, fontsize=9)
        ax.set_title("Radar Chart", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        fig_s.tight_layout()

        fig_p = go.Figure()
        for _, row in sample.iterrows():
            vals = [row[c] for c in num_cols]
            fig_p.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=num_cols + [num_cols[0]],
                fill="toself",
                name=str(row[label_col]),
            ))
        fig_p.update_layout(polar=dict(radialaxis=dict(visible=True)),
                            title="Radar Chart")
        code = get_code("radar", label_col=label_col, value_cols=num_cols)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Parity Plot ───────────────────────────────────────────────────────

def parity_plot(df: pd.DataFrame, actual_col: str, predicted_col: str):
    try:
        clean = df[[actual_col, predicted_col]].dropna()
        mn = min(clean[actual_col].min(), clean[predicted_col].min())
        mx = max(clean[actual_col].max(), clean[predicted_col].max())

        corr = np.corrcoef(clean[actual_col], clean[predicted_col])[0, 1]

        fig_s, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(clean[actual_col], clean[predicted_col], alpha=0.6, color="steelblue")
        ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect fit")
        ax.set_xlabel(f"Actual ({actual_col})")
        ax.set_ylabel(f"Predicted ({predicted_col})")
        ax.set_title(f"Parity Plot (r = {corr:.3f})")
        ax.legend()
        fig_s.tight_layout()

        fig_p = px.scatter(clean, x=actual_col, y=predicted_col,
                           title=f"Parity Plot (r = {corr:.3f})")
        fig_p.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                        line=dict(color="red", dash="dash"))
        code = get_code("parity_plot", actual_col=actual_col, predicted_col=predicted_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Bland-Altman ──────────────────────────────────────────────────────

def bland_altman_plot(df: pd.DataFrame, method1_col: str, method2_col: str):
    try:
        clean = df[[method1_col, method2_col]].dropna()
        mean = (clean[method1_col] + clean[method2_col]) / 2
        diff = clean[method1_col] - clean[method2_col]
        md = diff.mean()
        sd = diff.std()

        fig_s, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(mean, diff, alpha=0.6, color="steelblue")
        ax.axhline(md, color="blue", lw=2, label=f"Mean: {md:.2f}")
        ax.axhline(md + 1.96 * sd, color="red", linestyle="--",
                   label=f"+1.96 SD: {md + 1.96*sd:.2f}")
        ax.axhline(md - 1.96 * sd, color="red", linestyle="--",
                   label=f"-1.96 SD: {md - 1.96*sd:.2f}")
        ax.set_xlabel("Mean of methods")
        ax.set_ylabel("Difference")
        ax.set_title("Bland-Altman Plot")
        ax.legend()
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=mean, y=diff, mode="markers",
                                   marker=dict(color="steelblue", opacity=0.6),
                                   name="Data"))
        fig_p.add_hline(y=md, line_color="blue", annotation_text=f"Mean: {md:.2f}")
        fig_p.add_hline(y=md + 1.96 * sd, line_dash="dash", line_color="red",
                        annotation_text=f"+1.96 SD: {md + 1.96*sd:.2f}")
        fig_p.add_hline(y=md - 1.96 * sd, line_dash="dash", line_color="red",
                        annotation_text=f"-1.96 SD: {md - 1.96*sd:.2f}")
        fig_p.update_layout(title="Bland-Altman Plot",
                            xaxis_title="Mean of methods",
                            yaxis_title="Difference")
        code = get_code("bland_altman", method1_col=method1_col, method2_col=method2_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Kaplan-Meier ──────────────────────────────────────────────────────

def kaplan_meier_plot(df: pd.DataFrame, duration_col: str, event_col: str,
                      group_col: str = None):
    try:
        if not HAS_LIFELINES:
            return None, None, "# lifelines not installed. pip install lifelines"

        clean = df[[duration_col, event_col] + ([group_col] if group_col else [])].dropna()

        fig_s, ax = plt.subplots(figsize=(9, 6))
        kmf = KaplanMeierFitter()
        plotly_traces = []

        if group_col and group_col in clean.columns:
            groups = clean[group_col].unique()
            colors = plt.cm.tab10.colors
            for i, g in enumerate(groups):
                sub = clean[clean[group_col] == g]
                kmf.fit(sub[duration_col], sub[event_col], label=str(g))
                kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[i % len(colors)])
                t = kmf.survival_function_.index.tolist()
                s = kmf.survival_function_.iloc[:, 0].tolist()
                plotly_traces.append(go.Scatter(x=t, y=s, mode="lines", name=str(g)))
        else:
            kmf.fit(clean[duration_col], clean[event_col], label="All")
            kmf.plot_survival_function(ax=ax, ci_show=True, color="steelblue")
            t = kmf.survival_function_.index.tolist()
            s = kmf.survival_function_.iloc[:, 0].tolist()
            plotly_traces.append(go.Scatter(x=t, y=s, mode="lines", name="All"))

        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.set_title("Kaplan-Meier Survival Curve")
        fig_s.tight_layout()

        fig_p = go.Figure(plotly_traces)
        fig_p.update_layout(title="Kaplan-Meier Survival Curve",
                            xaxis_title="Time",
                            yaxis_title="Survival Probability",
                            yaxis=dict(range=[0, 1.05]))
        code = get_code("kaplan_meier", duration_col=duration_col,
                        event_col=event_col, group_col=group_col)
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── UMAP Plot ─────────────────────────────────────────────────────────

def umap_plot(df: pd.DataFrame, feature_cols: list, color_col: str = None):
    """UMAP dimensionality reduction plot"""
    try:
        try:
            import umap
            HAS_UMAP = True
        except ImportError:
            HAS_UMAP = False

        num_df = df[feature_cols].select_dtypes("number").dropna()
        if num_df.shape[0] < 10 or num_df.shape[1] < 2:
            return None, None, "# Need at least 10 rows and 2 numeric feature columns"

        X = StandardScaler().fit_transform(num_df.fillna(0))

        if HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X)
            x_label, y_label = "UMAP-1", "UMAP-2"
        else:
            # Fallback to PCA if umap not available
            pca = PCA(n_components=2)
            embedding = pca.fit_transform(X)
            ev = pca.explained_variance_ratio_
            x_label = f"PC1 ({ev[0]*100:.1f}%)"
            y_label = f"PC2 ({ev[1]*100:.1f}%)"

        emb_df = pd.DataFrame(embedding, columns=[x_label, y_label], index=num_df.index)
        color_vals = None
        if color_col and color_col in df.columns:
            emb_df["group"] = df.loc[num_df.index, color_col].values
            color_vals = "group"

        fig_s, ax = plt.subplots(figsize=(8, 6))
        if color_vals:
            groups = emb_df["group"].unique()
            colors = plt.cm.tab20.colors
            for i, g in enumerate(groups):
                sub = emb_df[emb_df["group"] == g]
                ax.scatter(sub[x_label], sub[y_label], alpha=0.7,
                           label=str(g), s=30, color=colors[i % len(colors)])
            ax.legend(fontsize=7)
        else:
            ax.scatter(emb_df[x_label], emb_df[y_label], alpha=0.7,
                       color="steelblue", s=30)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("UMAP" if HAS_UMAP else "PCA (UMAP fallback)")
        fig_s.tight_layout()

        fig_p = px.scatter(emb_df, x=x_label, y=y_label, color=color_vals,
                           title="UMAP" if HAS_UMAP else "PCA (UMAP fallback)")
        code = f"""# UMAP Plot
import umap
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df[{feature_cols}].fillna(0))
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X)
import plotly.express as px
fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1])
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── t-SNE Plot ────────────────────────────────────────────────────────

def tsne_plot(df: pd.DataFrame, feature_cols: list, color_col: str = None):
    """t-SNE plot using sklearn"""
    try:
        from sklearn.manifold import TSNE

        num_df = df[feature_cols].select_dtypes("number").dropna()
        if num_df.shape[0] < 10 or num_df.shape[1] < 2:
            return None, None, "# Need at least 10 rows and 2 numeric feature columns"

        sample = num_df.sample(min(1000, len(num_df)), random_state=42)
        X = StandardScaler().fit_transform(sample.fillna(0))
        perplexity = min(30, len(sample) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embedding = tsne.fit_transform(X)

        emb_df = pd.DataFrame(embedding, columns=["tSNE-1", "tSNE-2"], index=sample.index)
        color_vals = None
        if color_col and color_col in df.columns:
            emb_df["group"] = df.loc[sample.index, color_col].values
            color_vals = "group"

        fig_s, ax = plt.subplots(figsize=(8, 6))
        if color_vals:
            groups = emb_df["group"].unique()
            colors = plt.cm.tab20.colors
            for i, g in enumerate(groups):
                sub = emb_df[emb_df["group"] == g]
                ax.scatter(sub["tSNE-1"], sub["tSNE-2"], alpha=0.7,
                           label=str(g), s=30, color=colors[i % len(colors)])
            ax.legend(fontsize=7)
        else:
            ax.scatter(emb_df["tSNE-1"], emb_df["tSNE-2"], alpha=0.7,
                       color="steelblue", s=30)
        ax.set_xlabel("tSNE-1")
        ax.set_ylabel("tSNE-2")
        ax.set_title("t-SNE Plot")
        fig_s.tight_layout()

        fig_p = px.scatter(emb_df, x="tSNE-1", y="tSNE-2", color=color_vals,
                           title="t-SNE Plot")
        code = f"""# t-SNE Plot
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
X = StandardScaler().fit_transform(df[{feature_cols}].fillna(0))
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedding = tsne.fit_transform(X)
import plotly.express as px
fig = px.scatter(x=embedding[:, 0], y=embedding[:, 1])
fig.show()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Manhattan Plot ────────────────────────────────────────────────────

def manhattan_plot(df: pd.DataFrame, chr_col: str, pos_col: str, pval_col: str):
    """Manhattan plot for GWAS data"""
    try:
        df2 = df[[chr_col, pos_col, pval_col]].dropna().copy()
        df2["neg_log10p"] = -np.log10(df2[pval_col].clip(lower=1e-300))
        df2 = df2.sort_values([chr_col, pos_col])

        # Assign cumulative x position
        chrs = df2[chr_col].astype(str).unique()
        colors = ["#4575b4", "#d73027"] * (len(chrs) // 2 + 1)
        chr_colors = {c: colors[i % len(colors)] for i, c in enumerate(chrs)}

        x_pos = []
        x_offset = 0
        chr_midpoints = {}
        for c in chrs:
            sub = df2[df2[chr_col].astype(str) == c]
            positions = sub[pos_col].values
            x_pos.extend(positions - positions.min() + x_offset)
            chr_midpoints[c] = x_offset + (positions.max() - positions.min()) / 2
            x_offset += (positions.max() - positions.min()) + 1e6

        df2["x_pos"] = x_pos
        df2["color"] = df2[chr_col].astype(str).map(chr_colors)

        fig_s, ax = plt.subplots(figsize=(14, 5))
        for c in chrs:
            sub = df2[df2[chr_col].astype(str) == c]
            ax.scatter(sub["x_pos"], sub["neg_log10p"],
                       c=chr_colors[c], s=5, alpha=0.7)

        # Significance threshold
        sig_thresh = -np.log10(5e-8)
        ax.axhline(sig_thresh, color="red", linestyle="--", lw=1,
                   label="p=5×10⁻⁸")
        ax.set_xticks([chr_midpoints[c] for c in chrs])
        ax.set_xticklabels([str(c) for c in chrs], fontsize=7, rotation=45)
        ax.set_ylabel("-log₁₀(p-value)")
        ax.set_xlabel("Chromosome")
        ax.set_title("Manhattan Plot")
        ax.legend()
        fig_s.tight_layout()

        fig_p = px.scatter(df2, x="x_pos", y="neg_log10p", color=chr_col,
                           title="Manhattan Plot",
                           labels={"neg_log10p": "-log₁₀(p-value)", "x_pos": "Position"})
        fig_p.add_hline(y=sig_thresh, line_dash="dash", line_color="red",
                        annotation_text="p=5×10⁻⁸")

        code = f"""# Manhattan Plot
df['-log10p'] = -np.log10(df['{pval_col}'])
# Sort by chromosome and position
fig, ax = plt.subplots(figsize=(14, 5))
# Plot each chromosome with alternating colors
ax.axhline(-np.log10(5e-8), color='red', linestyle='--')
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Forest Plot ───────────────────────────────────────────────────────

def forest_plot(df: pd.DataFrame, study_col: str, effect_col: str,
                ci_low_col: str, ci_high_col: str):
    """Forest plot for meta-analysis"""
    try:
        sample = df[[study_col, effect_col, ci_low_col, ci_high_col]].dropna().head(20)
        y_pos = np.arange(len(sample))

        fig_s, ax = plt.subplots(figsize=(10, max(5, len(sample) * 0.6)))
        for i, (_, row) in enumerate(sample.iterrows()):
            ax.plot([row[ci_low_col], row[ci_high_col]], [i, i],
                    color="steelblue", lw=2, solid_capstyle="round")
            ax.plot(row[effect_col], i, "s", color="steelblue",
                    markersize=8, zorder=5)

        ax.axvline(0, color="black", lw=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sample[study_col].astype(str), fontsize=9)
        ax.set_xlabel("Effect Size")
        ax.set_title("Forest Plot")
        ax.grid(axis="x", alpha=0.3)
        fig_s.tight_layout()

        # Overall estimate
        pooled = sample[effect_col].mean()
        ci_l = sample[ci_low_col].mean()
        ci_h = sample[ci_high_col].mean()

        fig_p = go.Figure()
        for _, row in sample.iterrows():
            fig_p.add_trace(go.Scatter(
                x=[row[ci_low_col], row[effect_col], row[ci_high_col]],
                y=[str(row[study_col])] * 3,
                mode="lines+markers",
                marker=dict(size=[4, 10, 4], color="steelblue"),
                showlegend=False,
            ))
        fig_p.add_vline(x=0, line_dash="dash", line_color="black")
        fig_p.update_layout(title="Forest Plot", xaxis_title="Effect Size")

        code = f"""# Forest Plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, len(df) * 0.6))
for i, (_, row) in enumerate(df.iterrows()):
    ax.plot([row['{ci_low_col}'], row['{ci_high_col}']], [i, i], color='steelblue', lw=2)
    ax.plot(row['{effect_col}'], i, 's', color='steelblue', markersize=8)
ax.axvline(0, linestyle='--', color='black')
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['{study_col}'])
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Funnel Plot ───────────────────────────────────────────────────────

def funnel_plot(df: pd.DataFrame, effect_col: str, se_col: str,
                study_col: str = None):
    """Funnel plot for publication bias detection"""
    try:
        clean = df[[effect_col, se_col] +
                   ([study_col] if study_col and study_col in df.columns else [])].dropna()

        pooled = clean[effect_col].mean()
        precision = 1 / (clean[se_col] + 1e-9)

        # Funnel lines
        se_max = clean[se_col].max()
        se_range = np.linspace(0, se_max * 1.1, 100)
        upper = pooled + 1.96 * se_range
        lower = pooled - 1.96 * se_range

        fig_s, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(clean[effect_col], clean[se_col], color="steelblue",
                   alpha=0.7, s=40)
        ax.plot(upper, se_range, "r--", lw=1, label="95% CI")
        ax.plot(lower, se_range, "r--", lw=1)
        ax.axvline(pooled, color="gray", lw=1, linestyle="-")
        ax.invert_yaxis()
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Standard Error")
        ax.set_title("Funnel Plot")
        ax.legend()
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=clean[effect_col], y=clean[se_col],
            mode="markers",
            marker=dict(color="steelblue", size=8),
            text=clean[study_col].astype(str) if study_col and study_col in df.columns else None,
            name="Studies",
        ))
        fig_p.add_trace(go.Scatter(x=upper, y=se_range, mode="lines",
                                   line=dict(dash="dash", color="red"), name="95% CI"))
        fig_p.add_trace(go.Scatter(x=lower, y=se_range, mode="lines",
                                   line=dict(dash="dash", color="red"), showlegend=False))
        fig_p.update_yaxes(autorange="reversed")
        fig_p.update_layout(title="Funnel Plot",
                            xaxis_title="Effect size",
                            yaxis_title="Standard Error")

        code = f"""# Funnel Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df['{effect_col}'], df['{se_col}'], color='steelblue', alpha=0.7)
pooled = df['{effect_col}'].mean()
ax.axvline(pooled, color='gray', linestyle='-')
ax.invert_yaxis()
ax.set_xlabel('Effect size')
ax.set_ylabel('Standard Error')
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Calibration Curve ─────────────────────────────────────────────────

def calibration_curve(df: pd.DataFrame, y_true_col: str, y_prob_col: str):
    """Calibration/reliability curve for classifiers"""
    try:
        from sklearn.calibration import calibration_curve as sk_cal_curve

        clean = df[[y_true_col, y_prob_col]].dropna()
        y_true = clean[y_true_col].values
        y_prob = clean[y_prob_col].values

        n_bins = min(10, len(clean) // 10)
        if n_bins < 2:
            n_bins = 2
        fraction_pos, mean_predicted = sk_cal_curve(y_true, y_prob, n_bins=n_bins)

        fig_s, ax = plt.subplots(figsize=(7, 7))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(mean_predicted, fraction_pos, "o-", color="steelblue",
                lw=2, markersize=8, label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(dash="dash", color="black"),
                                   name="Perfect"))
        fig_p.add_trace(go.Scatter(x=mean_predicted, y=fraction_pos,
                                   mode="lines+markers",
                                   line=dict(color="steelblue"),
                                   marker=dict(size=8), name="Model"))
        fig_p.update_layout(title="Calibration Curve",
                            xaxis_title="Mean predicted probability",
                            yaxis_title="Fraction of positives")

        code = f"""# Calibration Curve
from sklearn.calibration import calibration_curve
fraction_pos, mean_pred = calibration_curve(df['{y_true_col}'], df['{y_prob_col}'],
                                            n_bins=10)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], 'k--', label='Perfect')
ax.plot(mean_pred, fraction_pos, 'o-', label='Model')
ax.legend()
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Residual Plot ─────────────────────────────────────────────────────

def residual_plot(df: pd.DataFrame, fitted_col: str, residual_col: str):
    """Residual plot for regression diagnostics"""
    try:
        clean = df[[fitted_col, residual_col]].dropna()

        fig_s, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs Fitted
        axes[0].scatter(clean[fitted_col], clean[residual_col],
                        alpha=0.5, color="steelblue", s=20)
        axes[0].axhline(0, color="red", linestyle="--", lw=1)
        axes[0].set_xlabel(f"Fitted ({fitted_col})")
        axes[0].set_ylabel(f"Residuals ({residual_col})")
        axes[0].set_title("Residuals vs Fitted")

        # Q-Q of residuals
        from scipy import stats as scipy_stats
        (osm, osr), (slope, intercept, r) = scipy_stats.probplot(
            clean[residual_col], dist="norm")
        axes[1].plot(osm, osr, "o", alpha=0.5, color="steelblue", ms=4)
        axes[1].plot(osm, slope * np.array(osm) + intercept, "r-", lw=2)
        axes[1].set_xlabel("Theoretical Quantiles")
        axes[1].set_ylabel("Sample Quantiles")
        axes[1].set_title("Q-Q Plot of Residuals")

        fig_s.tight_layout()

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=clean[fitted_col], y=clean[residual_col],
                                   mode="markers",
                                   marker=dict(color="steelblue", size=5, opacity=0.6),
                                   name="Residuals"))
        fig_p.add_hline(y=0, line_dash="dash", line_color="red")
        fig_p.update_layout(title="Residual Plot",
                            xaxis_title=f"Fitted ({fitted_col})",
                            yaxis_title=f"Residuals ({residual_col})")

        code = f"""# Residual Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df['{fitted_col}'], df['{residual_col}'], alpha=0.5, color='steelblue')
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('{fitted_col}')
axes[0].set_ylabel('{residual_col}')
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── UpSet Plot ────────────────────────────────────────────────────────

def upset_plot(df: pd.DataFrame, set_cols: list):
    """UpSet plot as alternative to Venn for many sets (manual implementation)"""
    try:
        valid_cols = [c for c in set_cols if c in df.columns]
        if len(valid_cols) < 2:
            return None, None, "# Need at least 2 boolean/binary set columns"

        # Binarize
        bin_df = df[valid_cols].fillna(0).astype(bool)
        n_sets = len(valid_cols)

        # Generate all combinations
        from itertools import combinations
        combo_sizes = {}
        # Single sets
        for c in valid_cols:
            mask = bin_df[c] & ~bin_df[[x for x in valid_cols if x != c]].any(axis=1)
            combo_sizes[tuple([c])] = mask.sum()
        # Intersections
        for r in range(2, n_sets + 1):
            for combo in combinations(valid_cols, r):
                mask = bin_df[list(combo)].all(axis=1)
                combo_sizes[tuple(sorted(combo))] = mask.sum()

        # Sort by size
        sorted_combos = sorted(combo_sizes.items(), key=lambda x: -x[1])[:15]
        combo_names = ["+".join(c) if len(c) <= 2 else f"{len(c)}-way" for c, _ in sorted_combos]
        combo_vals = [v for _, v in sorted_combos]

        fig_s, axes = plt.subplots(2, 1, figsize=(12, 8),
                                   gridspec_kw={"height_ratios": [2, 1]})

        # Bar chart on top
        axes[0].bar(range(len(combo_vals)), combo_vals, color="steelblue")
        axes[0].set_xticks(range(len(combo_names)))
        axes[0].set_xticklabels(combo_names, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Intersection size")
        axes[0].set_title("UpSet Plot")

        # Dot matrix on bottom
        for i, (combo, _) in enumerate(sorted_combos):
            for j, col in enumerate(valid_cols):
                if col in combo:
                    axes[1].plot(i, j, "o", color="steelblue", markersize=12)
                else:
                    axes[1].plot(i, j, "o", color="lightgray", markersize=12)

        axes[1].set_yticks(range(n_sets))
        axes[1].set_yticklabels(valid_cols, fontsize=8)
        axes[1].set_xticks([])
        axes[1].set_xlim(-0.5, len(sorted_combos) - 0.5)
        fig_s.tight_layout()

        # Plotly bar chart
        fig_p = go.Figure(go.Bar(
            x=combo_names, y=combo_vals, marker_color="steelblue"
        ))
        fig_p.update_layout(title="UpSet Plot — Intersection Sizes",
                            xaxis_title="Intersection",
                            yaxis_title="Count")

        code = f"""# UpSet Plot
# Shows intersections between binary/boolean set columns
# Columns: {valid_cols}
from itertools import combinations
bin_df = df[{valid_cols}].astype(bool)
# Compute intersection sizes for all combinations
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Radial Bar + Significance (NGplot-inspired 环形柱状图+误差线) ─────
# Based on NGplot's popular "环形柱状图-多变量-误差线-星号连接" templates.

try:
    from scipy.stats import f_oneway as _fow
    _HAS_SCIPY_SCI = True
except ImportError:
    _HAS_SCIPY_SCI = False


def radial_bar_sig(df: pd.DataFrame, group_col: str, value_col: str):
    """
    Radial Bar + Significance Chart (NGplot 环形柱状图 style):
    Polar bar chart with error bars. If scipy is available, runs a
    one-way ANOVA and annotates the significance level on the title.
    """
    try:
        agg = df.groupby(group_col)[value_col].agg(["mean", "std", "count"]).reset_index()
        agg.columns = [group_col, "mean", "std", "count"]
        agg["sem"] = agg["std"] / np.sqrt(agg["count"])
        n = len(agg)

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        bar_width = (2 * np.pi) / n * 0.7

        cmap = plt.cm.get_cmap("tab20", n)
        colors = [cmap(i) for i in range(n)]

        fig_s, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)

        max_val = (agg["mean"] + agg["sem"]).max() * 1.2
        ax.set_ylim(0, max_val)

        for i, (angle, row) in enumerate(zip(angles, agg.itertuples())):
            mean_val = row.mean
            sem_val = row.sem
            ax.bar(angle, mean_val, width=bar_width,
                   color=colors[i], alpha=0.85, label=str(getattr(row, group_col)))
            # Error bar
            ax.plot([angle, angle], [mean_val - sem_val, mean_val + sem_val],
                    color="black", lw=1.5, zorder=5)
            ax.plot([angle - bar_width * 0.15, angle + bar_width * 0.15],
                    [mean_val + sem_val, mean_val + sem_val],
                    color="black", lw=1.5, zorder=5)

        # Significance annotation (one-way ANOVA)
        sig_note = ""
        if _HAS_SCIPY_SCI and n > 1:
            try:
                groups_data = [df[df[group_col] == g][value_col].dropna().values
                                for g in agg[group_col]]
                groups_data = [g for g in groups_data if len(g) >= 2]
                if len(groups_data) > 1:
                    _, p = _fow(*groups_data)
                    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                    sig_note = f"  ANOVA: p={p:.2e} {stars}"
            except Exception:
                pass

        ax.set_xticks(angles)
        ax.set_xticklabels([str(g) for g in agg[group_col]], fontsize=9)
        ax.yaxis.set_visible(False)
        ax.set_title(f"Radial Bar + Significance\n{value_col} by {group_col}{sig_note}",
                     pad=20, fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
        fig_s.tight_layout()

        code = f"""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

agg = df.groupby('{group_col}')['{value_col}'].agg(['mean','std','count']).reset_index()
agg.columns = ['{group_col}','mean','std','count']
agg['sem'] = agg['std'] / np.sqrt(agg['count'])
n = len(agg)

angles = np.linspace(0, 2*np.pi, n, endpoint=False)
bar_width = (2*np.pi)/n * 0.7
cmap = plt.cm.get_cmap('tab20', n)

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={{'projection': 'polar'}})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_ylim(0, (agg['mean'] + agg['sem']).max() * 1.2)

for i, (angle, row) in enumerate(zip(angles, agg.itertuples())):
    ax.bar(angle, row.mean, width=bar_width, color=cmap(i), alpha=0.85)
    ax.plot([angle, angle], [row.mean - row.sem, row.mean + row.sem], 'k-', lw=1.5)

ax.set_xticks(angles)
ax.set_xticklabels(agg['{group_col}'].astype(str))
ax.set_title('Radial Bar + Significance', pad=20)
plt.tight_layout()
plt.savefig('radial_bar_sig.png', dpi=300)
plt.show()
"""
        return fig_s, None, code

    except Exception as e:
        return None, None, f"# Error: {e}"
