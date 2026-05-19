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
