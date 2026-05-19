"""
SciVizKit — Scientific Visualization Toolkit
Main Streamlit application
"""

import io
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data_analyzer import DataAnalyzer
from src.chart_registry import CHART_REGISTRY

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="SciVizKit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-title { font-size: 2.5rem; font-weight: 800; color: #667eea; }
.subtitle { font-size: 1.1rem; color: #666; margin-bottom: 1rem; }
.metric-card {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border-radius: 12px; padding: 16px; margin: 4px;
    border: 1px solid #667eea44;
}
.cat-header {
    font-size: 1.3rem; font-weight: 700;
    border-bottom: 3px solid #667eea;
    padding-bottom: 6px; margin: 24px 0 12px 0;
}
.chart-badge {
    display: inline-block;
    background: #667eea; color: white;
    border-radius: 12px; padding: 2px 10px;
    font-size: 0.75rem; margin: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Lazy generator imports ────────────────────────────────────────────
@st.cache_resource
def _load_generators():
    from src.generators import distribution, comparison, correlation
    from src.generators import timeseries, proportional, network, scientific
    return {
        "distribution": distribution,
        "comparison": comparison,
        "correlation": correlation,
        "timeseries": timeseries,
        "proportional": proportional,
        "network": network,
        "scientific": scientific,
    }


# ── Session state init ────────────────────────────────────────────────
def init_state():
    defaults = {
        "df": None,
        "analyzer": None,
        "charts_generated": False,
        "chart_results": {},
        "filename": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ── Helper: load example CSVs ─────────────────────────────────────────
def load_example(name: str):
    path = os.path.join(os.path.dirname(__file__), "examples", f"{name}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        st.session_state.df = df
        st.session_state.analyzer = DataAnalyzer(df)
        st.session_state.charts_generated = False
        st.session_state.chart_results = {}
        st.session_state.filename = f"{name}.csv"
        st.rerun()
    else:
        st.error(f"Example file not found: {path}")


# ── Helper: save fig to bytes ─────────────────────────────────────────
def fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ── Auto-select columns for a chart ──────────────────────────────────
def pick_cols(chart_id: str, analyzer: DataAnalyzer) -> dict:
    """Heuristically pick appropriate columns for each chart type."""
    num = analyzer.numeric_cols
    cat = analyzer.categorical_cols
    dt = analyzer.datetime_cols
    all_cols = list(analyzer.df.columns)

    x, y, c, s = (num + [None])[0], (num + [None, None])[1], (cat + [None])[0], (num + [None, None, None])[2]
    label = (cat + [None])[0]
    val1 = (num + [None])[0]
    val2 = (num + [None, None])[1]
    dt_col = (dt + [None])[0]

    mapping = {
        "histogram":     dict(x_col=x, color_col=c),
        "kde":           dict(x_col=x, color_col=c),
        "violin":        dict(x_col=c or (cat + all_cols + [None])[0], y_col=x),
        "boxplot":       dict(x_col=c or (cat + all_cols + [None])[0], y_col=x),
        "stripplot":     dict(x_col=c or (cat + all_cols + [None])[0], y_col=x),
        "beeswarm":      dict(x_col=c or (cat + all_cols + [None])[0], y_col=x),
        "ecdf":          dict(x_col=x),
        "qqplot":        dict(x_col=x),
        "bar":           dict(x_col=c or all_cols[0], y_col=x or all_cols[1] if len(all_cols)>1 else all_cols[0]),
        "grouped_bar":   dict(x_col=c or all_cols[0], y_col=x, color_col=(cat + [None, None])[1] or c),
        "stacked_bar":   dict(x_col=c or all_cols[0], y_col=x, color_col=(cat + [None, None])[1] or c),
        "lollipop":      dict(x_col=c or all_cols[0], y_col=x),
        "dumbbell":      dict(label_col=c or all_cols[0], val1_col=val1, val2_col=val2),
        "dotplot":       dict(x_col=c or all_cols[0], y_col=x),
        "slope":         dict(label_col=c or all_cols[0], val1_col=val1, val2_col=val2),
        "waterfall":     dict(label_col=c or all_cols[0], value_col=x),
        "errorbar":      dict(x_col=c or all_cols[0], y_col=x, err_col=val2 or x),
        "scatter":       dict(x_col=x, y_col=y, color_col=c),
        "bubble":        dict(x_col=x, y_col=y, size_col=s or val1, color_col=c),
        "hexbin":        dict(x_col=x, y_col=y),
        "corr_heatmap":  dict(),
        "pairplot":      dict(cols=num[:5] if num else []),
        "parallel_coords": dict(cols=num[:6] if num else [], color_col=c),
        "line":          dict(x_col=dt_col or x, y_cols=[y] if y else [x]),
        "area":          dict(x_col=dt_col or x, y_col=y or x),
        "stacked_area":  dict(x_col=dt_col or x, y_cols=num[:3] if len(num)>=2 else [x, y] if y else [x]),
        "step_line":     dict(x_col=dt_col or x, y_col=y or x),
        "pie":           dict(label_col=c or all_cols[0], value_col=x),
        "donut":         dict(label_col=c or all_cols[0], value_col=x),
        "treemap":       dict(label_col=c or all_cols[0], value_col=x),
        "sunburst":      dict(label_col=c or all_cols[0],
                              parent_col=(cat + [None, None])[1] or c,
                              value_col=x),
        "nightingale":   dict(label_col=c or all_cols[0], value_col=x),
        "sankey":        dict(source_col=c or all_cols[0],
                              target_col=(cat + [None, None])[1] or c,
                              value_col=x),
        "network_graph": dict(source_col=c or all_cols[0],
                              target_col=(cat + [None, None])[1] or all_cols[1] if len(all_cols)>1 else c),
        "dendrogram":    dict(cols=num[:8] if num else []),
        "volcano":       dict(fc_col=x, pval_col=y or x),
        "pca_plot":      dict(feature_cols=num[:10] if num else [], color_col=c),
        "roc_curve":     dict(y_true_col=x, y_score_col=y or x),
        "radar":         dict(label_col=c or all_cols[0], value_cols=num[:6] if num else [x]),
        "parity_plot":   dict(actual_col=x, predicted_col=y or x),
        "bland_altman":  dict(method1_col=x, method2_col=y or x),
        "kaplan_meier":  dict(duration_col=x, event_col=y or x, group_col=c),
    }
    return mapping.get(chart_id, {})


# ── Generate all charts ───────────────────────────────────────────────
def generate_charts(df: pd.DataFrame, analyzer: DataAnalyzer,
                    domain: str, chart_ids: list, gen_modules: dict) -> dict:
    results = {}
    progress = st.progress(0, text="Generating charts…")
    total = len(chart_ids)

    # Chart ID → generator function mapping
    fn_map = {
        "histogram":     ("distribution", "histogram"),
        "kde":           ("distribution", "kde_plot"),
        "violin":        ("distribution", "violin_plot"),
        "boxplot":       ("distribution", "box_plot"),
        "stripplot":     ("distribution", "strip_plot"),
        "beeswarm":      ("distribution", "beeswarm_plot"),
        "ecdf":          ("distribution", "ecdf_plot"),
        "qqplot":        ("distribution", "qq_plot"),
        "bar":           ("comparison", "bar_chart"),
        "grouped_bar":   ("comparison", "grouped_bar"),
        "stacked_bar":   ("comparison", "stacked_bar"),
        "lollipop":      ("comparison", "lollipop_chart"),
        "dumbbell":      ("comparison", "dumbbell_chart"),
        "dotplot":       ("comparison", "dot_plot"),
        "slope":         ("comparison", "slope_chart"),
        "waterfall":     ("comparison", "waterfall_chart"),
        "errorbar":      ("comparison", "error_bar_plot"),
        "scatter":       ("correlation", "scatter_plot"),
        "bubble":        ("correlation", "bubble_chart"),
        "hexbin":        ("correlation", "hexbin_plot"),
        "corr_heatmap":  ("correlation", "corr_heatmap"),
        "pairplot":      ("correlation", "pairplot"),
        "parallel_coords": ("correlation", "parallel_coords"),
        "line":          ("timeseries", "line_chart"),
        "area":          ("timeseries", "area_chart"),
        "stacked_area":  ("timeseries", "stacked_area"),
        "step_line":     ("timeseries", "step_line"),
        "pie":           ("proportional", "pie_chart"),
        "donut":         ("proportional", "donut_chart"),
        "treemap":       ("proportional", "treemap"),
        "sunburst":      ("proportional", "sunburst"),
        "nightingale":   ("proportional", "nightingale_rose"),
        "sankey":        ("network", "sankey_diagram"),
        "network_graph": ("network", "network_graph"),
        "dendrogram":    ("network", "dendrogram_plot"),
        "volcano":       ("scientific", "volcano_plot"),
        "pca_plot":      ("scientific", "pca_plot"),
        "roc_curve":     ("scientific", "roc_curve_plot"),
        "radar":         ("scientific", "radar_chart"),
        "parity_plot":   ("scientific", "parity_plot"),
        "bland_altman":  ("scientific", "bland_altman_plot"),
        "kaplan_meier":  ("scientific", "kaplan_meier_plot"),
    }

    for i, chart_id in enumerate(chart_ids):
        progress.progress((i + 1) / total, text=f"Generating: {CHART_REGISTRY[chart_id]['name']}")
        try:
            mod_key, fn_name = fn_map[chart_id]
            mod = gen_modules[mod_key]
            fn = getattr(mod, fn_name)
            kwargs = pick_cols(chart_id, analyzer)
            # Filter None values except where required
            kwargs_clean = {k: v for k, v in kwargs.items() if v is not None}
            result = fn(df, **kwargs_clean)
            results[chart_id] = result
        except Exception as ex:
            results[chart_id] = (None, None, f"# Error generating {chart_id}: {ex}")

    progress.empty()
    return results


# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/SciVizKit-v0.1.0-667eea?style=for-the-badge", use_column_width=True)
    st.markdown("## 📁 Data Input")

    uploaded = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Max 200 MB"
    )

    if uploaded:
        try:
            if uploaded.name.endswith((".xlsx", ".xls")):
                df_new = pd.read_excel(uploaded)
            else:
                df_new = pd.read_csv(uploaded)
            if not df_new.equals(st.session_state.df if st.session_state.df is not None else pd.DataFrame()):
                st.session_state.df = df_new
                st.session_state.analyzer = DataAnalyzer(df_new)
                st.session_state.charts_generated = False
                st.session_state.chart_results = {}
                st.session_state.filename = uploaded.name
        except Exception as e:
            st.error(f"Failed to load file: {e}")

    st.markdown("---")
    st.markdown("## 🔭 Domain")
    domain = st.radio(
        "Select research domain:",
        ["General", "Biology", "Chemistry/Materials", "Medicine", "Physics", "Social Science"],
        index=0,
    )

    st.markdown("---")
    st.markdown("## 🎨 Chart Mode")
    chart_mode = st.radio(
        "Display mode:",
        ["Interactive", "Static", "Both"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📦 Examples")
    col1, col2, col3 = st.columns(3)
    if col1.button("General", use_container_width=True):
        load_example("sample_general")
    if col2.button("Biology", use_container_width=True):
        load_example("sample_biology")
    if col3.button("Chemistry", use_container_width=True):
        load_example("sample_chemistry")


# ── Main content ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">🔬 SciVizKit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Scientific Visualization Toolkit · '
    '激发科研数据最佳可视化方案</div>',
    unsafe_allow_html=True
)

# ── Landing page ──────────────────────────────────────────────────────
if st.session_state.df is None:
    st.markdown("---")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
### ✨ What SciVizKit Can Do

- 📊 **50+ chart types** across 7 categories
- 🔍 **Auto-detects** column types (numeric, categorical, datetime, binary)
- 🎯 **Domain-aware** chart recommendations (Biology, Chemistry, Medicine, Physics…)
- 🖼️ **Static** (matplotlib/seaborn) + **Interactive** (Plotly) charts
- 💻 **Copy-paste code** for every chart — export to your own notebook
- ⬇️ **Download PNG** for any static chart

### 🗂️ Chart Categories

| Category | Charts |
|----------|--------|
| Distribution | Histogram, KDE, Violin, Box, Strip, Beeswarm, ECDF, Q-Q |
| Comparison | Bar, Grouped Bar, Stacked Bar, Lollipop, Dumbbell, Dot, Slope, Waterfall, Error Bar |
| Correlation | Scatter, Bubble, Hexbin, Corr Heatmap, Pair Plot, Parallel Coords |
| Time Series | Line, Area, Stacked Area, Step Line |
| Proportional | Pie, Donut, Treemap, Sunburst, Nightingale Rose |
| Network | Sankey, Network Graph, Dendrogram |
| Scientific | Volcano, PCA, ROC Curve, Radar, Parity, Bland-Altman, Kaplan-Meier |
""")
    with col_r:
        st.info("👈 **Upload your CSV/Excel** in the sidebar, or load an example dataset to get started.")
        st.markdown("#### 🚀 Quick Start")
        st.code("""
# 1. Upload your data (sidebar)
# 2. Select domain
# 3. Click 'Generate All Visualizations'
# 4. Explore 50+ charts!
        """, language="bash")

    st.stop()


# ── Data loaded — show overview ───────────────────────────────────────
df = st.session_state.df
analyzer = st.session_state.analyzer

st.success(f"✅ Loaded **{st.session_state.filename}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

# Data preview
with st.expander("📋 Data Preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)

# Column type summary
with st.expander("🔍 Column Type Analysis", expanded=True):
    p = analyzer.profile
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔢 Numeric", len(p["numeric_cols"]))
    c2.metric("🏷️ Categorical", len(p["categorical_cols"]))
    c3.metric("📅 Datetime", len(p["datetime_cols"]))
    c4.metric("⚡ Binary", len(p["binary_cols"]))

    if p["numeric_cols"]:
        st.markdown(f"**Numeric:** `{'` · `'.join(p['numeric_cols'])}`")
    if p["categorical_cols"]:
        st.markdown(f"**Categorical:** `{'` · `'.join(p['categorical_cols'])}`")
    if p["datetime_cols"]:
        st.markdown(f"**Datetime:** `{'` · `'.join(p['datetime_cols'])}`")

# Compatible charts
compat = analyzer.get_compatible_charts(domain)
st.markdown(f"**{len(compat)} compatible charts** found for domain: **{domain}**")

# ── Generate button ───────────────────────────────────────────────────
if st.button("🚀 Generate All Visualizations", type="primary", use_container_width=True):
    gen_modules = _load_generators()
    with st.spinner("Building your visualizations…"):
        results = generate_charts(df, analyzer, domain, compat, gen_modules)
    st.session_state.chart_results = results
    st.session_state.charts_generated = True
    st.rerun()


# ── Show charts grouped by category ──────────────────────────────────
if st.session_state.charts_generated and st.session_state.chart_results:
    results = st.session_state.chart_results

    # Group by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for cid, result in results.items():
        meta = CHART_REGISTRY.get(cid, {})
        cat = meta.get("category", "Other")
        by_cat[cat].append((cid, meta, result))

    cat_order = ["Distribution", "Comparison", "Correlation",
                 "Time Series", "Proportional", "Network", "Scientific"]
    cats_present = [c for c in cat_order if c in by_cat] + \
                   [c for c in by_cat if c not in cat_order]

    for cat in cats_present:
        items = by_cat[cat]
        st.markdown(f'<div class="cat-header">📊 {cat}</div>', unsafe_allow_html=True)

        # 3-column grid
        cols = st.columns(3)
        for idx, (cid, meta, (fig_s, fig_p, code)) in enumerate(items):
            col = cols[idx % 3]
            with col:
                with st.expander(f"**{meta.get('name', cid)}**", expanded=False):
                    has_content = fig_s is not None or fig_p is not None

                    if not has_content:
                        st.warning(f"⚠️ Chart could not be generated.\n\n`{code[:200]}`")
                    else:
                        # Show chart based on mode
                        if chart_mode in ("Interactive", "Both") and fig_p is not None:
                            st.plotly_chart(fig_p, use_container_width=True, key=f"plotly_{cid}")
                        if chart_mode in ("Static", "Both") and fig_s is not None:
                            st.pyplot(fig_s, use_container_width=True)
                        elif chart_mode == "Interactive" and fig_p is None and fig_s is not None:
                            # fallback to static if no interactive
                            st.pyplot(fig_s, use_container_width=True)

                        # When to use
                        st.info(f"💡 **When to use:** {meta.get('when_to_use', '')}")

                        # Download PNG
                        if fig_s is not None:
                            try:
                                png_bytes = fig_to_png(fig_s)
                                st.download_button(
                                    label="⬇️ Download PNG",
                                    data=png_bytes,
                                    file_name=f"{cid}.png",
                                    mime="image/png",
                                    key=f"dl_{cid}",
                                    use_container_width=True,
                                )
                            except Exception:
                                pass

                        # Code snippet
                        if code and not code.startswith("# Error"):
                            with st.expander("📋 Copy Code", expanded=False):
                                st.code(code, language="python")

    st.markdown("---")
    st.markdown(
        f"✅ Generated **{len([r for r in results.values() if r[0] is not None or r[1] is not None])}** "
        f"charts out of **{len(results)}** compatible types."
    )
