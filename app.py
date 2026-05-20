"""
SciVizKit — Scientific Visualization Toolkit
Inspire the best visualization for your research data.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, traceback, sys
sys.path.insert(0, '.')

st.set_page_config(
    page_title="SciVizKit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Chart card hover effect */
div[data-testid="stImage"] img {
    border-radius: 8px;
    transition: transform 0.2s;
    cursor: pointer;
}
div[data-testid="stImage"] img:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}
/* Category badge */
.cat-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 4px;
}
/* Favorite star button */
.stButton button[title="favorite"] {
    background: none;
    border: none;
    font-size: 20px;
    padding: 0;
}
/* Chart name */
.chart-name {
    font-weight: 600;
    font-size: 13px;
    color: #262730;
    margin: 4px 0 2px 0;
}
/* Progress section */
.gen-progress {
    padding: 12px;
    background: #f0f2f6;
    border-radius: 8px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Category Colors ──────────────────────────────────────────────────────────
CATEGORY_COLORS = {
    "Distribution": "#4DBBD5",
    "Comparison": "#E64B35",
    "Correlation": "#00A087",
    "Time Series": "#3C5488",
    "Proportional": "#F39B7F",
    "Network": "#8491B4",
    "Scientific": "#91D1C2",
    "Text": "#DC0000",
    "Geographic": "#7E6148",
    "3D": "#B09C85",
}

# ── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        'generated_charts': {},     # {chart_id: {name, category, fig_static, fig_plotly, code_str, error}}
        'favorites': set(),          # set of chart_ids
        'search_query': '',
        'active_category': 'All',
        'dt_path': [],
        'df': None,
        'profile': None,
        'show_chart_modal': None,    # chart_id to show in detail view
        'generation_progress': {},   # {category: status}  'pending'|'done'|'error'
        'col_mapping': {},           # {role: col_name}  e.g. {'x': 'age', 'y': 'income'}
        'chart_mode': 'Both',
        'palette': 'Default',
        'domain': 'General',
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Imports ──────────────────────────────────────────────────────────────────
from src.data_analyzer import DataAnalyzer
from src.chart_registry import CHART_REGISTRY
from src.themes.palettes import PALETTES
import src.generators.distribution as dist_gen
import src.generators.comparison as comp_gen
import src.generators.correlation as corr_gen
import src.generators.timeseries as ts_gen
import src.generators.proportional as prop_gen
import src.generators.network as net_gen
import src.generators.scientific as sci_gen

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=60)
    st.title("SciVizKit")
    st.caption("Scientific Visualization Toolkit")
    st.divider()

    uploaded_file = st.file_uploader(
        "📂 Upload Data",
        type=["csv", "xlsx", "xls"],
        help="CSV or Excel file up to 200MB"
    )

    st.divider()

    domain = st.selectbox("🔬 Research Domain", [
        "General", "Biology & Genomics", "Chemistry & Materials",
        "Medicine & Clinical", "Physics & Engineering", "Social Science"
    ])
    st.session_state.domain = domain

    chart_mode = st.radio(
        "📊 Chart Mode",
        ["Static (Publication)", "Interactive", "Both"],
        index=2,
        help="Static = matplotlib 300 DPI. Interactive = plotly."
    )
    st.session_state.chart_mode = chart_mode

    palette = st.selectbox(
        "🎨 Color Palette",
        list(PALETTES.keys()),
        help="Journal-standard color palettes"
    )
    st.session_state.palette = palette

    # Show palette preview swatches
    colors = PALETTES[palette]["categorical"][:6]
    swatch_html = "".join([
        f'<span style="display:inline-block;width:18px;height:18px;background:{c};border-radius:3px;margin:1px"></span>'
        for c in colors
    ])
    st.markdown(swatch_html, unsafe_allow_html=True)
    st.caption(PALETTES[palette]["description"])

    st.divider()

    if st.session_state.favorites:
        st.markdown(f"⭐ **{len(st.session_state.favorites)} favorited**")

    st.divider()
    st.markdown("📖 [GitHub](https://github.com/Yang1Bai/SciVizKit)")
    st.caption("v0.4 · MIT License")

# ── Load Data ────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Reset if new file
        if st.session_state.df is None or st.session_state.df.shape != df.shape:
            st.session_state.df = df
            st.session_state.generated_charts = {}
            st.session_state.generation_progress = {}
            analyzer = DataAnalyzer(df)
            st.session_state.profile = analyzer
            # Smart defaults for column mapping
            profile = analyzer
            st.session_state.col_mapping = {
                'x': profile.categorical_cols[0] if profile.categorical_cols else (profile.numeric_cols[0] if profile.numeric_cols else None),
                'y': profile.numeric_cols[0] if profile.numeric_cols else None,
                'color': profile.categorical_cols[1] if len(profile.categorical_cols) > 1 else None,
                'size': profile.numeric_cols[1] if len(profile.numeric_cols) > 1 else None,
            }
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()
else:
    df = None

# ── Landing Page ─────────────────────────────────────────────────────────────
if df is None:
    st.title("🔬 SciVizKit")
    st.subheader("Inspire the best visualization for your research data")
    st.markdown("*激发科研数据最佳可视化方案*")
    st.markdown("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 80+ Chart Types")
        st.markdown("Distribution, comparison, correlation, time series, network, scientific specialty, 3D, geographic, and more.")
    with col2:
        st.markdown("### 🌳 Smart Guide")
        st.markdown("Not sure which chart to use? Answer a few questions and get personalized recommendations.")
    with col3:
        st.markdown("### 🖼️ Figure Panel")
        st.markdown("Combine your favorite charts into a publication-ready multi-panel figure. Download PNG/SVG at 300 DPI.")

    st.markdown("---")
    st.markdown("### 🚀 Quick Start with Example Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧬 Biology / Genomics", use_container_width=True):
            st.session_state['load_example'] = 'biology'
    with col2:
        if st.button("⚗️ Chemistry / Materials", use_container_width=True):
            st.session_state['load_example'] = 'chemistry'
    with col3:
        if st.button("📊 General Dataset", use_container_width=True):
            st.session_state['load_example'] = 'general'

    # Load example data if button pressed
    if st.session_state.get('load_example'):
        example = st.session_state.pop('load_example')
        path_map = {
            'biology': 'examples/sample_biology.csv',
            'chemistry': 'examples/sample_chemistry.csv',
            'general': 'examples/sample_general.csv',
        }
        try:
            df = pd.read_csv(path_map[example])
            st.session_state.df = df
            st.session_state.generated_charts = {}
            analyzer = DataAnalyzer(df)
            st.session_state.profile = analyzer
            st.session_state.col_mapping = {
                'x': analyzer.categorical_cols[0] if analyzer.categorical_cols else (analyzer.numeric_cols[0] if analyzer.numeric_cols else None),
                'y': analyzer.numeric_cols[0] if analyzer.numeric_cols else None,
                'color': analyzer.categorical_cols[1] if len(analyzer.categorical_cols) > 1 else None,
                'size': analyzer.numeric_cols[1] if len(analyzer.numeric_cols) > 1 else None,
            }
            st.rerun()
        except Exception as e:
            st.error(f"Could not load example: {e}")

    st.stop()

# ── Main App (data loaded) ────────────────────────────────────────────────────
df = st.session_state.df
profile = st.session_state.profile

# Data summary bar
with st.expander(f"📋 Data: **{df.shape[0]:,}** rows × **{df.shape[1]}** columns", expanded=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df.head(5), use_container_width=True)
    with col2:
        st.markdown("**Column Types**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            nuniq = df[col].nunique()
            icon = "🔢" if pd.api.types.is_numeric_dtype(df[col]) else ("📅" if pd.api.types.is_datetime64_any_dtype(df[col]) else "🔤")
            st.markdown(f"{icon} `{col}` — {dtype}, {nuniq} unique")

# Column mapping panel
with st.expander("🎛️ Column Mapping", expanded=False):
    st.caption("Set which columns map to chart roles. Used as smart defaults for all charts.")
    cols = list(df.columns)
    none_option = ["(none)"] + cols
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_default = st.session_state.col_mapping.get('x')
        x_idx = none_option.index(x_default) if x_default in none_option else 0
        st.session_state.col_mapping['x'] = st.selectbox("X axis / Category", none_option, index=x_idx) or None
    with c2:
        y_default = st.session_state.col_mapping.get('y')
        y_idx = none_option.index(y_default) if y_default in none_option else 0
        st.session_state.col_mapping['y'] = st.selectbox("Y axis / Value", none_option, index=y_idx) or None
    with c3:
        color_default = st.session_state.col_mapping.get('color')
        color_idx = none_option.index(color_default) if color_default in none_option else 0
        st.session_state.col_mapping['color'] = st.selectbox("Color / Group", none_option, index=color_idx) or None
    with c4:
        size_default = st.session_state.col_mapping.get('size')
        size_idx = none_option.index(size_default) if size_default in none_option else 0
        st.session_state.col_mapping['size'] = st.selectbox("Size / Weight", none_option, index=size_idx) or None

    if st.button("🔄 Re-generate charts with new mapping"):
        st.session_state.generated_charts = {}
        st.session_state.generation_progress = {}
        st.rerun()

st.markdown("")

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Visualize Data", "🌳 Chart Guide", "🖼️ Figure Panel"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: VISUALIZE DATA
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    # Search + filter bar
    search_col, filter_col, fav_col = st.columns([3, 2, 1])
    with search_col:
        search = st.text_input("🔍 Search charts", placeholder="e.g. violin, scatter, PCA...", label_visibility="collapsed")
    with filter_col:
        all_cats = ["All"] + sorted(set(v['category'] for v in CHART_REGISTRY.values()))
        active_cat = st.selectbox("Category", all_cats, label_visibility="collapsed")
    with fav_col:
        show_favs = st.toggle("⭐ Only", help="Show favorited charts only")

    # Get compatible charts for current domain
    domain_map = {
        "General": "General",
        "Biology & Genomics": "Biology",
        "Chemistry & Materials": "Chemistry/Materials",
        "Medicine & Clinical": "Medicine",
        "Physics & Engineering": "Physics",
        "Social Science": "Social Science",
    }
    domain_key = domain_map.get(st.session_state.domain, "General")

    # Filter chart list
    def chart_matches(chart_id, chart):
        if domain_key != "General" and domain_key not in chart['domains'] and "General" not in chart['domains']:
            return False
        if search and search.lower() not in chart['name'].lower() and search.lower() not in chart.get('description', '').lower():
            return False
        if active_cat != "All" and chart['category'] != active_cat:
            return False
        if show_favs and chart_id not in st.session_state.favorites:
            return False
        return True

    filtered_charts = {k: v for k, v in CHART_REGISTRY.items() if chart_matches(k, v)}

    if not filtered_charts:
        st.info("No charts match your search. Try a different query or category.")
        st.stop()

    # Group by category
    categories_present = sorted(set(v['category'] for v in filtered_charts.values()))

    # ── Generate by Category (Lazy Loading) ──────────────────────────────────
    def generate_chart_for_id(chart_id: str, chart_meta: dict) -> dict:
        """Generate one chart and return result dict."""
        df = st.session_state.df
        mapping = st.session_state.col_mapping
        x_col = mapping.get('x')
        y_col = mapping.get('y')
        color_col = mapping.get('color')
        size_col = mapping.get('size')

        num_cols = profile.numeric_cols
        cat_cols = profile.categorical_cols
        dt_cols = getattr(profile, 'datetime_cols', [])

        fig_static, fig_plotly, code_str = None, None, ""
        error = None

        try:
            cat = chart_meta['category']
            cid = chart_id


            # Smart column selection per chart
            def first(*cols):
                for c in cols:
                    if c and c in df.columns:
                        return c
                return None

            num0 = first(y_col, num_cols[0] if num_cols else None)
            num1 = first(size_col, num_cols[1] if len(num_cols) > 1 else None)
            cat0 = first(x_col, cat_cols[0] if cat_cols else None)
            cat1 = first(color_col, cat_cols[1] if len(cat_cols) > 1 else None)

            if cat == "Distribution":
                if cid == "histogram":
                    fig_static, fig_plotly, code_str = dist_gen.histogram(df, num0)
                elif cid == "kde":
                    fig_static, fig_plotly, code_str = dist_gen.kde_plot(df, num0)
                elif cid == "violin" and cat0 and num0:
                    fig_static, fig_plotly, code_str = dist_gen.violin_plot(df, cat0, num0)
                elif cid == "boxplot" and cat0 and num0:
                    fig_static, fig_plotly, code_str = dist_gen.box_plot(df, cat0, num0)
                elif cid == "stripplot" and cat0 and num0:
                    fig_static, fig_plotly, code_str = dist_gen.strip_plot(df, cat0, num0)
                elif cid == "ecdf" and num0:
                    fig_static, fig_plotly, code_str = dist_gen.ecdf_plot(df, num0)
                elif cid == "qqplot" and num0:
                    fig_static, fig_plotly, code_str = dist_gen.qq_plot(df, num0)
                elif cid == "beeswarm" and cat0 and num0:
                    fig_static, fig_plotly, code_str = dist_gen.beeswarm_plot(df, cat0, num0)
                elif cid == "ridgeline" and num0 and cat0:
                    fig_static, fig_plotly, code_str = dist_gen.ridgeline_plot(df, num0, cat0)
                elif cid == "marginal_plot" and num0 and num1:
                    fig_static, fig_plotly, code_str = dist_gen.marginal_plot(df, num0, num1)
                elif cid == "raincloud" and cat0 and num0:
                    fig_static, fig_plotly, code_str = dist_gen.raincloud_plot(df, cat0, num0)

            elif cat == "Comparison":
                if cid == "bar" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.bar_chart(df, cat0, num0)
                elif cid == "grouped_bar" and cat0 and num0 and cat1:
                    fig_static, fig_plotly, code_str = comp_gen.grouped_bar(df, cat0, num0, cat1)
                elif cid == "stacked_bar" and cat0 and num0 and cat1:
                    fig_static, fig_plotly, code_str = comp_gen.stacked_bar(df, cat0, num0, cat1)
                elif cid == "lollipop" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.lollipop_chart(df, cat0, num0)
                elif cid == "dumbbell" and cat0 and num0 and num1:
                    fig_static, fig_plotly, code_str = comp_gen.dumbbell_chart(df, cat0, num0, num1)
                elif cid == "dotplot" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.dot_plot(df, cat0, num0)
                elif cid == "slope" and cat0 and num0 and num1:
                    fig_static, fig_plotly, code_str = comp_gen.slope_chart(df, cat0, num0, num1)
                elif cid == "waterfall" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.waterfall_chart(df, cat0, num0)
                elif cid == "errorbar" and cat0 and num0:
                    err_col = num1 or num0
                    fig_static, fig_plotly, code_str = comp_gen.error_bar_plot(df, cat0, num0, err_col)
                elif cid == "diverging_bar" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.diverging_bar(df, cat0, num0)
                elif cid == "population_pyramid" and num0 and num1 and cat0:
                    fig_static, fig_plotly, code_str = comp_gen.population_pyramid(df, cat0, num0, num1)
                elif cid == "percent_stacked_bar" and cat0 and num0 and cat1:
                    fig_static, fig_plotly, code_str = comp_gen.percent_stacked_bar(df, cat0, num0, cat1)
                elif cid == "radial_bar" and cat0 and num0:
                    fig_static, fig_plotly, code_str = comp_gen.radial_bar_chart(df, cat0, num0)
                elif cid == "bullet_chart" and cat0 and num0 and num1:
                    fig_static, fig_plotly, code_str = comp_gen.bullet_chart(df, cat0, num0, num1)
                elif cid == "tornado_chart" and cat0 and num0 and num1:
                    fig_static, fig_plotly, code_str = comp_gen.tornado_chart(df, cat0, num0, num1)

            elif cat == "Correlation":
                if cid == "scatter" and num0 and num1:
                    fig_static, fig_plotly, code_str = corr_gen.scatter_plot(df, num0, num1, color_col=cat0)
                elif cid == "bubble" and num0 and num1:
                    size_c = num1
                    fig_static, fig_plotly, code_str = corr_gen.bubble_chart(df, num0, num1, size_c, color_col=cat0)
                elif cid == "hexbin" and num0 and num1:
                    fig_static, fig_plotly, code_str = corr_gen.hexbin_plot(df, num0, num1)
                elif cid == "corr_heatmap" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = corr_gen.corr_heatmap(df)
                elif cid == "pairplot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = corr_gen.pairplot(df, num_cols[:4])
                elif cid == "parallel_coords" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = corr_gen.parallel_coords(df, num_cols[:5], color_col=cat0)
                elif cid == "contour_2d" and num0 and num1:
                    fig_static, fig_plotly, code_str = corr_gen.contour_2d(df, num0, num1)
                elif cid == "connected_scatter" and num0 and num1:
                    fig_static, fig_plotly, code_str = corr_gen.connected_scatter(df, num0, num1, cat0 or num1)
                elif cid == "marginal_scatter" and num0 and num1:
                    fig_static, fig_plotly, code_str = corr_gen.marginal_scatter(df, num0, num1, color_col=cat0)

            elif cat == "Time Series":
                x_ts = first(x_col, dt_cols[0] if dt_cols else cat0)
                if x_ts and num0:
                    if cid == "line":
                        fig_static, fig_plotly, code_str = ts_gen.line_chart(df, x_ts, [num0])
                    elif cid == "area":
                        fig_static, fig_plotly, code_str = ts_gen.area_chart(df, x_ts, num0)
                    elif cid == "stacked_area" and len(num_cols) >= 2:
                        fig_static, fig_plotly, code_str = ts_gen.stacked_area(df, x_ts, num_cols[:3])
                    elif cid == "step_line":
                        fig_static, fig_plotly, code_str = ts_gen.step_line(df, x_ts, num0)
                    elif cid == "streamgraph" and len(num_cols) >= 2:
                        fig_static, fig_plotly, code_str = ts_gen.streamgraph(df, x_ts, num_cols[:4])
                    elif cid == "bump_chart" and cat0:
                        fig_static, fig_plotly, code_str = ts_gen.bump_chart(df, x_ts, cat0, num0)
                    elif cid == "calendar_heatmap":
                        fig_static, fig_plotly, code_str = ts_gen.calendar_heatmap(df, x_ts, num0)
                    elif cid == "candlestick" and len(num_cols) >= 4:
                        fig_static, fig_plotly, code_str = ts_gen.candlestick_chart(df, x_ts, num_cols[0], num_cols[1], num_cols[2], num_cols[3])
                    elif cid == "spiral_chart":
                        fig_static, fig_plotly, code_str = ts_gen.spiral_chart(df, x_ts, num0)

            elif cat == "Proportional":
                if cid == "pie" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.pie_chart(df, cat0, num0)
                elif cid == "donut" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.donut_chart(df, cat0, num0)
                elif cid == "treemap" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.treemap(df, cat0, num0)
                elif cid == "sunburst" and cat0 and cat1 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.sunburst(df, cat0, cat1, num0)
                elif cid == "nightingale" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.nightingale_rose(df, cat0, num0)
                elif cid == "waffle" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.waffle_chart(df, cat0, num0)
                elif cid == "marimekko" and cat0 and cat1 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.marimekko_chart(df, cat0, cat1, num0)
                elif cid == "circle_packing" and cat0 and num0:
                    fig_static, fig_plotly, code_str = prop_gen.circle_packing(df, cat0, num0)

            elif cat == "Network":
                # For network charts, need source/target cols
                src = cat0
                tgt = cat1 or (cat_cols[2] if len(cat_cols) > 2 else cat0)
                if src and tgt:
                    if cid == "sankey":
                        fig_static, fig_plotly, code_str = net_gen.sankey_diagram(df, src, tgt, num0 or src)
                    elif cid == "network_graph":
                        fig_static, fig_plotly, code_str = net_gen.network_graph(df, src, tgt)
                    elif cid == "chord_diagram":
                        fig_static, fig_plotly, code_str = net_gen.chord_diagram(df, src, tgt, num0 or src)
                    elif cid == "arc_diagram":
                        fig_static, fig_plotly, code_str = net_gen.arc_diagram(df, src, tgt)
                    elif cid == "alluvial":
                        stage_cols = cat_cols[:3] if len(cat_cols) >= 2 else [cat0, cat0]
                        fig_static, fig_plotly, code_str = net_gen.alluvial_diagram(df, stage_cols)
                if cid == "dendrogram" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = net_gen.dendrogram_plot(df, num_cols[:6])

            elif cat == "Scientific":
                if cid == "volcano_plot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.volcano_plot(df, num_cols[0], num_cols[1])
                elif cid == "pca_plot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.pca_plot(df, num_cols[:6], color_col=cat0)
                elif cid == "umap_plot" and len(num_cols) >= 3:
                    fig_static, fig_plotly, code_str = sci_gen.umap_plot(df, num_cols[:6], color_col=cat0)
                elif cid == "tsne_plot" and len(num_cols) >= 3:
                    fig_static, fig_plotly, code_str = sci_gen.tsne_plot(df, num_cols[:6], color_col=cat0)
                elif cid == "roc_curve" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.roc_curve_plot(df, num_cols[0], num_cols[1])
                elif cid == "radar" and cat0 and len(num_cols) >= 3:
                    fig_static, fig_plotly, code_str = sci_gen.radar_chart(df, cat0, num_cols[:5])
                elif cid == "parity_plot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.parity_plot(df, num_cols[0], num_cols[1])
                elif cid == "bland_altman" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.bland_altman_plot(df, num_cols[0], num_cols[1])
                elif cid == "kaplan_meier" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.kaplan_meier_plot(df, num_cols[0], num_cols[1], group_col=cat0)
                elif cid == "manhattan_plot" and cat0 and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.manhattan_plot(df, cat0, num_cols[0], num_cols[1])
                elif cid == "forest_plot" and cat0 and len(num_cols) >= 2:
                    if len(num_cols) >= 3:
                        fig_static, fig_plotly, code_str = sci_gen.forest_plot(df, cat0, num_cols[0], num_cols[1], num_cols[2])
                    else:
                        fig_static, fig_plotly, code_str = sci_gen.forest_plot(df, cat0, num_cols[0], num_cols[1] if len(num_cols) > 1 else num_cols[0], None)
                elif cid == "funnel_plot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.funnel_plot(df, num_cols[0], num_cols[1])
                elif cid == "calibration_curve" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.calibration_curve(df, num_cols[0], num_cols[1])
                elif cid == "residual_plot" and len(num_cols) >= 2:
                    fig_static, fig_plotly, code_str = sci_gen.residual_plot(df, num_cols[0], num_cols[1])
                elif cid == "upset_plot" and len(cat_cols) >= 3:
                    fig_static, fig_plotly, code_str = sci_gen.upset_plot(df, cat_cols[:5])

            elif cat == "Text":
                try:
                    import src.generators.text_viz as text_gen
                    text_col_candidate = next((c for c in df.columns if df[c].dtype == object and df[c].str.len().mean() > 5), cat0)
                    if cid == "wordcloud" and text_col_candidate:
                        fig_static, fig_plotly, code_str = text_gen.wordcloud_plot(df, text_col_candidate)
                    elif cid == "venn":
                        # Build 2-3 sets from boolean/categorical cols
                        sets = {}
                        for c in cat_cols[:3]:
                            vals = df[c].dropna().unique()
                            if len(vals) == 2:
                                sets[c] = set(df[df[c] == vals[0]].index)
                        if len(sets) >= 2:
                            fig_static, fig_plotly, code_str = text_gen.venn_diagram(sets)
                except Exception:
                    pass

            elif cat == "Geographic":
                try:
                    import src.generators.geo_viz as geo_gen
                    if cid == "choropleth" and cat0 and num0:
                        fig_static, fig_plotly, code_str = geo_gen.choropleth_map(df, cat0, num0)
                    elif cid == "bubble_map" and len(num_cols) >= 3:
                        fig_static, fig_plotly, code_str = geo_gen.bubble_map(df, num_cols[0], num_cols[1], num_cols[2])
                except Exception:
                    pass

            elif cat == "3D":
                try:
                    import src.generators.threed_viz as viz3d
                    if cid == "scatter_3d" and len(num_cols) >= 3:
                        fig_static, fig_plotly, code_str = viz3d.scatter_3d(df, num_cols[0], num_cols[1], num_cols[2], color_col=cat0)
                    elif cid == "surface_3d" and len(num_cols) >= 3:
                        fig_static, fig_plotly, code_str = viz3d.surface_3d(df, num_cols[0], num_cols[1], num_cols[2])
                    elif cid == "bar_3d" and len(num_cols) >= 1 and len(cat_cols) >= 2:
                        fig_static, fig_plotly, code_str = viz3d.bar_3d(df, cat0, cat1, num0)
                except Exception:
                    pass

        except Exception as e:
            error = traceback.format_exc()

        if fig_static:
            plt.close(fig_static)
        return {
            'name': chart_meta['name'],
            'category': chart_meta['category'],
            'description': chart_meta.get('description', ''),
            'when_to_use': chart_meta.get('when_to_use', ''),
            'fig_static': fig_static,
            'fig_plotly': fig_plotly,
            'code_str': code_str,
            'error': error,
        }

    # ── Chart Grid UI ─────────────────────────────────────────────────────────
    # Show charts by category with lazy "Generate" button per category
    for category in categories_present:
        cat_charts = {k: v for k, v in filtered_charts.items() if v['category'] == category}
        if not cat_charts:
            continue

        color = CATEGORY_COLORS.get(category, "#667eea")
        generated_count = sum(
            1 for cid in cat_charts
            if cid in st.session_state.generated_charts
            and st.session_state.generated_charts[cid].get('fig_static') is not None
        )

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;margin:16px 0 8px 0">'
            f'<span style="background:{color};color:white;padding:3px 12px;border-radius:12px;font-weight:600;font-size:13px">{category}</span>'
            f'<span style="color:#888;font-size:13px">{len(cat_charts)} charts'
            f'{f" · {generated_count} generated" if generated_count else ""}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Check if already generated
        cat_generated = all(cid in st.session_state.generated_charts for cid in cat_charts)

        if not cat_generated:
            if st.button(f"⚡ Generate {category} charts ({len(cat_charts)})", key=f"gen_{category}"):
                prog_bar = st.progress(0, text=f"Generating {category}...")
                chart_ids = list(cat_charts.keys())
                for i, chart_id in enumerate(chart_ids):
                    if chart_id not in st.session_state.generated_charts:
                        result = generate_chart_for_id(chart_id, CHART_REGISTRY[chart_id])
                        st.session_state.generated_charts[chart_id] = result
                        # Store for figure panel
                        if result.get('fig_static'):
                            if 'generated_charts_panel' not in st.session_state:
                                st.session_state['generated_charts_panel'] = {}
                            st.session_state['generated_charts_panel'][result['name']] = {
                                'fig_static': result['fig_static'],
                                'fig_plotly': result['fig_plotly']
                            }
                    prog_bar.progress(
                        (i + 1) / len(chart_ids),
                        text=f"Generated {i+1}/{len(chart_ids)}: {CHART_REGISTRY[chart_id]['name']}"
                    )
                prog_bar.empty()
                st.rerun()
        else:
            # Show chart grid (3 columns)
            chart_items = list(cat_charts.items())
            for row_start in range(0, len(chart_items), 3):
                row_items = chart_items[row_start:row_start+3]
                cols = st.columns(3)
                for col_idx, (chart_id, chart_meta) in enumerate(row_items):
                    with cols[col_idx]:
                        result = st.session_state.generated_charts.get(chart_id, {})
                        is_fav = chart_id in st.session_state.favorites

                        # Chart name + favorite button
                        name_col, fav_btn_col = st.columns([5, 1])
                        with name_col:
                            st.markdown(f"<div class='chart-name'>{chart_meta['name']}</div>", unsafe_allow_html=True)
                        with fav_btn_col:
                            if st.button("⭐" if is_fav else "☆", key=f"fav_{chart_id}", help="Favorite"):
                                if is_fav:
                                    st.session_state.favorites.discard(chart_id)
                                else:
                                    st.session_state.favorites.add(chart_id)
                                st.rerun()

                        # Show chart thumbnail
                        if result.get('fig_static'):
                            try:
                                buf = io.BytesIO()
                                result['fig_static'].savefig(buf, format='png', dpi=80, bbox_inches='tight')
                                buf.seek(0)
                                st.image(buf, use_container_width=True)
                            except Exception:
                                st.warning("Preview unavailable")

                            # Action buttons
                            mode = st.session_state.chart_mode
                            b1, b2, b3 = st.columns(3)
                            with b1:
                                # Download PNG
                                try:
                                    dl_buf = io.BytesIO()
                                    result['fig_static'].savefig(dl_buf, format='png', dpi=300, bbox_inches='tight')
                                    dl_buf.seek(0)
                                    st.download_button(
                                        "⬇️ PNG", data=dl_buf,
                                        file_name=f"{chart_id}.png", mime="image/png",
                                        key=f"dl_{chart_id}", use_container_width=True
                                    )
                                except Exception:
                                    pass
                            with b2:
                                # Show code
                                if st.button("💻 Code", key=f"code_{chart_id}", use_container_width=True):
                                    st.session_state[f'show_code_{chart_id}'] = not st.session_state.get(f'show_code_{chart_id}', False)
                                    st.rerun()
                            with b3:
                                # Add to panel
                                in_panel = chart_meta['name'] in st.session_state.get('panel_selection', set())
                                if st.button(
                                    "🖼️ +Panel" if not in_panel else "🖼️ ✓",
                                    key=f"panel_{chart_id}", use_container_width=True
                                ):
                                    if 'panel_selection' not in st.session_state:
                                        st.session_state['panel_selection'] = set()
                                    if in_panel:
                                        st.session_state['panel_selection'].discard(chart_meta['name'])
                                    else:
                                        st.session_state['panel_selection'].add(chart_meta['name'])
                                    st.rerun()

                            # Interactive chart
                            if mode in ["Interactive", "Both"] and result.get('fig_plotly'):
                                with st.expander("📈 Interactive version"):
                                    st.plotly_chart(result['fig_plotly'], use_container_width=True, key=f"plotly_{chart_id}")

                            # Code block
                            if st.session_state.get(f'show_code_{chart_id}') and result.get('code_str'):
                                with st.expander("💻 Python Code", expanded=True):
                                    st.code(result['code_str'], language='python')

                            # When to use
                            st.caption(f"💡 {chart_meta.get('when_to_use', '')}")

                        elif result.get('error'):
                            st.markdown(
                                '<div style="background:#fff3cd;border-radius:6px;padding:8px;font-size:12px;color:#856404">⚠️ Not applicable for this data</div>',
                                unsafe_allow_html=True
                            )
                            with st.expander("Why?"):
                                st.caption(result['error'][:300])
                        else:
                            st.markdown(
                                '<div style="background:#f8f9fa;border-radius:6px;padding:30px;text-align:center;color:#aaa">Not generated yet</div>',
                                unsafe_allow_html=True
                            )
                        st.markdown("")

    # Generate All button at bottom
    st.divider()
    not_generated = [cid for cid in filtered_charts if cid not in st.session_state.generated_charts]
    if not_generated:
        if st.button(f"⚡ Generate ALL remaining charts ({len(not_generated)})", type="primary", use_container_width=True):
            prog = st.progress(0, text="Generating all charts...")
            for i, chart_id in enumerate(not_generated):
                result = generate_chart_for_id(chart_id, CHART_REGISTRY[chart_id])
                st.session_state.generated_charts[chart_id] = result
                if result.get('fig_static'):
                    if 'generated_charts_panel' not in st.session_state:
                        st.session_state['generated_charts_panel'] = {}
                    st.session_state['generated_charts_panel'][result['name']] = {
                        'fig_static': result['fig_static'],
                        'fig_plotly': result['fig_plotly']
                    }
                prog.progress(
                    (i + 1) / len(not_generated),
                    text=f"{i+1}/{len(not_generated)}: {CHART_REGISTRY[chart_id]['name']}"
                )
            prog.empty()
            st.rerun()
    else:
        st.success(f"✅ All {len(filtered_charts)} charts generated!")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: CHART GUIDE (Decision Tree)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🌳 Chart Selection Guide")
    st.caption("Answer a few questions to find the best chart for your data and goal.")

    from src.decision_tree import DECISION_TREE, get_recommendations

    if "dt_path" not in st.session_state:
        st.session_state.dt_path = []

    path = st.session_state.dt_path

    # Traverse tree
    node = DECISION_TREE["start"]
    for choice in path:
        options = node.get("options", {})
        next_node = options.get(choice)
        if next_node is None:
            break
        if isinstance(next_node, str):
            next_node = DECISION_TREE.get(next_node, {})
        node = next_node
        if isinstance(node, dict) and "recommend" in node:
            break

    # Show breadcrumb
    if path:
        breadcrumb = " → ".join([p[:30] for p in path])
        st.caption(f"📍 {breadcrumb}")

    if isinstance(node, dict) and "recommend" in node:
        st.success(f"**Recommended charts:** {', '.join(node['recommend'])}")
        st.info(f"💡 {node['reason']}")
        st.markdown("---")
        st.markdown("**These charts work best for your use case:**")
        for chart_id in node["recommend"]:
            if chart_id in CHART_REGISTRY:
                chart = CHART_REGISTRY[chart_id]
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**{chart['name']}**")
                    st.caption(chart['description'])
                with col2:
                    if chart_id in st.session_state.generated_charts and st.session_state.generated_charts[chart_id].get('fig_static'):
                        try:
                            buf = io.BytesIO()
                            st.session_state.generated_charts[chart_id]['fig_static'].savefig(buf, format='png', dpi=60, bbox_inches='tight')
                            buf.seek(0)
                            st.image(buf, width=120)
                        except Exception:
                            pass
                st.markdown("")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.dt_path = []
                st.rerun()
        with col2:
            if st.button("← Back one step", use_container_width=True):
                if st.session_state.dt_path:
                    st.session_state.dt_path.pop()
                    st.rerun()
    else:
        if node and "question" in node:
            st.markdown(f"### {node['question']}")
            st.markdown("")
            options = list(node.get("options", {}).keys())
            for opt in options:
                if st.button(opt, key=f"dt_{opt[:30]}_{len(path)}", use_container_width=True):
                    st.session_state.dt_path.append(opt)
                    st.rerun()
            if path:
                st.markdown("")
                if st.button("← Back", use_container_width=True):
                    st.session_state.dt_path.pop()
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: FIGURE PANEL
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🖼️ Figure Panel Builder")
    st.caption("Combine multiple charts into a single publication-ready figure.")

    from src.figure_panel import combine_figures, fig_to_bytes

    # Use generated_charts_panel or generated_charts
    panel_source = st.session_state.get('generated_charts_panel') or {}
    if not panel_source:
        # Fallback: build from generated_charts
        for cid, res in st.session_state.get('generated_charts', {}).items():
            if res.get('fig_static'):
                panel_source[res['name']] = {
                    'fig_static': res['fig_static'],
                    'fig_plotly': res.get('fig_plotly')
                }

    if not panel_source:
        st.info("👆 Go to **Visualize Data** tab and generate some charts first, then come back here.")
    else:
        available = list(panel_source.keys())
        pre_selected = list(
            st.session_state.get('panel_selection', set()) & set(available)
        ) or available[:min(4, len(available))]

        selected = st.multiselect("Select charts for the panel:", options=available, default=pre_selected)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ncols = st.slider("Columns", 1, 4, min(2, len(selected)) if selected else 2)
        with col2:
            panel_w = st.slider("Panel width (in)", 4, 10, 6)
        with col3:
            panel_h = st.slider("Panel height (in)", 3, 8, 4)
        with col4:
            export_dpi = st.select_slider("DPI", [72, 150, 300, 600], value=300)

        add_labels = st.checkbox("Add A, B, C... panel labels", value=True)

        if selected and st.button("🖼️ Build Figure Panel", type="primary", use_container_width=True):
            figs = [
                (name, panel_source[name]['fig_static'])
                for name in selected
                if panel_source[name].get('fig_static')
            ]
            if figs:
                with st.spinner("Building combined figure..."):
                    combined = combine_figures(
                        figs, ncols=ncols, panel_labels=add_labels,
                        figsize_per_panel=(panel_w, panel_h), dpi=export_dpi
                    )
                if combined:
                    st.pyplot(combined)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button(
                            "⬇️ Download PNG (300 DPI)",
                            data=fig_to_bytes(combined, dpi=export_dpi, fmt='png'),
                            file_name="figure_panel.png", mime="image/png",
                            use_container_width=True
                        )
                    with c2:
                        st.download_button(
                            "⬇️ Download SVG (Vector)",
                            data=fig_to_bytes(combined, dpi=export_dpi, fmt='svg'),
                            file_name="figure_panel.svg", mime="image/svg+xml",
                            use_container_width=True
                        )
