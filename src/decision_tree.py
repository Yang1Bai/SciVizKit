"""
Chart Decision Tree — guides users to the right chart type
based on their data characteristics and visualization goal.
"""

DECISION_TREE = {
    "start": {
        "question": "What is your main visualization goal?",
        "options": {
            "📊 Compare values across categories": "compare",
            "📈 Show change over time": "time",
            "🔗 Show relationship between variables": "relation",
            "🥧 Show part-to-whole / proportions": "proportion",
            "📉 Show distribution of data": "distribution",
            "🌐 Show geographic data": "geo",
            "🧬 Scientific / specialty chart": "science",
        }
    },
    "compare": {
        "question": "How many categories do you have?",
        "options": {
            "Few categories (2–7)": "compare_few",
            "Many categories (8+)": "compare_many",
            "Two groups to compare before/after": "compare_two",
        }
    },
    "compare_few": {
        "question": "Do you have subcategories?",
        "options": {
            "No subcategories": {"recommend": ["bar", "lollipop", "dot_plot"], "reason": "Simple comparison — bar or lollipop chart works best."},
            "Yes, show breakdown": {"recommend": ["grouped_bar", "stacked_bar", "percent_stacked_bar"], "reason": "Grouped or stacked bar to show subcategory structure."},
            "Show deviation from baseline": {"recommend": ["diverging_bar", "waterfall_chart"], "reason": "Diverging bar or waterfall for positive/negative differences."},
        }
    },
    "compare_many": {
        "question": "What matters most?",
        "options": {
            "Ranking / top values": {"recommend": ["bar", "lollipop", "dot_plot"], "reason": "Horizontal bar chart is ideal for many categories."},
            "Sensitivity / impact analysis": {"recommend": ["tornado_chart", "bullet_chart"], "reason": "Tornado chart shows which factors matter most."},
        }
    },
    "compare_two": {
        "question": "What type of comparison?",
        "options": {
            "Change between two timepoints per item": {"recommend": ["dumbbell", "slope_chart"], "reason": "Dumbbell or slope chart clearly shows change per item."},
            "Two demographic groups (e.g., male/female)": {"recommend": ["population_pyramid"], "reason": "Population pyramid is the standard for two-group demographics."},
        }
    },
    "time": {
        "question": "What aspect of time do you want to show?",
        "options": {
            "Trend over time": {"recommend": ["line", "area"], "reason": "Line chart is the classic for trends; area for volume emphasis."},
            "Multiple series over time": {"recommend": ["line", "stacked_area", "streamgraph"], "reason": "Multi-series line or stacked area."},
            "Ranking changes over time": {"recommend": ["bump_chart"], "reason": "Bump chart tracks rank positions over time."},
            "Daily/seasonal patterns": {"recommend": ["calendar_heatmap", "spiral_chart"], "reason": "Calendar heatmap reveals day-by-day patterns."},
            "Financial OHLC data": {"recommend": ["candlestick"], "reason": "Candlestick is the standard for OHLC financial data."},
        }
    },
    "relation": {
        "question": "How many variables?",
        "options": {
            "Two continuous variables": {"recommend": ["scatter", "hexbin", "contour_2d"], "reason": "Scatter plot; hexbin/contour for dense data."},
            "Three variables (x, y, size)": {"recommend": ["bubble_chart"], "reason": "Bubble chart encodes a third variable as size."},
            "All pairwise relationships": {"recommend": ["pairplot", "corr_heatmap", "parallel_coords"], "reason": "Pair plot or correlation heatmap for all relationships at once."},
            "Trajectory / connected path": {"recommend": ["connected_scatter"], "reason": "Connected scatter shows the path through data space."},
            "High-dimensional data": {"recommend": ["pca_plot", "umap_plot", "tsne_plot", "parallel_coords"], "reason": "Dimensionality reduction to visualize clusters."},
        }
    },
    "proportion": {
        "question": "How many categories?",
        "options": {
            "2–5 categories": {"recommend": ["pie", "donut", "waffle"], "reason": "Pie or donut for simple proportions; waffle for count-based."},
            "5+ categories, flat": {"recommend": ["treemap", "nightingale"], "reason": "Treemap uses area to show proportions effectively."},
            "Hierarchical (nested categories)": {"recommend": ["sunburst", "circle_packing"], "reason": "Sunburst or circle packing for nested proportions."},
            "Two categorical dimensions": {"recommend": ["marimekko"], "reason": "Marimekko encodes both dimensions proportionally."},
        }
    },
    "distribution": {
        "question": "How many groups?",
        "options": {
            "One variable": {"recommend": ["histogram", "kde", "ecdf"], "reason": "Histogram or KDE for a single distribution."},
            "Compare 2–4 groups": {"recommend": ["violin", "boxplot", "raincloud"], "reason": "Violin + box shows shape and summary stats; raincloud adds data points."},
            "Compare many groups": {"recommend": ["ridgeline", "beeswarm", "stripplot"], "reason": "Ridgeline/joy plot is excellent for many groups side by side."},
            "Two variables jointly": {"recommend": ["marginal_plot", "marginal_scatter"], "reason": "Marginal plot shows joint distribution + each marginal."},
        }
    },
    "geo": {
        "question": "What geographic data do you have?",
        "options": {
            "Country/region names with values": {"recommend": ["choropleth"], "reason": "Choropleth map — color regions by value."},
            "Lat/lon coordinates with sizes": {"recommend": ["bubble_map"], "reason": "Bubble map — plot sized circles at locations."},
        }
    },
    "science": {
        "question": "Which scientific domain?",
        "options": {
            "Genomics / differential expression": {"recommend": ["volcano_plot", "manhattan_plot", "pca_plot", "umap_plot"], "reason": "Volcano plot for DE genes; Manhattan for GWAS."},
            "Clinical / survival analysis": {"recommend": ["kaplan_meier", "roc_curve", "forest_plot", "bland_altman_plot"], "reason": "Kaplan-Meier for survival; ROC for classifier performance."},
            "ML model evaluation": {"recommend": ["roc_curve", "calibration_curve", "parity_plot", "residual_plot", "confusion_heatmap"], "reason": "ROC + calibration + residuals for complete model diagnostics."},
            "Meta-analysis": {"recommend": ["forest_plot", "funnel_plot"], "reason": "Forest plot for effect sizes; funnel for publication bias."},
            "Materials / chemistry": {"recommend": ["scatter", "parity_plot", "corr_heatmap", "pca_plot"], "reason": "Scatter with fit + parity plot for property-performance relationships."},
            "Network / flow data": {"recommend": ["sankey", "chord_diagram", "network_graph", "alluvial"], "reason": "Sankey for flows; chord for bidirectional; network for connections."},
        }
    }
}


def get_recommendations(path: list) -> dict | None:
    """
    Traverse the decision tree given a list of chosen option keys.
    Returns the leaf node (with 'recommend' and 'reason') or None if still navigating.
    """
    node = DECISION_TREE["start"]
    for choice in path:
        options = node.get("options", {})
        node = options.get(choice)
        if node is None:
            return None
        # If the value is a string, it's a reference to another tree node
        if isinstance(node, str):
            node = DECISION_TREE.get(node)
            if node is None:
                return None
        if isinstance(node, dict) and "recommend" in node:
            return node
    return node
