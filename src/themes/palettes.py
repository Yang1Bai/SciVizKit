"""Publication-quality color palettes for scientific journals."""

PALETTES = {
    "Nature": {
        "categorical": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000", "#7E6148"],
        "sequential": "YlOrRd",
        "diverging": "RdBu",
        "description": "Nature journal family colors"
    },
    "Science": {
        "categorical": ["#3B4992", "#EE0000", "#008B45", "#631879", "#008280", "#BB0021", "#5F559B", "#A20056", "#808180"],
        "sequential": "Blues",
        "diverging": "bwr",
        "description": "Science journal family colors"
    },
    "Cell": {
        "categorical": ["#00BCD4", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#2196F3", "#F44336", "#009688", "#607D8B"],
        "sequential": "viridis",
        "diverging": "coolwarm",
        "description": "Cell Press journal colors"
    },
    "ACS": {
        "categorical": ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22"],
        "sequential": "plasma",
        "diverging": "seismic",
        "description": "American Chemical Society style"
    },
    "Colorblind Safe": {
        "categorical": ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#D55E00", "#CC79A7", "#000000"],
        "sequential": "cividis",
        "diverging": "PuOr",
        "description": "Accessible to colorblind readers (Wong palette)"
    },
    "Default": {
        "categorical": ["#667eea", "#f093fb", "#4facfe", "#43e97b", "#fa709a", "#fee140", "#30cfd0", "#a18cd1"],
        "sequential": "viridis",
        "diverging": "RdYlBu",
        "description": "SciVizKit default"
    }
}


def get_palette(name: str) -> dict:
    return PALETTES.get(name, PALETTES["Default"])


def apply_palette_to_fig_matplotlib(fig, palette_name: str):
    """Apply categorical colors to all axes in a matplotlib figure."""
    import matplotlib.pyplot as plt
    palette = get_palette(palette_name)
    colors = palette["categorical"]
    for ax in fig.get_axes():
        for i, line in enumerate(ax.get_lines()):
            line.set_color(colors[i % len(colors)])
    return fig
