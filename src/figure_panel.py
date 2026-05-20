"""
Figure Panel — combine multiple charts into a publication-ready figure.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import io
import base64
from typing import List, Tuple, Optional


def combine_figures(
    figures: List[Tuple[str, plt.Figure]],
    ncols: int = 2,
    panel_labels: bool = True,
    figsize_per_panel: Tuple[float, float] = (6, 4),
    dpi: int = 300,
) -> plt.Figure:
    """
    Combine multiple matplotlib figures into a single publication-ready figure.

    Args:
        figures: List of (label, fig) tuples — label is chart name, fig is matplotlib Figure
        ncols: Number of columns in the grid
        panel_labels: Whether to add A, B, C, D... labels to panels
        figsize_per_panel: Size of each panel in inches
        dpi: Output DPI

    Returns:
        Combined matplotlib Figure
    """
    import string
    n = len(figures)
    if n == 0:
        return None

    nrows = (n + ncols - 1) // ncols
    fig_width = figsize_per_panel[0] * ncols
    fig_height = figsize_per_panel[1] * nrows

    combined = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

    labels = list(string.ascii_uppercase)

    for i, (name, src_fig) in enumerate(figures):
        ax = combined.add_subplot(nrows, ncols, i + 1)

        # Render source figure to image and display in combined figure
        buf = io.BytesIO()
        src_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        import matplotlib.image as mpimg
        img = mpimg.imread(buf)
        ax.imshow(img)
        ax.axis('off')

        # Add panel label (A, B, C...)
        if panel_labels and i < len(labels):
            ax.text(
                0.02, 0.98, labels[i],
                transform=ax.transAxes,
                fontsize=14, fontweight='bold',
                va='top', ha='left',
                color='black'
            )

        # Add chart name as title
        ax.set_title(name, fontsize=10, pad=4)

    combined.tight_layout(pad=1.5)
    return combined


def fig_to_bytes(fig: plt.Figure, dpi: int = 300, fmt: str = 'png') -> bytes:
    """Convert a matplotlib figure to bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()
