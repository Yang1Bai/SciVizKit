"""
SciVizKit — Text visualization generators
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

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    from matplotlib_venn import venn2, venn3
    HAS_VENN = True
except ImportError:
    HAS_VENN = False


# ── Word Cloud ────────────────────────────────────────────────────────

def wordcloud_plot(df: pd.DataFrame, text_col: str, color_by: str = None):
    """Word cloud from text column. Uses wordcloud library."""
    try:
        if not HAS_WORDCLOUD:
            return None, None, "# wordcloud not installed. pip install wordcloud"

        text = " ".join(df[text_col].dropna().astype(str).tolist())
        if not text.strip():
            return None, None, "# No text data found"

        wc = WordCloud(
            width=800, height=500,
            background_color="white",
            max_words=200,
            colormap="viridis",
        ).generate(text)

        fig_s, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud: {text_col}")
        fig_s.tight_layout()

        # Plotly version: word frequency bar chart
        from collections import Counter
        words = text.lower().split()
        # Basic stopword removal
        stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at",
                     "to", "for", "of", "with", "by", "from", "is", "it",
                     "that", "this", "are", "was", "were", "be", "been"}
        words = [w.strip(".,!?;:()[]{}\"'") for w in words
                 if w not in stopwords and len(w) > 2]
        counter = Counter(words).most_common(30)
        if counter:
            words_top, freqs = zip(*counter)
            fig_p = go.Figure(go.Bar(
                x=list(words_top), y=list(freqs),
                marker_color="steelblue",
            ))
            fig_p.update_layout(title=f"Top Words in {text_col}",
                                xaxis_title="Word", yaxis_title="Frequency")
        else:
            fig_p = None

        code = f"""# Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt
text = ' '.join(df['{text_col}'].dropna().astype(str))
wc = WordCloud(width=800, height=500, background_color='white').generate(text)
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(wc, interpolation='bilinear')
ax.axis('off')
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"


# ── Venn Diagram ──────────────────────────────────────────────────────

def venn_diagram(sets_dict: dict):
    """Venn diagram for 2-3 sets. Uses matplotlib-venn."""
    try:
        if not HAS_VENN:
            return None, None, "# matplotlib-venn not installed. pip install matplotlib-venn"

        n = len(sets_dict)
        if n < 2 or n > 3:
            return None, None, "# venn_diagram requires exactly 2 or 3 sets"

        fig_s, ax = plt.subplots(figsize=(8, 6))
        names = list(sets_dict.keys())
        sets = [set(v) for v in sets_dict.values()]

        if n == 2:
            venn2(sets, set_labels=names, ax=ax)
        else:
            venn3(sets, set_labels=names, ax=ax)

        ax.set_title("Venn Diagram")
        fig_s.tight_layout()

        # Plotly: Euler-like representation using overlapping circles
        sizes = [len(s) for s in sets]
        fig_p = go.Figure()
        positions = [(0.3, 0.5), (0.7, 0.5), (0.5, 0.8)]
        for i, (name, sz) in enumerate(zip(names, sizes)):
            if i < len(positions):
                x0, y0 = positions[i]
                r = 0.2
                theta = np.linspace(0, 2 * np.pi, 100)
                fig_p.add_trace(go.Scatter(
                    x=x0 + r * np.cos(theta),
                    y=y0 + r * np.sin(theta),
                    fill="toself",
                    fillcolor=f"rgba({50+i*80},{100+i*50},{200-i*60},0.3)",
                    line=dict(width=2),
                    name=f"{name} (n={sz})",
                ))
        fig_p.update_layout(title="Venn Diagram",
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False,
                                       showticklabels=False, scaleanchor="x"))

        code = """# Venn Diagram
from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 6))
venn2([set_a, set_b], set_labels=('Set A', 'Set B'), ax=ax)
plt.tight_layout()
"""
        return fig_s, fig_p, code
    except Exception as e:
        return None, None, f"# Error: {e}"
