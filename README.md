# 🔬 SciVizKit — Scientific Visualization Toolkit

*Inspire the best visualization for your research data | 激发科研数据最佳可视化方案*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Charts](https://img.shields.io/badge/Charts-50%2B-667eea)](.)
[![Domains](https://img.shields.io/badge/Domains-6-764ba2)](.)

---

## ✨ Features

- 📊 **50+ chart types** spanning 7 visualization categories
- 🤖 **Auto-detection** of column types (numeric, categorical, datetime, binary, high-cardinality)
- 🎯 **Domain-aware** recommendations — Biology, Chemistry/Materials, Medicine, Physics, Social Science, General
- 🖼️ **Dual rendering** — static (matplotlib/seaborn) AND interactive (Plotly) for every chart
- 💻 **Copy-paste Python code** — every chart comes with production-ready code
- ⬇️ **PNG download** for any static figure
- 📁 **CSV & Excel** upload support (up to 200 MB)
- 🚀 **Zero-config** — upload data, click generate, explore instantly

---

## 🗂️ Chart Types

| Category | Charts | Count |
|----------|--------|-------|
| **Distribution** | Histogram, KDE, Violin, Box Plot, Strip Plot, Beeswarm, ECDF, Q-Q Plot | 8 |
| **Comparison** | Bar, Grouped Bar, Stacked Bar, Lollipop, Dumbbell, Dot Plot, Slope, Waterfall, Error Bar | 9 |
| **Correlation** | Scatter, Bubble, Hexbin, Correlation Heatmap, Pair Plot, Parallel Coordinates | 6 |
| **Time Series** | Line, Area, Stacked Area, Step Line | 4 |
| **Proportional** | Pie, Donut, Treemap, Sunburst, Nightingale Rose | 5 |
| **Network** | Sankey Diagram, Network Graph, Dendrogram | 3 |
| **Scientific** | Volcano Plot, PCA Plot, ROC Curve, Radar Chart, Parity Plot, Bland-Altman, Kaplan-Meier | 7 |
| **Total** | | **42+** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/SciVizKit.git
cd SciVizKit

# (Optional) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

The app opens at `http://localhost:8501` 🎉

### Using the App

1. **Upload** your CSV or Excel file via the sidebar (or load an example dataset)
2. **Select** your research domain (General, Biology, Chemistry, Medicine, Physics, Social Science)
3. **Choose** chart mode: Interactive / Static / Both
4. **Click** "🚀 Generate All Visualizations"
5. **Explore** charts grouped by category — download PNGs, copy code, and inspect tips

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `matplotlib` | Static chart rendering |
| `seaborn` | Statistical visualization |
| `plotly` | Interactive charts |
| `scipy` | Scientific computing (Q-Q plot, dendrogram) |
| `scikit-learn` | PCA, ROC curve |
| `squarify` | Treemap layout |
| `networkx` | Network graph |
| `lifelines` | Kaplan-Meier survival analysis |
| `kaleido` | Plotly PNG export |
| `openpyxl` | Excel file reading |

---

## ☁️ Deploy to Streamlit Cloud

1. **Fork** this repository on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your fork → branch: `main` → main file: `app.py`
5. Click **Deploy** — done in ~2 minutes!

> **Note:** Streamlit Cloud has a 1 GB memory limit. `umap-learn` and `lifelines` are optional; the app degrades gracefully if they are absent.

---

## 📂 Project Structure

```
SciVizKit/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml           # Theme & server config
├── src/
│   ├── data_analyzer.py      # DataFrame profiling & chart compatibility
│   ├── chart_registry.py     # Central registry of 50+ chart types
│   ├── code_generator.py     # Copy-paste Python code templates
│   └── generators/
│       ├── distribution.py   # Histogram, KDE, Violin, Box, Strip, Beeswarm, ECDF, Q-Q
│       ├── comparison.py     # Bar, Grouped Bar, Stacked, Lollipop, Dumbbell, Slope…
│       ├── correlation.py    # Scatter, Bubble, Hexbin, Heatmap, Pairplot, Parallel
│       ├── timeseries.py     # Line, Area, Stacked Area, Step Line
│       ├── proportional.py   # Pie, Donut, Treemap, Sunburst, Nightingale
│       ├── network.py        # Sankey, Network Graph, Dendrogram
│       └── scientific.py     # Volcano, PCA, ROC, Radar, Parity, Bland-Altman, KM
└── examples/
    ├── sample_general.csv    # General dataset (200 rows)
    ├── sample_biology.csv    # Genomics/DEG dataset (300 rows)
    └── sample_chemistry.csv  # Reaction yield dataset (150 rows)
```

---

## 🧪 Example Datasets

| File | Description | Rows | Key columns |
|------|-------------|------|-------------|
| `sample_general.csv` | Demographics & scores | 200 | age, income, education, region, score, date |
| `sample_biology.csv` | Differential expression | 300 | gene_name, log2FoldChange, pvalue, padj, group |
| `sample_chemistry.csv` | Reaction optimisation | 150 | temperature, pressure, yield_pct, catalyst, solvent |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-chart`
3. Add your chart to the registry (`src/chart_registry.py`) and implement it in the appropriate generator module
4. Update `app.py`'s `fn_map` and `pick_cols` functions
5. Submit a pull request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built with ❤️ using [Streamlit](https://streamlit.io), [Plotly](https://plotly.com), [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org), [scikit-learn](https://scikit-learn.org), and many other open-source tools.

---

*Made for researchers, data scientists, and anyone who loves great charts.*
