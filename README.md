# 🔬 SciVizKit — Scientific Visualization Toolkit

*Inspire the best visualization for your research data | 激发科研数据最佳可视化方案*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Charts](https://img.shields.io/badge/Charts-80%2B-667eea)](.)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B)](https://scivizkit-hvbzujsahst6uec2pupvrx.streamlit.app/)

---

## 🚀 Live Demo | 在线体验

**👉 [https://scivizkit-hvbzujsahst6uec2pupvrx.streamlit.app/](https://scivizkit-hvbzujsahst6uec2pupvrx.streamlit.app/)**

> No installation needed — upload your data and explore 80+ chart types instantly.
> 无需安装 — 直接上传数据，即刻探索 80+ 种科研图表。

---

> **[English](#english) | [中文](#中文)**

---

<a name="english"></a>
## 🇬🇧 English

### What is SciVizKit?

SciVizKit is an open-source scientific visualization toolkit built with Streamlit. Upload any dataset (CSV or Excel), and it automatically generates **80+ chart types** across 10 categories — from standard statistics charts to domain-specific scientific plots. Pick the best one for your paper, download publication-quality figures (300 DPI), and copy the Python code.

### ✨ Features

- 📊 **80+ chart types** across 10 categories (Distribution, Comparison, Correlation, Time Series, Proportional, Network, Scientific, Text, Geographic, 3D)
- 🌳 **Chart Guide** — interactive decision tree that recommends the best chart based on your data and goal
- 🖼️ **Figure Panel Builder** — combine multiple charts into a single publication-ready multi-panel figure (PNG/SVG, 300–600 DPI)
- 🎨 **Journal color palettes** — Nature, Science, Cell, ACS, Colorblind Safe
- 🤖 **Smart column detection** — auto-detects numeric, categorical, datetime columns
- 🎯 **Domain-aware** — Biology & Genomics, Chemistry & Materials, Medicine & Clinical, Physics & Engineering, Social Science
- 💻 **Copy-paste Python code** for every chart
- ⚡ **Lazy loading** — generate charts by category, not all at once
- ⭐ **Favorites** — star charts you like, filter to favorites only
- 🚀 **Zero-config** — works with any tabular data, no setup needed

### 🗂️ Chart Types

| Category | Count | Examples |
|----------|-------|---------|
| Distribution | 11 | Histogram, Violin, Ridgeline, Raincloud, ECDF |
| Comparison | 15 | Bar, Lollipop, Dumbbell, Slope, Tornado, Radial Bar |
| Correlation | 9 | Scatter, Hexbin, 2D KDE, Pair Plot, Parallel Coords |
| Time Series | 9 | Line, Streamgraph, Bump Chart, Calendar Heatmap, Candlestick |
| Proportional | 8 | Treemap, Sunburst, Waffle, Marimekko, Circle Packing |
| Network | 6 | Sankey, Chord, Arc Diagram, Alluvial, Network Graph |
| Scientific | 15 | Volcano, PCA, UMAP, t-SNE, ROC, Kaplan-Meier, Manhattan, Forest |
| Text | 2 | Word Cloud, Venn Diagram |
| Geographic | 2 | Choropleth Map, Bubble Map |
| 3D | 3 | 3D Scatter, 3D Surface, 3D Bar |
| **Total** | **80+** | |

### 🚀 Quick Start

```bash
git clone https://github.com/Yang1Bai/SciVizKit.git
cd SciVizKit
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501` 🎉

### ☁️ Deploy to Streamlit Cloud (Free)

1. Fork this repo on GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select `SciVizKit` → branch `main` → file `app.py`
4. Click **Deploy** — live in ~3 minutes

### 📂 Project Structure

```
SciVizKit/
├── app.py                     # Main Streamlit app
├── requirements.txt
├── .streamlit/config.toml     # Theme config
├── src/
│   ├── data_analyzer.py       # Column type detection
│   ├── chart_registry.py      # Registry of 80+ chart types
│   ├── decision_tree.py       # Chart recommendation tree
│   ├── figure_panel.py        # Multi-panel figure builder
│   ├── themes/palettes.py     # Journal color palettes
│   └── generators/            # Chart generation modules
│       ├── distribution.py
│       ├── comparison.py
│       ├── correlation.py
│       ├── timeseries.py
│       ├── proportional.py
│       ├── network.py
│       ├── scientific.py
│       ├── text_viz.py
│       ├── geo_viz.py
│       └── threed_viz.py
└── examples/
    ├── sample_general.csv
    ├── sample_biology.csv
    └── sample_chemistry.csv
```

### 🤝 Contributing

1. Fork → create branch `feature/your-chart`
2. Add chart metadata to `src/chart_registry.py`
3. Implement generator in the appropriate `src/generators/*.py`
4. Submit a pull request

### 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

<a name="中文"></a>
## 🇨🇳 中文

### 什么是 SciVizKit？

SciVizKit 是一个基于 Streamlit 的开源科研可视化工具。上传任意数据集（CSV 或 Excel），自动生成 **80+ 种图表**，覆盖 10 大类型——从基础统计图到领域专用科研图。选出最适合论文的方案，下载发表级图片（300 DPI），并一键复制 Python 代码。

### ✨ 核心功能

- 📊 **80+ 种图表**，覆盖 10 大类（分布/比较/相关性/时间序列/比例/网络/科研专用/文本/地理/3D）
- 🌳 **图表引导** — 交互式决策树，根据你的数据和目标推荐最佳图表
- 🖼️ **多图面板** — 把多张图拼成一张发表级多面板图（PNG/SVG，300–600 DPI）
- 🎨 **期刊配色** — Nature、Science、Cell、ACS、色盲友好
- 🤖 **智能列检测** — 自动识别数值、分类、时间列
- 🎯 **领域专用** — 生物基因组、化学材料、医学临床、物理工程、社会科学
- 💻 **每张图附带可复制的 Python 代码**
- ⚡ **懒加载** — 按类别分批生成，秒级响应
- ⭐ **收藏功能** — 标记喜欢的图，一键过滤
- 🚀 **零配置** — 任何表格数据直接用，无需额外设置

### 🗂️ 图表类型

| 类别 | 数量 | 代表图表 |
|------|------|---------|
| 分布 Distribution | 11 | 直方图、提琴图、Ridgeline、雨云图、ECDF |
| 比较 Comparison | 15 | 条形图、棒棒糖、哑铃图、斜坡图、龙卷风图 |
| 相关性 Correlation | 9 | 散点图、六边形密度、2D KDE、平行坐标 |
| 时间序列 Time Series | 9 | 折线图、流图、排名图、日历热力图、K线图 |
| 比例 Proportional | 8 | 树图、旭日图、华夫饼图、马赛克图、圆形填充 |
| 网络 Network | 6 | Sankey、弦图、弧线图、冲积图、网络图 |
| 科研专用 Scientific | 15 | 火山图、PCA、UMAP、t-SNE、ROC、生存曲线、曼哈顿图 |
| 文本 Text | 2 | 词云、韦恩图 |
| 地理 Geographic | 2 | 等值线地图、气泡地图 |
| 3D | 3 | 3D散点、3D曲面、3D柱图 |
| **合计** | **80+** | |

### 🚀 本地运行

```bash
git clone https://github.com/Yang1Bai/SciVizKit.git
cd SciVizKit
pip install -r requirements.txt
streamlit run app.py
```

浏览器访问 `http://localhost:8501` 即可使用 🎉

### ☁️ 部署到 Streamlit Cloud（免费）

1. Fork 本仓库到你的 GitHub 账号
2. 访问 [share.streamlit.io](https://share.streamlit.io)，用 GitHub 账号登录
3. 点击 **"New app"** → 选择仓库 `SciVizKit` → 分支 `main` → 入口文件 `app.py`
4. 点击 **Deploy** — 约 3 分钟后获得公开链接

### 📂 项目结构

```
SciVizKit/
├── app.py                     # 主应用
├── requirements.txt           # 依赖
├── .streamlit/config.toml     # 主题配置
├── src/
│   ├── data_analyzer.py       # 数据列类型检测
│   ├── chart_registry.py      # 80+ 图表注册表
│   ├── decision_tree.py       # 图表推荐决策树
│   ├── figure_panel.py        # 多图面板构建器
│   ├── themes/palettes.py     # 期刊配色方案
│   └── generators/            # 各类图表生成模块
└── examples/                  # 示例数据集
```

### 🤝 贡献指南

1. Fork 本仓库，创建分支 `feature/你的图表名`
2. 在 `src/chart_registry.py` 添加图表元数据
3. 在对应的 `src/generators/*.py` 实现生成函数
4. 提交 Pull Request

### 📝 开源协议

MIT License，详见 [LICENSE](LICENSE)。

---

*为科研人员、数据科学家和所有热爱好图表的人而生。*
*Made for researchers, data scientists, and anyone who loves great charts.*
