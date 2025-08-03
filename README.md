# 🌍 Event Context: Global News Intelligence Engine

**Event Context** is an automated news analysis system that collects global news from RSS feeds, classifies them into topics using AI (BERTopic), and generates daily reports with insights on topic trends, co-occurrence, and semantic relationships.

Perfect for researchers, analysts, and globally-minded citizens who want to understand the world through data-driven news intelligence.

---

## 🚀 Features

- ✅ **Daily news collection** from global RSS sources
- 🔍 **Topic modeling** using BERTopic (NLP)
- 📊 **Topic co-occurrence & semantic similarity analysis**
- 📈 **Interactive visualizations**: evolution over time, heatmaps, network graphs
- 📅 **Automated daily reports** with historical tracking
- 🤖 **Fully automated** via GitHub Actions (runs every 24h)
- 💾 **Persistent history** of news and analysis by date
- 🌐 **Blog-ready output** (Markdown posts for Hugo or any static site)

---

## 📋 How It Works

1. **Collect**: Fetches news titles from RSS feeds (e.g., BBC, Reuters, NYT).
2. **Analyze**: Uses BERTopic to classify news into dynamic topics.
3. **Relate**: Builds co-occurrence and semantic similarity matrices.
4. **Visualize**: Generates interactive plots and static graphs.
5. **Report**: Saves results and creates a daily blog post.
6. **Automate**: Runs daily via GitHub Actions and opens a pull request.

> All results are saved by date in `data/daily/` and `results/daily/`.

---

## 🧩 Tech Stack

| Component | Technology |
|--------|------------|
| News Collection | RSS + `feedparser` |
| Topic Modeling | `BERTopic` + `sentence-transformers` |
| NLP | `scikit-learn`, `spaCy` (via BERTopic) |
| Visualization | `plotly`, `seaborn`, `matplotlib`, `networkx` |
| Automation | GitHub Actions |
| Blog Output | Markdown (compatible with Hugo, Jekyll, etc.) |
| Hosting | GitHub Pages (optional) |

---

## 🛠️ Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/jluisvim/event-context.git
cd event-context
