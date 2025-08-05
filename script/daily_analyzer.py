"""
📊 Topic Relations Analyzer - Daily HTML Version
------------------------------------------------
Author: JLuisVM
Date: 2025-08-03

This script:
1. Fetches news from RSS feeds.
2. Classifies them into topics using BERTopic.
3. Analyzes co-occurrence and semantic similarity.
4. Generates interactive visualizations.
5. Builds a daily HTML report and index page in `docs/`.
6. Output is ready for GitHub Pages.

All results are saved by date for historical tracking.
"""

import feedparser
import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import os
import glob
import shutil


# ===================================
# 1. CONFIGURATION
# ===================================

TODAY = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "data/daily"
RESULTS_DIR = f"results/daily/{TODAY}"
DOCS_DIR = "docs"
FEEDS_FILE = "feeds/sources.txt"

# Create required directories
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def copy_results_to_docs(TODAY):
    """Copia los gráficos y HTML de resultados a docs/assets/ para que GitHub Pages los sirva."""
    os.makedirs(f"docs/assets/{TODAY}", exist_ok=True)

    # Archivos a copiar
    files_to_copy = [
        f"results/daily/{TODAY}/topic_evolution.html",
        f"results/daily/{TODAY}/cooccurrence.png",
        f"results/daily/{TODAY}/semantic_similarity.png"
    ]

    for src in files_to_copy:
        if os.path.exists(src):
            dst = f"docs/assets/{TODAY}/" + os.path.basename(src)
            shutil.copy(src, dst)
            print(f"[+] Copiado: {src} → {dst}")
        else:
            print(f"[!] No encontrado: {src}")

# ===================================
# 2. LOAD RSS SOURCES
# ===================================

def load_sources(filename=FEEDS_FILE):
    """Load list of RSS feed URLs from file."""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            sources = [line.strip() for line in file if line.strip() and not line.startswith("#")]
        print(f"[+] Loaded {len(sources)} RSS sources.")
        return sources
    except FileNotFoundError:
        print(f"[!] Feed file not found: {filename}")
        return []


# ===================================
# 3. FETCH NEWS FROM RSS
# ===================================

def fetch_feed(url, limit=50):
    """Fetch news titles and dates from an RSS feed."""
    try:
        feed = feedparser.parse(url)
        titles = []
        dates = []
        for entry in feed.entries[:limit]:
            title = getattr(entry, 'title', '').strip()
            pub_date = getattr(entry, 'published_parsed', None)
            if title:
                titles.append(title)
                if pub_date:
                    dates.append(datetime(*pub_date[:6]))
                else:
                    dates.append(datetime.now())
        return titles, dates
    except Exception as e:
        print(f"[!] Error fetching {url}: {e}")
        return [], []


# ===================================
# 4. TOPIC MODELING WITH BERTOPIC
# ===================================

def run_topic_modeling(titles):
    """Apply BERTopic to classify news into topics."""
    if not titles:
        return None, []

    print(f"[+] Running BERTopic on {len(titles)} articles...")

    # Vectorization
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))

    # BERTopic model
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=True,
        nr_topics="auto",
        vectorizer_model=vectorizer,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(titles)
    return topic_model, topics


# ===================================
# 5. TOPIC CO-OCCURRENCE MATRIX
# ===================================

def build_cooccurrence_matrix(df, window_size=10):
    """Build co-occurrence matrix based on temporal proximity."""
    unique_topics = sorted(df["topic"].unique())
    matrix = pd.DataFrame(0, index=unique_topics, columns=unique_topics)
    for i in range(0, len(df), window_size):
        window = df.iloc[i:i+window_size]
        topics_in_window = window["topic"].unique()
        for t1 in topics_in_window:
            for t2 in topics_in_window:
                if t1 != t2:
                    matrix.loc[t1, t2] += 1
    return matrix


# ===================================
# 6. SEMANTIC SIMILARITY BETWEEN TOPICS
# ===================================

def compute_semantic_similarity(topic_names):
    """Compute semantic similarity using sentence embeddings."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(topic_names)
    sim_matrix = cosine_similarity(embeddings)
    return pd.DataFrame(sim_matrix, index=topic_names, columns=topic_names)


# ===================================
# 7. VISUALIZATIONS
# ===================================

def plot_topic_evolution(grouped_df, topic_labels, today):
    """Plot topic evolution over time."""
    grouped_df.index = grouped_df.index.astype(str)
    renamed = grouped_df.rename(columns=topic_labels)
    fig = px.line(
        renamed.reset_index(),
        x='period',
        y=renamed.columns[1:],
        title="Topic Evolution Over Time",
        labels={"value": "Number of News", "variable": "Topic"}
    )
    output_path = f"results/daily/{today}/topic_evolution.html"
    fig.write_html(output_path)
    print(f"[+] Topic evolution saved to {output_path}")


def plot_cooccurrence(matrix, today):
    """Save co-occurrence heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt="g")
    plt.title("Topic Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(f"results/daily/{today}/cooccurrence.png", dpi=150)
    plt.close()


def plot_semantic_similarity(sim_df, today):
    """Save semantic similarity heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Semantic Similarity Between Topics")
    plt.tight_layout()
    plt.savefig(f"results/daily/{today}/semantic_similarity.png", dpi=150)
    plt.close()


# ===================================
# 8. GENERATE HTML REPORTS
# ===================================

def generate_daily_html(df, topic_model, topic_evolution, topic_labels, TODAY):
    """Generate a modern, responsive HTML report for today."""
    top_topics = df['topic_name'].value_counts().head(6)
    output_path = f"docs/daily-{TODAY}.html"

    # Generar tarjetas de temas
    topic_cards = ""
    for topic, count in top_topics.items():
        topic_cards += f"""
        <div class="card">
            <h3>{topic}</h3>
            <p class="count">{count} articles</p>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Daily News Analysis - {TODAY}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #ffffff;
            --text: #1a1a1a;
            --accent: #0056b3;
            --light: #f8f9fa;
            --border: #e9ecef;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            line-height: 1.7;
            color: var(--text);
            background: var(--bg);
            margin: 0;
            padding: 0;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #0056b3 0%, #003d82 100%);
            color: white;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0;
            font-weight: 600;
            font-size: 2.2em;
        }}
        .meta {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-top: 8px;
        }}
        .top-topics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}
        .card {{
            background: var(--light);
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid var(--accent);
            transition: transform 0.2s;
        }}
        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.05);
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            font-size: 1.2em;
            color: var(--accent);
        }}
        .card .count {{
            margin: 0;
            font-weight: 500;
            color: #495057;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            background: #fbfbfb;
            border-radius: 10px;
            border: 1px solid var(--border);
        }}
        .section h2 {{
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 8px;
            margin-top: 0;
        }}
        iframe {{
            width: 100%;
            height: 500px;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 15px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }}
        @media (max-width: 768px) {{
            .top-topics {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>🌍 Daily News Analysis - {TODAY}</h1>
        <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | Articles: {len(df)}</div>
    </header>

    <h2>🔝 Top Topics Today</h2>
    <div class="top-topics">
        {topic_cards}
    </div>

    <div class="section">
        <h2>📊 Topic Evolution Over Time</h2>
        <iframe src="assets/{TODAY}/topic_evolution.html" title="Topic Evolution"></iframe>
    </div>

    <div class="section">
        <h2>🔗 Topic Co-occurrence Matrix</h2>
        <img src="assets/{TODAY}/cooccurrence.png" alt="Co-occurrence Matrix">
    </div>

    <div class="section">
        <h2>🧠 Semantic Similarity</h2>
        <img src="assets/{TODAY}/semantic_similarity.png" alt="Semantic Similarity">
    </div>

    <footer>
        <p><a href="index.html">← Back to all reports</a></p>
        <p>Event Context | AI-Powered News Intelligence</p>
    </footer>
</body>
</html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Daily HTML report saved: {output_path}")


def generate_index_html():
    """Generate a modern index.html with archive of daily reports."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Event Context - Archive</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #ffffff;
            --text: #1a1a1a;
            --accent: #0056b3;
            --light: #f8f9fa;
            --border: #e9ecef;
        }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            line-height: 1.7;
            color: var(--text);
            background: var(--bg);
            margin: 0;
            padding: 0;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        h1 {{
            color: var(--accent);
            margin: 0;
            font-size: 2.4em;
            font-weight: 600;
        }}
        .subtitle {{
            font-size: 1.1em;
            color: #6c757d;
            margin: 10px 0 0 0;
        }}
        .reports {{
            list-style: none;
            padding: 0;
        }}
        .reports li {{
            margin: 15px 0;
            padding: 15px;
            background: var(--light);
            border-radius: 10px;
            border-left: 4px solid var(--accent);
            transition: background 0.2s;
        }}
        .reports li:hover {{
            background: #f1f3f5;
        }}
        .reports a {{
            text-decoration: none;
            font-size: 1.1em;
            color: var(--text);
            font-weight: 500;
        }}
        .date {{
            color: #6c757d;
            font-size: 0.9em;
            margin-left: 10px;
        }}
        footer {{
            text-align: center;
            margin-top: 60px;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <header>
        <h1>📰 Event Context</h1>
        <p class="subtitle">Daily AI-Powered News Intelligence</p>
    </header>

    <h2>📋 Archive of Daily Reports</h2>
    <ul class="reports">
"""

    # Listar todos los informes
    files = sorted(glob.glob("docs/daily-*.html"), reverse=True)
    for file in files:
        date_str = os.path.basename(file).replace("daily-", "").replace(".html", "")
        html += f'        <li><a href="{os.path.basename(file)}">Daily Report - {date_str}</a> <span class="date">({date_str})</span></li>\n'

    html += """    </ul>

    <footer>
        <p>Automated analysis using BERTopic and RSS feeds.</p>
        <p>Generated with ❤️ by Event Context</p>
    </footer>
</body>
</html>"""

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("[+] index.html generated with modern design.")


# ===================================
# 9. MAIN FUNCTION
# ===================================

def main():
    print(f"\n[+] Starting daily analysis for {TODAY}...")

    # 1. Load sources
    sources = load_sources()
    if not sources:
        print("[!] No sources loaded. Exiting.")
        return

    # 2. Fetch news
    all_titles, all_dates = [], []
    for source in sources:
        titles, dates = fetch_feed(source)
        all_titles.extend(titles)
        all_dates.extend(dates)
        print(f"[+] Fetched {len(titles)} articles from {source}")

    if not all_titles:
        print("[!] No news collected. Exiting.")
        return

    # 3. Create DataFrame
    df = pd.DataFrame({"title": all_titles, "date": all_dates})
    df.to_csv(f"{DATA_DIR}/{TODAY}.csv", index=False)

    # 4. Run topic modeling
    topic_model, topics = run_topic_modeling(all_titles)
    if topic_model is None:
        print("[!] Topic modeling failed. Exiting.")
        return

    df["topic"] = topics
    df["topic_name"] = [topic_model.topic_labels_[t] for t in topics]

    # 5. Time-based grouping
    df['period'] = pd.to_datetime(df['date']).dt.to_period('D')
    topic_evolution = pd.crosstab(df['period'], df['topic_name'])

    # 6. Save results
    df.to_csv(f"{RESULTS_DIR}/news_with_topics.csv", index=False)

    # 7. Co-occurrence
    cooc_matrix = build_cooccurrence_matrix(df)
    plot_cooccurrence(cooc_matrix, TODAY)

    # 8. Semantic similarity
    topic_names = [topic_model.topic_labels_[t] for t in sorted(df["topic"].unique())]
    sim_df = compute_semantic_similarity(topic_names)
    plot_semantic_similarity(sim_df, TODAY)

    # 9. Topic evolution plot
    plot_topic_evolution(topic_evolution, topic_model.topic_labels_, TODAY)

    # 10. Copy results to docs for web access
    copy_results_to_docs(TODAY)

    # 11. Generate HTML reports
    print("[+] Generating HTML reports in docs/...")
    generate_daily_html(df, topic_model, topic_evolution, topic_model.topic_labels_, TODAY)
    generate_index_html()

    print(f"[✓] Daily analysis completed for {TODAY}.\n")


if __name__ == "__main__":
    main()
