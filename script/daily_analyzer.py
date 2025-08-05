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
    """Generate a standalone HTML report for today."""
    top_topics = df['topic_name'].value_counts().head(6)
    output_path = f"docs/daily-{TODAY}.html"

    # Start HTML content
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Daily News Analysis - {TODAY}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 20px; }}
        h1, h2 {{ color: #1a1a1a; }}
        iframe {{ width: 100%; height: 500px; border: 1px solid #ddd; margin: 20px 0; }}
        .meta {{ color: #666; font-size: 0.9em; margin-bottom: 20px; }}
        .top-topic {{ background: #f8f9fa; padding: 10px 15px; border-left: 4px solid #0056b3; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🌍 Daily News Analysis - {TODAY}</h1>
    <div class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC | Articles: {len(df)}</div>

    <h2>🔝 Top Topics Today</h2>
    <ul>
"""
    for topic, count in top_topics.items():
        html += f"<li><strong>{topic}</strong>: {count} articles</li>\n"

    html += f"""
    </ul>

    <h2>📊 Topic Evolution Over Time</h2>
    <iframe src="assets/{TODAY}/topic_evolution.html" title="Topic Evolution"></iframe>

    <h2>🔗 Topic Co-occurrence Matrix</h2>
    <p><img src="assets/{TODAY}/cooccurrence.png" alt="Co-occurrence" style="max-width:100%;"></p>

    <h2>🧠 Semantic Similarity</h2>
    <p><img src="assets/{TODAY}/semantic_similarity.png" alt="Semantic Similarity" style="max-width:100%;"></p>

    <hr>
    <p><a href="index.html"> <<<  Back to all days</a></p>
</body>
</html>
    """

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Daily HTML report saved: {output_path}")


def generate_index_html():
    """Generate a modern, user-friendly index.html with a clean layout and search functionality."""
    files = sorted(glob.glob("docs/daily-*.html"), reverse=True)
    
    if not files:
        print("[!] No daily reports found to list in index.")
        return

    # Build the list of reports
    report_items = ""
    for file in files:
        date_str = os.path.basename(file).replace("daily-", "").replace(".html", "")
        # Convert YYYY-MM-DD to readable format: Aug 3, 2025
        try:
            display_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%b %d, %Y")
        except:
            display_date = date_str

        report_items += f'''
        <li class="report-item">
            <a href="{os.path.basename(file)}" class="report-link">
                <span class="report-title">Daily Intelligence Report</span>
                <span class="report-date">{display_date}</span>
            </a>
        </li>
        '''

    # Full HTML with CSS and optional search
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Event Context — News Intelligence Archive</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg: #f8f9fa;
            --text: #212529;
            --accent: #0056b3;
            --border: #dee2e6;
            --card-bg: #ffffff;
            --hover: #e9ecef;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
            padding: 20px;
        }}
        header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        header h1 {{
            font-size: 2.2em;
            color: var(--accent);
            margin: 0;
        }}
        header p {{
            font-size: 1.1em;
            color: #495057;
            margin-top: 10px;
        }}
        .search-box {{
            margin-bottom: 30px;
            text-align: center;
        }}
        #searchInput {{
            padding: 12px 20px;
            width: 80%;
            max-width: 500px;
            border: 2px solid var(--border);
            border-radius: 50px;
            font-size: 1em;
            outline: none;
            transition: all 0.3s;
        }}
        #searchInput:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 10px rgba(0, 86, 179, 0.1);
        }}
        .reports-list {{
            list-style: none;
            padding: 0;
        }}
        .report-item {{
            margin-bottom: 12px;
        }}
        .report-link {{
            display: block;
            padding: 18px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            text-decoration: none;
            color: var(--text);
            transition: all 0.3s ease;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
        }}
        .report-link:hover {{
            border-color: var(--accent);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
            background: var(--hover);
        }}
        .report-title {{
            font-size: 1.1em;
        }}
        .report-date {{
            font-size: 0.95em;
            color: #6c757d;
            white-space: nowrap;
            margin-left: 15px;
        }}
        footer {{
            text-align: center;
            margin-top: 50px;
            color: #6c757d;
            font-size: 0.9em;
        }}
        @media (max-width: 600px) {{
            .report-link {{
                flex-direction: column;
                text-align: center;
            }}
            .report-date {{
                margin-top: 8px;
                font-size: 0.9em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-globe-americas"></i> Event Context</h1>
            <p><em>(Evolving Vectorized Entities in News Texts - CONtextual
            Topic EXtraction Technique)</em></p>
        </header>

        <ul class="reports-list">
            {report_items}
        </ul>

        <footer>
            <p>Generated automatically with EVENT-CONTEXT | <a href="https://github.com/jluisvim/event-context" target="_blank">View on GitHub</a></p>
        </footer>
    </div>
</body>
</html>'''

    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("[+] Modern index.html generated with search and responsive design.")


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
