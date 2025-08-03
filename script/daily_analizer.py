"""
Topic Relations Analyzer - Daily Version
Collects news, analyzes topics, and saves daily report for Hugo blog.
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
import shutil

# Config
TODAY = datetime.now().strftime("%Y-%m-%d")
DATA_DIR = "data/daily"
RESULTS_DIR = f"results/daily/{TODAY}"
BLOG_POSTS_DIR = "blog/content/posts"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BLOG_POSTS_DIR, exist_ok=True)

def load_sources(filename="feeds/sources.txt"):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

def fetch_feed(url, limit=50):
    feed = feedparser.parse(url)
    titles = []
    dates = []
    for entry in feed.entries[:limit]:
        title = getattr(entry, 'title', '')
        pub_date = getattr(entry, 'published_parsed', None)
        if title:
            titles.append(title)
            dates.append(datetime(*pub_date[:6]) if pub_date else datetime.now())
    return titles, dates

def classify_by_period(df, period='day'):
    df['date'] = pd.to_datetime(df['date'])
    df['period'] = df['date'].dt.to_period('D')
    return df

def build_cooccurrence_matrix(df, window_size=10):
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

def compute_semantic_similarity(topic_names):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(topic_names)
    sim_matrix = cosine_similarity(embeddings)
    return pd.DataFrame(sim_matrix, index=topic_names, columns=topic_names)

def plot_topic_evolution(grouped_df, topic_labels):
    grouped_df.index = grouped_df.index.astype(str)
    renamed = grouped_df.rename(columns=topic_labels)
    fig = px.line(
        renamed.reset_index(),
        x='period',
        y=renamed.columns[1:],
        title="Topic Evolution Over Time",
        labels={"value": "Number of News", "variable": "Topic"}
    )
    fig.write_html(f"{RESULTS_DIR}/topic_evolution.html")
    print(f"[+] Topic evolution saved to {RESULTS_DIR}/topic_evolution.html")

def main():
    print(f"[+] Starting daily news analysis for {TODAY}...")

    # 1. Fetch news
    sources = load_sources()
    all_titles, all_dates = [], []
    for source in sources:
        try:
            titles, dates = fetch_feed(source)
            all_titles.extend(titles)
            all_dates.extend(dates)
            print(f"[+] Fetched {len(titles)} from {source}")
        except Exception as e:
            print(f"[!] Error fetching {source}: {e}")

    if len(all_titles) == 0:
        print("[!] No news collected. Exiting.")
        return

    df = pd.DataFrame({"title": all_titles, "date": all_dates})
    df.to_csv(f"{DATA_DIR}/{TODAY}.csv", index=False)

    # 2. Topic modeling
    vectorizer_model = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    topic_model = BERTopic(
        language="en",
        calculate_probabilities=True,
        nr_topics="auto",
        vectorizer_model=vectorizer_model,
        verbose=False
    )

    topics, probs = topic_model.fit_transform(df["title"])
    df["topic"] = topics
    df["topic_name"] = [topic_model.topic_labels_[t] for t in topics]

    # 3. Time classification
    df = classify_by_period(df, period='day')
    topic_evolution = pd.crosstab(df['period'], df['topic_name'])

    # 4. Save analysis
    df.to_csv(f"{RESULTS_DIR}/news_with_topics.csv", index=False)

    # Co-occurrence
    cooc_matrix = build_cooccurrence_matrix(df)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cooc_matrix, annot=True, cmap="YlGnBu", fmt="g")
    plt.title(f"Topic Co-occurrence ({TODAY})")
    plt.savefig(f"{RESULTS_DIR}/cooccurrence.png")
    plt.close()

    # Semantic similarity
    topic_names = [topic_model.topic_labels_[t] for t in sorted(df["topic"].unique())]
    sim_df = compute_semantic_similarity(topic_names)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Semantic Similarity ({TODAY})")
    plt.savefig(f"{RESULTS_DIR}/semantic_similarity.png")
    plt.close()

    # Topic evolution
    plot_topic_evolution(topic_evolution, topic_model.topic_labels_)

    # 5. Generate Hugo blog post (Markdown)
    top_topics = df['topic_name'].value_counts().head(5)
    summary_md = f"""---
title: "Daily News Analysis - {TODAY}"
date: {TODAY}
tags: {list(top_topics.index[:3])}
categories: ["Daily Report"]
author: "News Intelligence Engine"
---

## 🌍 Global News Summary - {TODAY}

Collected **{len(df)} news articles** from international sources.

### 🔝 Top Topics Today
"""
    for topic, count in top_topics.items():
        summary_md += f"- **{topic}**: {count} articles\n"

    summary_md += f"""

### 📊 Visualizations

#### Topic Evolution
{{{{< rawhtml >}}}}
<iframe src="/results/daily/{TODAY}/topic_evolution.html" width="100%" height="500"></iframe>
{{{{< /rawhtml >}}}}

#### Co-occurrence Matrix
![Co-occurrence]({{{{ '/results/daily/{TODAY}/cooccurrence.png' | relative_url }}}})

#### Semantic Similarity
![Similarity]({{{{ '/results/daily/{TODAY}/semantic_similarity.png' | relative_url }}}})

---

*Generated automatically with BERTopic and RSS analysis.*
"""

    with open(f"{BLOG_POSTS_DIR}/daily-{TODAY}.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"[+] Daily report saved to blog: daily-{TODAY}.md")

if __name__ == "__main__":
    main()
