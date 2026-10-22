"""
app.py
======
Interactive Web Application & Fast Front End
Project: GitHub Project Issue Classifier
Author: Priyanshu Yadav (S25CSEU0725) | Batch-25

Serves:
- Real-time Issue Classifier with confidence scoring, probabilities, and routing
- Dataset & Quality Audit Explorer (interactive figures gallery & metrics)
- Architecture & Viva Reference Guide
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scripts.preprocess_dataset import clean_text

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global model and metadata state
MODEL_PIPELINE = None
CLASSES = []
DATASET_STATS = {}
FEATURE_NAMES = []


def initialize_model():
    global MODEL_PIPELINE, CLASSES, DATASET_STATS, FEATURE_NAMES
    supervised_path = "data/processed/github_issues_supervised.parquet"
    if not os.path.exists(supervised_path):
        raise FileNotFoundError(f"Supervised dataset not found at: {supervised_path}")

    print(f"Loading dataset from {supervised_path}...")
    df = pd.read_parquet(supervised_path)

    # Dataset distribution stats
    class_counts = df["target_class"].value_counts().to_dict()
    DATASET_STATS = {
        "raw_records": 7540,
        "deduplicated_records": 7440,
        "pull_requests_filtered": 4325,
        "pure_issues": 3115,
        "supervised_issues": len(df),
        "unlabeled_issues": 1493,
        "conflicting_issues": 34,
        "class_distribution": class_counts,
    }

    # Train sublinear TF-IDF + Logistic Regression
    print("Training baseline TF-IDF + Logistic Regression pipeline...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=42,
    )

    MODEL_PIPELINE = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])

    MODEL_PIPELINE.fit(df["text"], df["target_class"])
    CLASSES = list(MODEL_PIPELINE.classes_)
    FEATURE_NAMES = vectorizer.get_feature_names_out()
    t1 = time.time()
    print(f"Model initialized in {t1 - t0:.2f}s with classes: {CLASSES}")


# Initialize model on startup
initialize_model()


def estimate_priority(text: str, predicted_class: str) -> str:
    """Heuristic priority estimation based on submission-time signals."""
    text_lower = text.lower()
    high_keywords = [
        "crash", "fatal", "segmentation fault", "data loss", "corruption",
        "deadlock", "memory leak", "panic", "broken", "critical", "vulnerability"
    ]
    if any(k in text_lower for k in high_keywords):
        return "HIGH"
    if predicted_class == "DOCUMENTATION":
        return "LOW"
    if predicted_class in ["FEATURE", "QUESTION", "OTHER"]:
        return "MEDIUM"
    return "MEDIUM"


def suggest_routing(text: str, predicted_class: str) -> dict:
    """Suggests an engineering area and team based on issue text and predicted type."""
    text_lower = text.lower()
    if any(k in text_lower for k in ["vision", "image", "pil", "pixel"]):
        area = "Vision & Multimodal"
        team = "team-vision-datasets"
    elif any(k in text_lower for k in ["audio", "speech", "sound", "wav", "torchaudio"]):
        area = "Audio & Speech"
        team = "team-audio-datasets"
    elif any(k in text_lower for k in ["stream", "iterable", "chunk", "yield"]):
        area = "Streaming & Iterators"
        team = "team-core-engine"
    elif any(k in text_lower for k in ["hub", "push_to_hub", "hf_hub", "auth", "token"]):
        area = "HuggingFace Hub Integration"
        team = "team-platform-integrations"
    elif predicted_class == "DOCUMENTATION":
        area = "Docs & Tutorials"
        team = "team-devrel-docs"
    elif predicted_class == "BUG":
        area = "Core Framework"
        team = "team-core-triage"
    elif predicted_class == "FEATURE":
        area = "Dataset Integrations"
        team = "team-feature-leads"
    else:
        area = "General Triage"
        team = "team-community-triage"

    return {"area": area, "team": team}


def extract_top_keywords(text: str, top_n: int = 5) -> list:
    """Extracts top TF-IDF unigrams/bigrams from the input text."""
    if not text.strip():
        return []
    tfidf = MODEL_PIPELINE.named_steps["tfidf"]
    vec = tfidf.transform([text])
    feature_indices = vec.nonzero()[1]
    scores = [(FEATURE_NAMES[idx], vec[0, idx]) for idx in feature_indices]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [term for term, _ in scores[:top_n]]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        title = data.get("title", "").strip()
        body = data.get("body", "").strip()
        author_association = data.get("author_association", "NONE")

        if not title:
            return jsonify({"error": "Issue title is required."}), 400

        # Preprocess and combine text
        clean_title = clean_text(title)
        clean_body = clean_text(body)
        combined_text = (clean_title + "\n\n" + clean_body).strip() if clean_body else clean_title

        # Inference
        probs = MODEL_PIPELINE.predict_proba([combined_text])[0]
        pred_idx = np.argmax(probs)
        predicted_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        # Probabilities dictionary
        prob_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}

        # Secondary estimations
        priority = estimate_priority(combined_text, predicted_class)
        routing = suggest_routing(combined_text, predicted_class)
        keywords = extract_top_keywords(combined_text, top_n=6)

        return jsonify({
            "status": "success",
            "prediction": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probabilities": prob_dict,
            "priority": priority,
            "routing_area": routing["area"],
            "routing_team": routing["team"],
            "keywords": keywords,
            "stats": {
                "title_length": len(clean_title),
                "body_length": len(clean_body),
                "word_count": len(combined_text.split()),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats")
def stats():
    return jsonify(DATASET_STATS)


@app.route("/api/sample-issues")
def sample_issues():
    samples = [
        {
            "id": "crash_bug",
            "name": "Fatal Memory Crash Bug",
            "title": "Fatal memory leak and segmentation fault in arrow batch streaming",
            "body": "When calling load_dataset() with streaming=True and iterating through 50,000 samples, memory grows unboundedly until Python crashes with a segmentation fault. Traceback:\n  File 'datasets/arrow_dataset.py', line 1402, in _get_batch\nFatal: Memory allocation failed.",
            "author_association": "NONE"
        },
        {
            "id": "feature_req",
            "name": "Feature Request",
            "title": "Add support for WebDataset tar archive streaming with shards",
            "body": "It would be wonderful to add native support for WebDataset formatted shards directly in load_dataset. Currently users have to unpack tar files to disk first. Adding an on-the-fly streaming reader would significantly accelerate multimodal training pipelines.",
            "author_association": "CONTRIBUTOR"
        },
        {
            "id": "doc_typo",
            "name": "Documentation Clarification",
            "title": "Typo and broken link in Quickstart tutorial code example",
            "body": "In docs/source/quickstart.md line 42, the code snippet references 'dataset.map_batches()' which does not exist. It should be 'dataset.map(batched=True)'. Also the tutorial hyperlink to audio datasets points to a 404 error page.",
            "author_association": "NONE"
        },
        {
            "id": "usage_question",
            "name": "Usage Question",
            "title": "How to properly configure custom cache directory across distributed PyTorch nodes?",
            "body": "I am training a model using PyTorch DDP on 4 nodes. How can I ensure that all worker processes share a single cache directory without race conditions when writing cached arrow files? Is setting HF_DATASETS_CACHE sufficient?",
            "author_association": "NONE"
        },
        {
            "id": "refactor_task",
            "name": "Internal Refactoring / Maintenance",
            "title": "Refactor deprecated numpy array casting in tensor conversion helpers",
            "body": "Clean up deprecated np.matrix and array conversion calls in datasets/features/tensor.py to maintain compatibility with upcoming NumPy 2.0 release. All unit tests must be updated accordingly.",
            "author_association": "MEMBER"
        }
    ]
    return jsonify(samples)


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename):
    fig_dir = os.path.abspath("reports/figures")
    return send_from_directory(fig_dir, filename)


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print(" GitHub Project Issue Classifier - Web Dashboard")
    print(" Running at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
