"""
app.py
======
Milestone 2 Interactive Web Application & Live GitHub Issue Triage Explorer
Project: GitHub Project Issue Classifier
Author: Priyanshu Yadav (S25CSEU0725) | Batch-25

Features:
- Live GitHub Issue URL Ingestion (Local verified cache + Live GitHub REST API)
- Milestone 2 Data Processing Inspector (PR filtering, text cleaning, leakage firewall)
- Real-time Baseline Classifier (TF-IDF + Logistic Regression with cost-sensitive weights)
- Milestone 2 EDA Gallery (11 Publication-grade figures + written interpretations)
- Formal Deliverables & Viva Download Center (PDF report, PPTX presentation, Audit JSON)
"""

import os
import re
import json
import time
import urllib.request
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from scripts.preprocess_dataset import (
    clean_text,
    extract_label_names,
    resolve_target_label,
    TYPE_MAPPING,
    EXCLUDED_LEAKAGE_FIELDS,
    FEATURE_ALLOWLIST,
)

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global state
MODEL_PIPELINE = None
CLASSES = []
FEATURE_NAMES = []
LOCAL_ISSUES_MAP = {}
RAW_ISSUES_MAP = {}
SUPERVISED_DF = None


def initialize_app_data():
    global MODEL_PIPELINE, CLASSES, FEATURE_NAMES, LOCAL_ISSUES_MAP, RAW_ISSUES_MAP, SUPERVISED_DF

    supervised_path = "data/processed/github_issues_supervised.parquet"
    cleaned_path = "data/processed/github_issues_cleaned.parquet"
    raw_path = "data/raw/github_issues_raw.parquet"

    print("Initializing Milestone 2 Model and Datasets...")
    if os.path.exists(supervised_path):
        SUPERVISED_DF = pd.read_parquet(supervised_path)
    else:
        raise FileNotFoundError(f"Supervised dataset not found at: {supervised_path}")

    # Build URL lookup maps for fast 0ms resolution of verified dataset issues
    if os.path.exists(cleaned_path):
        df_clean = pd.read_parquet(cleaned_path)
        for _, r in df_clean.iterrows():
            url_key = str(r["html_url"]).strip().lower()
            LOCAL_ISSUES_MAP[url_key] = r.to_dict()
            num_key = f"huggingface/datasets#{r['number']}"
            LOCAL_ISSUES_MAP[num_key] = r.to_dict()

    if os.path.exists(raw_path):
        df_raw = pd.read_parquet(raw_path)
        for _, r in df_raw.iterrows():
            url_key = str(r["html_url"]).strip().lower()
            RAW_ISSUES_MAP[url_key] = r.to_dict()

    # Train baseline TF-IDF + Logistic Regression
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
    MODEL_PIPELINE.fit(SUPERVISED_DF["text"], SUPERVISED_DF["target_class"])
    CLASSES = list(MODEL_PIPELINE.classes_)
    FEATURE_NAMES = vectorizer.get_feature_names_out()
    t1 = time.time()
    print(f"Model initialized in {t1 - t0:.2f}s with classes: {CLASSES}")


initialize_app_data()


def estimate_priority(text: str, predicted_class: str) -> str:
    text_lower = text.lower()
    high_keywords = [
        "crash", "fatal", "segmentation fault", "data loss", "corruption",
        "deadlock", "memory leak", "panic", "broken", "critical", "vulnerability"
    ]
    if any(k in text_lower for k in high_keywords):
        return "HIGH"
    if predicted_class == "DOCUMENTATION":
        return "LOW"
    return "MEDIUM"


def suggest_routing(text: str, predicted_class: str) -> dict:
    text_lower = text.lower()
    if any(k in text_lower for k in ["vision", "image", "pil", "pixel"]):
        return {"area": "Vision & Multimodal", "team": "team-vision-datasets"}
    if any(k in text_lower for k in ["audio", "speech", "sound", "wav", "torchaudio"]):
        return {"area": "Audio & Speech", "team": "team-audio-datasets"}
    if any(k in text_lower for k in ["stream", "iterable", "chunk", "yield"]):
        return {"area": "Streaming & Iterators", "team": "team-core-engine"}
    if any(k in text_lower for k in ["hub", "push_to_hub", "hf_hub", "auth", "token"]):
        return {"area": "HuggingFace Hub Integration", "team": "team-platform-integrations"}
    if predicted_class == "DOCUMENTATION":
        return {"area": "Documentation & Tutorials", "team": "team-devrel-docs"}
    if predicted_class == "BUG":
        return {"area": "Core Engine & Defect Triage", "team": "team-core-triage"}
    if predicted_class == "FEATURE":
        return {"area": "Features & Dataset Extensions", "team": "team-feature-leads"}
    return {"area": "Community Triage", "team": "team-community-triage"}


def extract_top_keywords(text: str, top_n: int = 6) -> list:
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


@app.route("/api/fetch-github-issue", methods=["POST"])
def fetch_github_issue():
    """
    Fetches a live GitHub issue via URL:
    1. Checks local verified cache for instant 0ms retrieval
    2. Or queries live GitHub REST API for any external repository
    3. Detects and reports Pull Requests per Milestone 2 filtering protocol
    """
    try:
        data = request.get_json(force=True)
        raw_url = data.get("url", "").strip()

        if not raw_url:
            return jsonify({"error": "Please provide a valid GitHub Issue URL."}), 400

        clean_url = raw_url.split("#")[0].rstrip("/")
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)/(issues|pull)/(\d+)", clean_url, re.IGNORECASE)

        if not match:
            return jsonify({
                "error": "Invalid GitHub URL format. Expected: https://github.com/owner/repo/issues/123"
            }), 400

        owner, repo, issue_or_pr, number = match.groups()
        repo_slug = f"{owner}/{repo}"

        # 1. Pull Request Check (Milestone 2 PR Filtering Protocol)
        if issue_or_pr.lower() == "pull":
            return jsonify({
                "status": "pull_request_filtered",
                "is_pull_request": True,
                "message": (
                    f"Milestone 2 Protocol Enforced: PR #{number} in {repo_slug} is a Pull Request, "
                    "not an Issue ticket. In Milestone 2, pull requests are explicitly filtered out "
                    "to prevent code review diffs and merge messages from skewing issue classification."
                ),
                "url": clean_url,
                "owner": owner,
                "repo": repo,
                "number": int(number),
            })

        # 2. Check local indexed dataset
        url_lower = clean_url.lower()
        if url_lower in LOCAL_ISSUES_MAP:
            cached_issue = LOCAL_ISSUES_MAP[url_lower]
            raw_meta = RAW_ISSUES_MAP.get(url_lower, {})

            # Extract raw labels list
            labels_raw = cached_issue.get("original_labels", "")
            if isinstance(labels_raw, (np.ndarray, list)):
                labels_list = [str(x) for x in labels_raw]
            elif isinstance(labels_raw, str):
                labels_list = [l.strip() for l in labels_raw.split(";") if l.strip()]
            else:
                labels_list = []

            # Map target class
            target_class, reason = resolve_target_label(labels_list)

            # Safely parse comments count
            comments_val = raw_meta.get("comments", 0)
            if isinstance(comments_val, (list, np.ndarray)):
                num_comments = len(comments_val)
            else:
                try:
                    num_comments = int(comments_val)
                except Exception:
                    num_comments = 0

            return jsonify({
                "status": "success",
                "source": "Local Verified Dataset (Hugging Face Datasets Repo)",
                "is_pull_request": False,
                "url": clean_url,
                "repository": str(cached_issue.get("repository", repo_slug)),
                "number": int(cached_issue.get("number", number)),
                "title": str(cached_issue.get("clean_title", "")),
                "body": str(cached_issue.get("clean_body", "")),
                "author_association": str(cached_issue.get("author_association", "NONE")),
                "created_at": str(cached_issue.get("created_at", "")),
                "labels": labels_list,
                "ground_truth_class": target_class,
                "ground_truth_reason": reason,
                "raw_post_triage_metadata": {
                    "state": str(raw_meta.get("state", "closed")),
                    "comments": num_comments,
                    "assignee": bool(raw_meta.get("assignee") is not None),
                    "milestone": bool(raw_meta.get("milestone") is not None),
                }
            })

        # 3. Live External GitHub REST API fetch
        api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{number}"
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                gh_payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as he:
            if he.code == 404:
                return jsonify({"error": f"GitHub Issue #{number} not found in repository {repo_slug}."}), 404
            elif he.code == 403:
                return jsonify({"error": "GitHub API rate limit reached. Try using one of our verified presets."}), 429
            else:
                return jsonify({"error": f"GitHub API error: {he}"}), 500

        # Check if returned object is actually a PR
        if "pull_request" in gh_payload and gh_payload["pull_request"]:
            return jsonify({
                "status": "pull_request_filtered",
                "is_pull_request": True,
                "message": (
                    f"Milestone 2 Protocol Enforced: #{number} in {repo_slug} is a Pull Request. "
                    "PRs are filtered out from training and classification."
                ),
                "url": clean_url,
                "owner": owner,
                "repo": repo,
                "number": int(number),
            })

        # Extract labels
        raw_labels = [l.get("name") for l in gh_payload.get("labels", []) if isinstance(l, dict) and "name" in l]
        target_class, reason = resolve_target_label(raw_labels)

        return jsonify({
            "status": "success",
            "source": f"Live GitHub REST API ({repo_slug})",
            "is_pull_request": False,
            "url": clean_url,
            "repository": repo_slug,
            "number": int(gh_payload.get("number", number)),
            "title": gh_payload.get("title", ""),
            "body": gh_payload.get("body", "") or "",
            "author_association": gh_payload.get("author_association", "NONE"),
            "created_at": gh_payload.get("created_at", ""),
            "labels": raw_labels,
            "ground_truth_class": target_class,
            "ground_truth_reason": reason,
            "raw_post_triage_metadata": {
                "state": gh_payload.get("state"),
                "comments": gh_payload.get("comments", 0),
                "assignee": gh_payload.get("assignee") is not None,
                "milestone": gh_payload.get("milestone") is not None,
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Applies the full Milestone 2 pipeline:
    1. Preprocessing & Text Normalization
    2. Data Leakage Firewall (strips leaky post-triage fields)
    3. Baseline TF-IDF + Logistic Regression inference
    """
    try:
        data = request.get_json(force=True)
        title = data.get("title", "").strip()
        body = data.get("body", "").strip()
        author_association = data.get("author_association", "NONE")
        labels = data.get("labels", [])
        raw_post_triage = data.get("raw_post_triage_metadata", {})

        if not title:
            return jsonify({"error": "Issue title is required."}), 400

        # Step 1: Preprocessing & Cleaning (Milestone 2 Task 5)
        clean_title_text = clean_text(title)
        clean_body_text = clean_text(body)
        combined_text = (clean_title_text + "\n\n" + clean_body_text).strip() if clean_body_text else clean_title_text

        # Step 2: Target Leakage Firewall Audit
        firewalled_fields = []
        for field in ["state", "comments", "assignee", "milestone", "closed_at"]:
            if field in raw_post_triage and raw_post_triage[field] is not None:
                firewalled_fields.append({
                    "field": field,
                    "value": str(raw_post_triage[field]),
                    "reason": "Post-triage metadata; excluded to prevent target leakage."
                })

        # Step 3: Ground Truth Label Mapping (Milestone 2 Task 4)
        ground_truth, mapping_reason = resolve_target_label(labels) if labels else ("UNLABELED", "No source labels")

        # Step 4: Model Inference
        probs = MODEL_PIPELINE.predict_proba([combined_text])[0]
        pred_idx = np.argmax(probs)
        predicted_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        prob_dict = {cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)}
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
            "ground_truth": ground_truth,
            "ground_truth_reason": mapping_reason,
            "pipeline_trace": {
                "clean_title": clean_title_text,
                "clean_body_snippet": clean_body_text[:200] + "..." if len(clean_body_text) > 200 else clean_body_text,
                "title_char_len": len(clean_title_text),
                "body_char_len": len(clean_body_text),
                "word_count": len(combined_text.split()),
                "firewalled_fields": firewalled_fields,
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/milestone-stats")
def milestone_stats():
    """Returns official Milestone 2 audit and dataset metrics."""
    return jsonify({
        "raw_records": 7540,
        "raw_columns": 33,
        "duplicate_rows_removed": 100,
        "unique_entities": 7440,
        "pull_requests_filtered": 4325,
        "pure_issues": 3115,
        "supervised_issues": 1459,
        "unlabeled_issues": 1493,
        "unmapped_issues": 129,
        "conflicting_issues": 34,
        "class_imbalance_ratio": "45.4:1 (BUG: 771 vs OTHER: 17)",
        "classes": {
            "BUG": 771,
            "FEATURE": 615,
            "QUESTION": 31,
            "DOCUMENTATION": 25,
            "OTHER": 17,
        },
        "tests_passed": "7 of 7 (100% pass rate in 2.86s)",
        "figures_count": 11,
    })


@app.route("/api/verified-presets")
def verified_presets():
    """Provides curated real GitHub issue URLs from the verified dataset."""
    presets = [
        {
            "id": "bug_issue",
            "label": "Fatal Arrow Bug (#116)",
            "url": "https://github.com/huggingface/datasets/issues/116",
            "badge": "BUG",
            "desc": "Real verified bug ticket reporting PyArrow table conversion error."
        },
        {
            "id": "feature_stream",
            "label": "Feature Request (#129)",
            "url": "https://github.com/huggingface/datasets/issues/129",
            "badge": "FEATURE",
            "desc": "Real verified feature request proposing torch DataLoader integration."
        },
        {
            "id": "doc_fix",
            "label": "Documentation (#2009)",
            "url": "https://github.com/huggingface/datasets/issues/2009",
            "badge": "DOCUMENTATION",
            "desc": "Real verified documentation ambiguity report."
        },
        {
            "id": "usage_qst",
            "label": "Usage Question (#138)",
            "url": "https://github.com/huggingface/datasets/issues/138",
            "badge": "QUESTION",
            "desc": "Real community question on dataset caching across processes."
        },
        {
            "id": "refactor_task",
            "label": "Maintenance Chore (#4629)",
            "url": "https://github.com/huggingface/datasets/issues/4629",
            "badge": "OTHER",
            "desc": "Internal repository maintenance task."
        },
        {
            "id": "sample_pr",
            "label": "Pull Request (#20) [PR Filter Test]",
            "url": "https://github.com/huggingface/datasets/pull/20",
            "badge": "PR FILTER",
            "desc": "Tests Milestone 2 rule enforcing pull request exclusion."
        }
    ]
    return jsonify(presets)


@app.route("/download/<doc_type>")
def download_deliverable(doc_type):
    """Direct deliverable download for college report & viva."""
    mapping = {
        "report-pdf": ("reports/milestone_2_report.pdf", "GitHub_Issue_Classifier_Milestone_2_Report.pdf"),
        "presentation-pptx": ("reports/milestone_2_presentation.pptx", "Milestone_2_Presentation_Priyanshu.pptx"),
        "audit-json": ("reports/dataset_audit.json", "dataset_audit.json"),
        "audit-md": ("reports/dataset_audit.md", "dataset_audit.md"),
    }
    if doc_type in mapping:
        path, filename = mapping[doc_type]
        if os.path.exists(path):
            return send_file(os.path.abspath(path), as_attachment=True, download_name=filename)
    return jsonify({"error": "File not found"}), 404


@app.route("/reports/figures/<path:filename>")
def serve_figure(filename):
    return send_from_directory(os.path.abspath("reports/figures"), filename)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" GitHub Project Issue Classifier - Milestone 2 Dashboard")
    print(" Running at: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
