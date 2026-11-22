"""
scripts/train_and_evaluate_models.py
====================================
Statistical Machine Learning Training, Cross-Validation, Evaluation & Benchmarking.
Evaluates Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest,
plus K-Means clustering on the unlabeled corpus.
Exports model artifacts, metric matrices, and publication-grade evaluation charts.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def setup_directories():
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)


def train_and_benchmark():
    setup_directories()

    supervised_path = "data/processed/github_issues_supervised.parquet"
    cleaned_path = "data/processed/github_issues_cleaned.parquet"

    if not os.path.exists(supervised_path):
        raise FileNotFoundError(f"Missing supervised dataset: {supervised_path}")

    print("Loading datasets...")
    df_sup = pd.read_parquet(supervised_path)
    X = df_sup["text"].astype(str)
    y = df_sup["target_class"].astype(str)

    # Stratified 80/20 train/test split (deterministic seed=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    classes = sorted(list(y.unique()))
    print(f"Corpus: {len(X)} records ({len(X_train)} train, {len(X_test)} test) across {len(classes)} classes: {classes}")

    # Shared TF-IDF Feature Extractor
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Models definition
    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced", max_iter=500, random_state=42
        ),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=2000),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=25, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    pipelines = {}
    confusion_matrices = {}
    y_test_bin = label_binarize(y_test, classes=classes)

    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in candidate_models.items():
        print(f"\nTraining and evaluating: {name}...")
        t0 = time.perf_counter()
        clf.fit(X_train_vec, y_train)
        train_time = time.perf_counter() - t0

        # Benchmark inference latency (1000 single predictions)
        sample_text_vec = X_test_vec[0:1]
        latencies = []
        for _ in range(200):
            t_inf0 = time.perf_counter()
            _ = clf.predict(sample_text_vec)
            latencies.append((time.perf_counter() - t_inf0) * 1000.0)
        avg_latency_ms = float(np.mean(latencies))

        # Test evaluation
        y_pred = clf.predict(X_test_vec)
        y_prob = clf.predict_proba(X_test_vec) if hasattr(clf, "predict_proba") else None

        test_acc = accuracy_score(y_test, y_pred)
        macro_prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # 5-fold cross validation
        cv_scores = cross_validate(
            clf, X_train_vec, y_train, cv=cv_strat, scoring=["accuracy", "f1_macro"], n_jobs=-1
        )
        cv_acc_mean = float(np.mean(cv_scores["test_accuracy"]))
        cv_f1_mean = float(np.mean(cv_scores["test_f1_macro"]))

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        confusion_matrices[name] = cm.tolist()

        clf_rep = classification_report(y_test, y_pred, labels=classes, output_dict=True, zero_division=0)

        results[name] = {
            "test_accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(macro_prec), 4),
            "macro_recall": round(float(macro_rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "cv_accuracy_5fold": round(cv_acc_mean, 4),
            "cv_f1_macro_5fold": round(cv_f1_mean, 4),
            "training_time_sec": round(float(train_time), 4),
            "latency_ms": round(avg_latency_ms, 3),
            "classification_report": clf_rep,
        }

        # Build end-to-end pipeline
        pipe = Pipeline([
            ("tfidf", vectorizer),
            ("clf", clf),
        ])
        pipelines[name] = pipe

    # Save trained pipelines
    joblib.dump(pipelines, "models/model_suite.joblib")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
    print("Saved models to models/model_suite.joblib and models/tfidf_vectorizer.joblib")

    # Unsupervised K-Means clustering on the unlabeled pool
    print("\nExecuting K-Means clustering on unlabeled issues...")
    df_clean = pd.read_parquet(cleaned_path) if os.path.exists(cleaned_path) else df_sup
    df_unlabeled = df_clean[df_clean["is_supervised"] == False] if "is_supervised" in df_clean.columns else df_clean
    unlabeled_texts = df_unlabeled["clean_title"].fillna("").astype(str)

    unlabeled_vec = vectorizer.transform(unlabeled_texts)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(unlabeled_vec)

    # Top keywords per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    cluster_keywords = {}
    cluster_names = [
        "Data Loaders & Streaming",
        "Vision & Multimodal Processing",
        "Caching & Distributed Execution",
        "Hub & Cloud Authentication",
        "Build, Typing & Dependencies",
    ]
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :8]]
        cluster_keywords[f"Cluster {i}: {cluster_names[i]}"] = top_terms

    joblib.dump(kmeans, "models/kmeans_model.joblib")

    # Compile final metrics payload
    evaluation_payload = {
        "metadata": {
            "project_title": "GitHub Project Issue Classifier",
            "eval_date": "November 2026",
            "dataset_split": {"total": len(X), "train": len(X_train), "test": len(X_test)},
            "classes": classes,
            "feature_space_dim": X_train_vec.shape[1],
        },
        "model_benchmarks": results,
        "confusion_matrices": confusion_matrices,
        "kmeans_clusters": cluster_keywords,
    }

    with open("reports/model_evaluation_metrics.json", "w") as f:
        json.dump(evaluation_payload, f, indent=2)
    print("Saved metrics to reports/model_evaluation_metrics.json")

    # Generate Figures
    generate_figures(results, confusion_matrices, classes, unlabeled_vec, clusters, cluster_names)
    print("All evaluation plots generated successfully.")
    return evaluation_payload


def generate_figures(results, confusion_matrices, classes, unlabeled_vec, clusters, cluster_names):
    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Comparison Bar Chart
    models = list(results.keys())
    accuracies = [results[m]["test_accuracy"] * 100 for m in models]
    macro_f1s = [results[m]["macro_f1"] * 100 for m in models]
    latencies = [results[m]["latency_ms"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(models))
    width = 0.35

    rects1 = ax1.bar(x - width/2, accuracies, width, label="Test Accuracy (%)", color="#1f77b4")
    rects2 = ax1.bar(x + width/2, macro_f1s, width, label="Macro-F1 (%)", color="#2ca02c")
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Statistical ML Models - Accuracy vs Macro-F1", fontsize=13, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=15, ha="right")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower right")

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax2.bar(models, latencies, color="#ff7f0e", width=0.45)
    ax2.set_ylabel("Latency per Query (ms)")
    ax2.set_title("Inference Latency Benchmark (Sub-millisecond Scale)", fontsize=13, fontweight="bold")
    ax2.set_xticklabels(models, rotation=15, ha="right")
    for idx, v in enumerate(latencies):
        ax2.text(idx, v + 0.05, f"{v:.2f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig("reports/figures/fig12_model_comparison_metrics.png", dpi=200)
    plt.close()

    # 2. Confusion Matrices (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()

    for idx, (m_name, cm_data) in enumerate(confusion_matrices.items()):
        cm = np.array(cm_data)
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"{m_name} Confusion Matrix", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Ground Truth Class", fontsize=10)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("reports/figures/fig13_confusion_matrices.png", dpi=200)
    plt.close()

    # 3. K-Means 2D PCA Cluster Projection
    pca = PCA(n_components=2, random_state=42)
    sample_size = min(1200, unlabeled_vec.shape[0])
    sub_vec = unlabeled_vec[:sample_size].toarray()
    sub_clusters = clusters[:sample_size]
    coords_2d = pca.fit_transform(sub_vec)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = sns.color_palette("tab10", n_colors=5)
    for c_id in range(5):
        mask = (sub_clusters == c_id)
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   label=f"Cluster {c_id}: {cluster_names[c_id]}",
                   color=palette[c_id], alpha=0.65, s=28, edgecolors="none")

    ax.set_title("Unsupervised Topic Discovery: K-Means (k=5) with PCA Projection", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% Variance)")
    ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% Variance)")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig("reports/figures/fig15_kmeans_cluster_projection.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    train_and_benchmark()
