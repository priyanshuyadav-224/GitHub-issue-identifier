"""
assess_split_readiness.py
=========================
Milestone 2 - Task 7: Dataset Split Readiness Assessment
Author: Priyanshu Yadav (S25CSEU0725)
Project: GitHub Project Issue Classifier

This script evaluates whether the cleaned supervised dataset can support
rigorous, unbiased ML evaluation in Milestone 3. It checks class counts,
evaluates stratified k-fold feasibility, detects near-duplicate title clusters
that could cross train/test boundaries, evaluates temporal split feasibility,
and outputs docs/split_readiness_report.md.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def assess_splits(supervised_path: str, output_report: str):
    if not os.path.exists(supervised_path):
        raise FileNotFoundError(f"File not found: {supervised_path}")

    df = pd.read_parquet(supervised_path)
    total_records = len(df)

    # 1. Class Counts & Stratification Feasibility
    class_counts = df["target_class"].value_counts()
    min_class = class_counts.min()
    min_class_name = class_counts.idxmin()

    # Stratified K-Fold check (can we do 5-fold?)
    k_folds = 5
    stratified_kfold_feasible = min_class >= k_folds

    # 80/20 Train/Test split simulation
    test_ratio = 0.20
    test_min_samples = int(np.floor(min_class * test_ratio))

    # 2. Near-Duplicate Analysis
    # Vectorize titles to find cosine similarity > 0.85
    tfidf = TfidfVectorizer(stop_words="english", max_features=1000)
    title_vecs = tfidf.fit_transform(df["clean_title"])
    sim_matrix = cosine_similarity(title_vecs)
    np.fill_diagonal(sim_matrix, 0)

    # Pairs with cosine similarity >= 0.85
    high_sim_pairs = np.argwhere(sim_matrix >= 0.85)
    # Deduplicate unordered pairs
    unique_sim_pairs = set()
    for i, j in high_sim_pairs:
        if i < j:
            unique_sim_pairs.add((i, j))
    near_dup_count = len(unique_sim_pairs)

    # Exact title duplicates
    exact_title_dups = total_records - df["clean_title"].str.lower().nunique()

    # 3. Temporal Split Assessment
    df["created_dt"] = pd.to_datetime(df["created_at"])
    df = df.sort_values("created_dt").reset_index(drop=True)
    cutoff_idx = int(total_records * 0.80)
    cutoff_date = df.iloc[cutoff_idx]["created_dt"]

    train_temp = df.iloc[:cutoff_idx]
    test_temp = df.iloc[cutoff_idx:]

    temp_train_counts = train_temp["target_class"].value_counts().to_dict()
    temp_test_counts = test_temp["target_class"].value_counts().to_dict()

    # Write report
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("# Dataset Split Readiness Assessment Report\n\n")
        f.write("**Author:** Priyanshu Yadav (S25CSEU0725)  \n")
        f.write("**Project:** GitHub Project Issue Classifier  \n")
        f.write(f"**Target Evaluated Dataset:** `{supervised_path}` ({total_records:,} supervised records)  \n\n")

        f.write("## 1. Class Frequencies & Stratification Feasibility\n\n")
        f.write("| Target Class | Total Count | % of Dataset | 5-Fold Min Count/Fold | 80/20 Test Set Expected |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for cls, cnt in class_counts.items():
            pct = (cnt / total_records) * 100
            fold_cnt = cnt // 5
            test_cnt = int(np.round(cnt * 0.20))
            f.write(f"| **`{cls}`** | {cnt:,} | {pct:.2f}% | ~{fold_cnt} | ~{test_cnt} |\n")
        f.write("\n")

        f.write(f"- **Minimum Class Size:** `{min_class_name}` has {min_class} instances.\n")
        f.write(f"- **5-Fold Stratified Split Feasible:** `{'YES' if stratified_kfold_feasible else 'NO'}` (Minimum class {min_class} >= 5 folds).\n")
        f.write(f"- **10-Fold Stratified Split Feasible:** `{'YES' if min_class >= 10 else 'NO'}` (Minimum class {min_class} >= 10 folds).\n\n")

        f.write("## 2. Duplicate & Near-Duplicate Contamination Risk\n\n")
        f.write(f"- **Exact Title Matches:** {exact_title_dups} issues share identical lowercase titles (e.g. repeated error messages like `ImportError: cannot import name ...`).\n")
        f.write(f"- **Near-Duplicate Pairs (TF-IDF Cosine Similarity >= 0.85):** {near_dup_count} pairs detected.\n")
        f.write("- **Contamination Hazard:** If near-duplicate issue reports are split arbitrarily between train and test partitions, the model can memorize specific title phrases and inflate test accuracy. A deduplicated stratified split or title-grouped split must be enforced during Milestone 3 cross-validation.\n\n")

        f.write("## 3. Split Strategy Comparative Analysis\n\n")
        f.write("| Split Strategy | Feasibility | Strengths | Limitations | Recommendation |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        f.write("| **Random Stratified Split** | **High** | Preserves minority class proportions (QUESTION, DOCUMENTATION, OTHER) in both train and test. | May allow near-duplicate issues into test set without grouping. | **Primary Evaluation Scheme** (with near-duplicate grouping). |\n")
        f.write("| **Temporal (Time-Based) Split** | **Moderate** | Perfectly reflects production deployment: train on past issues, test on future incoming issues. | Extreme minority classes (`OTHER`: 17) may be sparsely represented in recent test window. | **Secondary Benchmark Scheme** for temporal robustness. |\n")
        f.write("| **Repository-Grouped Split** | **N/A** | Tests cross-repo generalizability. | Current verified dataset is from single monolithic repository (`huggingface/datasets`). | Deferred to multi-repo extensions in Milestone 5. |\n\n")

        f.write("## 4. Temporal Split Simulation (80% Past / 20% Future)\n\n")
        f.write(f"- **Cutoff Timestamp:** `{cutoff_date}`\n")
        f.write(f"- **Past (Train) Partition:** {len(train_temp):,} issues (`{train_temp['created_dt'].min()}` to `{cutoff_date}`)\n")
        f.write(f"- **Future (Test) Partition:** {len(test_temp):,} issues (`{cutoff_date}` to `{test_temp['created_dt'].max()}`)\n\n")
        f.write("Class distribution across temporal split:\n\n")
        f.write("| Class | Train Partition Count | Test Partition Count |\n")
        f.write("| :--- | :--- | :--- |\n")
        all_classes = sorted(list(set(list(temp_train_counts.keys()) + list(temp_test_counts.keys()))))
        for cls in all_classes:
            tr = temp_train_counts.get(cls, 0)
            te = temp_test_counts.get(cls, 0)
            f.write(f"| `{cls}` | {tr:,} | {te:,} |\n")
        f.write("\n")

        f.write("## 5. Strict Data Leakage & Evaluation Rules for Milestone 3\n\n")
        f.write("1. **No Fitting on Full Dataset:** TF-IDF vectorizers, scalers, and dimensionality reducers (PCA) must only be fit on `X_train`. Calling `.fit_transform()` on the full dataset before splitting is strictly forbidden.\n")
        f.write("2. **Target Leakage Exclusion:** Input feature vectors must contain only creation-time fields (`text`, `clean_title`, `clean_body`, `author_association`, `created_at`). Excluded fields (`state`, `comments`, `assignee`, `milestone`, `reactions`) must never enter `X`.\n")
        f.write("3. **Macro-Averaging:** Due to the 45:1 class imbalance between `BUG` (771) and `OTHER` (17), all milestone scorecards must report **Macro-F1, Balanced Accuracy, and per-class metrics** in addition to overall accuracy.\n")

    print(f"Split readiness assessment complete. Report saved to: {output_report}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assess dataset split readiness.")
    parser.add_argument(
        "--input",
        type=str,
        default="c:/Projects/data/processed/github_issues_supervised.parquet",
        help="Path to supervised parquet dataset",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="c:/Projects/docs/split_readiness_report.md",
        help="Path for split readiness Markdown report",
    )
    args = parser.parse_args()

    assess_splits(args.input, args.output_report)
