"""
preprocess_dataset.py
=====================
Milestone 2 - Task 2, Task 4, and Task 5: Ingestion, Target Definition, and Preprocessing
Author: Priyanshu Yadav (S25CSEU0725)
Project: GitHub Project Issue Classifier

This script ingests raw GitHub issue data, applies deterministic cleaning,
removes duplicate records and pull requests, prevents target leakage by strictly
separating creation-time features from post-triage metadata, maps source labels
to an explicit 5-class target taxonomy (BUG, FEATURE, QUESTION, DOCUMENTATION, OTHER),
and outputs both the full cleaned issue dataset and the supervised training dataset.
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd


# Explicit Label Taxonomy Mapping
# Maps raw community GitHub labels to normalized target classes.
TYPE_MAPPING = {
    "bug": "BUG",
    "dataset bug": "BUG",
    "metric bug": "BUG",
    "enhancement": "FEATURE",
    "dataset request": "FEATURE",
    "dataset contribution": "FEATURE",
    "documentation": "DOCUMENTATION",
    "question": "QUESTION",
    "generic discussion": "QUESTION",
    "Dataset discussion": "QUESTION",
    "Metric discussion": "QUESTION",
    "maintenance": "OTHER",
    "refactoring": "OTHER",
}

# Orthogonal tags that do not define the issue type
ORTHOGONAL_LABELS = {
    # Triage and difficulty tags
    "good first issue",
    "good second issue",
    "duplicate",
    "hacktoberfest",
    "wontfix",
    "help wanted",
    # Component and technical area tags
    "dataset-viewer",
    "vision",
    "streaming",
    "speech",
    "nlp-viewer",
    "hosted-on-google-drive",
    "audio_column",
    "arrow",
    "graph",
    "dataset-viewer-rgba-images",
    "dataset-viewer-gated",
}

# Prediction-Time Feature Allowlist (Available when issue is submitted)
FEATURE_ALLOWLIST = [
    "id",                  # Provenance / stable identifier
    "number",              # Provenance / GitHub issue number
    "html_url",            # Provenance / issue web link
    "repository",          # Domain context (e.g. huggingface/datasets)
    "author_association",  # Relationship of creator to project at creation time
    "created_at",          # Timestamp of creation (for temporal splitting)
    "clean_title",         # Cleaned issue title
    "clean_body",          # Cleaned issue description
    "text",                # Documented combined feature: clean_title + \n\n + clean_body
    "title_char_len",      # Metadata feature: title character length
    "body_char_len",       # Metadata feature: body character length
    "text_char_len",       # Metadata feature: combined character length
    "text_word_count",     # Metadata feature: word count
    "original_labels",     # Raw label list (audit / provenance only)
    "target_class",        # Classification label (BUG, FEATURE, QUESTION, DOCUMENTATION, OTHER, UNLABELED, UNMAPPED, CONFLICTING)
    "mapping_status",      # Rationale for target mapping
]

# Excluded Fields (Target Leakage & Post-Triage Metadata)
EXCLUDED_LEAKAGE_FIELDS = [
    "state",               # Post-triage closure outcome ('closed' or 'open')
    "closed_at",           # Post-triage timestamp of resolution
    "closed_by",           # Post-triage user identity who resolved
    "state_reason",        # Post-triage resolution reason ('completed', 'not_planned')
    "comments",            # Accumulated discussions after creation
    "comments_url",        # Post-triage comments endpoint
    "events_url",          # Post-triage event timeline
    "timeline_url",        # Post-triage timeline URL
    "assignee",            # Maintainer assignment after triage
    "assignees",           # Multiple maintainer assignments
    "milestone",           # Release target milestone assigned after triage
    "reactions",           # Community votes / reactions accumulated over time
    "locked",              # Administrative lock applied after creation
    "active_lock_reason",  # Lock rationale
    "performed_via_github_app"
]


def extract_label_names(lbls):
    """Safely extracts label string names from dicts or arrays."""
    if lbls is None or (isinstance(lbls, float) and np.isnan(lbls)):
        return []
    res = []
    for item in lbls:
        if isinstance(item, dict) and "name" in item:
            res.append(item["name"])
        elif isinstance(item, str):
            res.append(item)
    return res


def resolve_target_label(label_names: list) -> tuple:
    """
    Applies the documented target taxonomy policy.
    Returns: (target_class, resolution_reason)
    """
    if not label_names:
        return "UNLABELED", "No labels present on issue"

    matched_types = set()
    for lbl in label_names:
        if lbl in TYPE_MAPPING:
            matched_types.add(TYPE_MAPPING[lbl])

    if len(matched_types) == 0:
        return "UNMAPPED", f"Only non-type / orthogonal labels present: {label_names}"
    elif len(matched_types) == 1:
        return list(matched_types)[0], "Single mapped category"
    else:
        return "CONFLICTING", f"Conflicting categories detected: {sorted(list(matched_types))}"


def clean_text(text: str) -> str:
    """
    Normalizes issue text while strictly preserving technical identifiers,
    error messages, version numbers, paths, code symbols, and negations.
    """
    if text is None or pd.isna(text):
        return ""
    text = str(text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip HTML tags (e.g. <details>, <summary>, <br>)
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip markdown image tags ![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)

    # Normalize horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive newlines (> 2 newlines to 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def preprocess_data(
    input_path: str,
    output_cleaned_parquet: str,
    output_cleaned_csv: str,
    output_supervised_parquet: str,
    output_supervised_csv: str,
    seed: int = 42,
) -> tuple:
    """
    Executes the deterministic preprocessing and leakage prevention pipeline.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input raw dataset file not found: {input_path}")

    print(f"Loading raw dataset from: {input_path}")
    df_raw = pd.read_parquet(input_path)
    initial_count = len(df_raw)
    print(f"Initial raw record count: {initial_count:,}")

    # Step 1: Deduplication on stable ID
    df_dedup = df_raw.drop_duplicates(subset=["id"], keep="first").copy()
    dedup_count = len(df_dedup)
    duplicate_count = initial_count - dedup_count
    print(f"Removed {duplicate_count:,} duplicate rows. Remaining: {dedup_count:,}")

    # Step 2: Filter pull requests (PRs)
    if "pull_request" in df_dedup.columns:
        is_pr = df_dedup["pull_request"].apply(
            lambda x: isinstance(x, dict) or (isinstance(x, (list, np.ndarray)) and len(x) > 0) or pd.notna(x)
        )
    else:
        is_pr = pd.Series([False] * len(df_dedup))

    df_issues = df_dedup[~is_pr].copy()
    pr_count = int(is_pr.sum())
    pure_issues_count = len(df_issues)
    print(f"Filtered {pr_count:,} pull requests. Remaining pure issues: {pure_issues_count:,}")

    # Step 3: Extract repository identifier
    if "repository_url" in df_issues.columns:
        df_issues["repository"] = df_issues["repository_url"].apply(
            lambda x: str(x).replace("https://api.github.com/repos/", "")
        )
    else:
        df_issues["repository"] = "unknown"

    # Step 4: Extract and resolve labels
    df_issues["original_labels"] = df_issues["labels"].apply(extract_label_names)
    target_results = df_issues["original_labels"].apply(resolve_target_label)
    df_issues["target_class"] = [res[0] for res in target_results]
    df_issues["mapping_status"] = [res[1] for res in target_results]

    # Step 5: Clean and combine text fields
    df_issues["clean_title"] = df_issues["title"].apply(clean_text)
    df_issues["clean_body"] = df_issues["body"].apply(clean_text)
    df_issues["text"] = df_issues.apply(
        lambda row: (row["clean_title"] + "\n\n" + row["clean_body"]).strip()
        if row["clean_body"]
        else row["clean_title"].strip(),
        axis=1,
    )

    # Compute text metadata
    df_issues["title_char_len"] = df_issues["clean_title"].apply(len)
    df_issues["body_char_len"] = df_issues["clean_body"].apply(len)
    df_issues["text_char_len"] = df_issues["text"].apply(len)
    df_issues["text_word_count"] = df_issues["text"].apply(lambda x: len(x.split()))

    # Normalize author association
    if "author_association" not in df_issues.columns:
        df_issues["author_association"] = "NONE"
    else:
        df_issues["author_association"] = df_issues["author_association"].fillna("NONE")

    # Normalize html_url if absent
    if "html_url" not in df_issues.columns:
        df_issues["html_url"] = ""

    # Select strictly allowed features (Preventing Target Leakage)
    df_cleaned = df_issues[FEATURE_ALLOWLIST].copy()

    # Sort deterministically by id
    df_cleaned = df_cleaned.sort_values(by="id").reset_index(drop=True)

    # Filter supervised subset (valid target classes only)
    valid_classes = ["BUG", "FEATURE", "DOCUMENTATION", "QUESTION", "OTHER"]
    df_supervised = df_cleaned[df_cleaned["target_class"].isin(valid_classes)].copy()
    df_supervised = df_supervised.sort_values(by="id").reset_index(drop=True)

    # Save outputs
    for path in [output_cleaned_parquet, output_cleaned_csv, output_supervised_parquet, output_supervised_csv]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    print(f"\nSaving full cleaned issues to:\n  - {output_cleaned_parquet}\n  - {output_cleaned_csv}")
    df_cleaned.to_parquet(output_cleaned_parquet, index=False)
    # Convert list column to string for clean CSV output
    df_cleaned_csv = df_cleaned.copy()
    df_cleaned_csv["original_labels"] = df_cleaned_csv["original_labels"].apply(lambda x: ";".join(x))
    df_cleaned_csv.to_csv(output_cleaned_csv, index=False)

    print(f"\nSaving supervised classification dataset to:\n  - {output_supervised_parquet}\n  - {output_supervised_csv}")
    df_supervised.to_parquet(output_supervised_parquet, index=False)
    df_sup_csv = df_supervised.copy()
    df_sup_csv["original_labels"] = df_sup_csv["original_labels"].apply(lambda x: ";".join(x))
    df_sup_csv.to_csv(output_supervised_csv, index=False)

    print("\n--- Preprocessing Summary ---")
    print(f"Total pure issues processed: {len(df_cleaned):,}")
    print(f"Supervised issues count: {len(df_supervised):,}")
    print("\nSupervised Class Distribution:")
    class_counts = df_supervised["target_class"].value_counts()
    for cls, cnt in class_counts.items():
        pct = (cnt / len(df_supervised)) * 100
        print(f"  {cls:<15}: {cnt:>5,} ({pct:>5.2f}%)")

    print("\nFiltered/Non-Supervised Issues Breakdown:")
    non_sup = df_cleaned[~df_cleaned["target_class"].isin(valid_classes)]
    for status, cnt in non_sup["target_class"].value_counts().items():
        print(f"  {status:<15}: {cnt:>5,} ({cnt/len(df_cleaned)*100:>5.2f}%)")

    return df_cleaned, df_supervised


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess GitHub issues dataset.")
    parser.add_argument(
        "--input",
        type=str,
        default="c:/Projects/data/raw/github_issues_raw.parquet",
        help="Path to raw dataset file",
    )
    parser.add_argument(
        "--output-cleaned-parquet",
        type=str,
        default="c:/Projects/data/processed/github_issues_cleaned.parquet",
        help="Path to output cleaned parquet",
    )
    parser.add_argument(
        "--output-cleaned-csv",
        type=str,
        default="c:/Projects/data/processed/github_issues_cleaned.csv",
        help="Path to output cleaned csv",
    )
    parser.add_argument(
        "--output-supervised-parquet",
        type=str,
        default="c:/Projects/data/processed/github_issues_supervised.parquet",
        help="Path to output supervised parquet",
    )
    parser.add_argument(
        "--output-supervised-csv",
        type=str,
        default="c:/Projects/data/processed/github_issues_supervised.csv",
        help="Path to output supervised csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    preprocess_data(
        args.input,
        args.output_cleaned_parquet,
        args.output_cleaned_csv,
        args.output_supervised_parquet,
        args.output_supervised_csv,
        args.seed,
    )
