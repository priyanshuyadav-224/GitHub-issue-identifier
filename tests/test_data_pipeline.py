"""
test_data_pipeline.py
=====================
Automated Test Suite for Milestone 2 Data Pipeline
Author: Priyanshu Yadav (S25CSEU0725)
Project: GitHub Project Issue Classifier

Tests verify:
- Loading of raw parquet format
- Missing text handling and normalization
- Deterministic label mapping and conflict detection
- Duplicate ID detection and deduplication
- Pull request filtering
- Target leakage prevention (exclusion of post-triage fields)
- Deterministic cleaning and text feature extraction
- Cleaned and supervised output schema validation
- Real processed dataset integrity
"""

import os
import sys

# Ensure project root is in Python module path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from scripts.preprocess_dataset import (
    clean_text,
    extract_label_names,
    resolve_target_label,
    preprocess_data,
    TYPE_MAPPING,
    FEATURE_ALLOWLIST,
    EXCLUDED_LEAKAGE_FIELDS,
)


@pytest.fixture
def synthetic_raw_dataset():
    """
    Creates a small synthetic dataset representing raw GitHub API records
    with known edge cases.
    NOTE: SYNTHETIC DATA FIXTURE - USED ONLY FOR AUTOMATED TESTING.
    """
    data = [
        # Normal Bug issue
        {
            "id": 1001,
            "number": 1,
            "title": "Fix memory leak in arrow iterator",
            "body": "When calling __iter__ repeatedly, memory grows unboundedly. Traceback attached.",
            "labels": np.array([{"name": "bug", "color": "e11d21"}], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 2,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-01T10:00:00Z",
        },
        # Duplicate of record 1001
        {
            "id": 1001,
            "number": 1,
            "title": "Fix memory leak in arrow iterator",
            "body": "When calling __iter__ repeatedly, memory grows unboundedly. Traceback attached.",
            "labels": np.array([{"name": "bug", "color": "e11d21"}], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 2,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-01T10:00:00Z",
        },
        # Pull Request record (should be filtered out)
        {
            "id": 1002,
            "number": 2,
            "title": "Add support for audio streaming dataset",
            "body": "This PR implements audio streaming.",
            "labels": np.array([{"name": "enhancement", "color": "a2eeef"}], dtype=object),
            "state": "closed",
            "closed_at": "2022-01-03T12:00:00Z",
            "comments": 5,
            "assignee": {"login": "maintainer"},
            "milestone": {"title": "v2.0"},
            "pull_request": {"url": "https://api.github.com/repos/huggingface/datasets/pulls/2"},
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "CONTRIBUTOR",
            "created_at": "2022-01-02T10:00:00Z",
        },
        # Feature issue with orthogonal triage tag
        {
            "id": 1003,
            "number": 3,
            "title": "Request: support WebDataset tar format",
            "body": "It would be great to load webdataset archives directly.",
            "labels": np.array([{"name": "dataset request"}, {"name": "good first issue"}], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 0,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-03T10:00:00Z",
        },
        # Conflicting labels issue (both bug and enhancement)
        {
            "id": 1004,
            "number": 4,
            "title": "Ambiguous behavior in cache cleanup",
            "body": "Is this a bug or a feature? Cache files delete randomly.",
            "labels": np.array([{"name": "bug"}, {"name": "enhancement"}], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 1,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-04T10:00:00Z",
        },
        # Issue with Missing Body (null) and Documentation tag
        {
            "id": 1005,
            "number": 5,
            "title": "Typo in quickstart documentation code sample",
            "body": None,
            "labels": np.array([{"name": "documentation"}], dtype=object),
            "state": "closed",
            "closed_at": "2022-01-06T10:00:00Z",
            "comments": 1,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "MEMBER",
            "created_at": "2022-01-05T10:00:00Z",
        },
        # Unlabeled Issue (0 labels)
        {
            "id": 1006,
            "number": 6,
            "title": "Random question about downloading dataset splits",
            "body": "How do I split into train and test?",
            "labels": np.array([], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 0,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-06T10:00:00Z",
        },
        # Question / Discussion issue
        {
            "id": 1007,
            "number": 7,
            "title": "Question: best way to stream parquet shards?",
            "body": "Should I use map or stream iterator?",
            "labels": np.array([{"name": "question"}], dtype=object),
            "state": "open",
            "closed_at": None,
            "comments": 3,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "NONE",
            "created_at": "2022-01-07T10:00:00Z",
        },
        # Maintenance / Refactoring issue -> OTHER
        {
            "id": 1008,
            "number": 8,
            "title": "Refactor internal tensor conversion helpers",
            "body": "Clean up deprecated numpy array casting.",
            "labels": np.array([{"name": "maintenance"}], dtype=object),
            "state": "closed",
            "closed_at": "2022-01-08T10:00:00Z",
            "comments": 1,
            "assignee": None,
            "milestone": None,
            "pull_request": None,
            "repository_url": "https://api.github.com/repos/huggingface/datasets",
            "html_url": "https://github.com/huggingface/datasets/issues/1",
            "author_association": "MEMBER",
            "created_at": "2022-01-08T10:00:00Z",
        },
    ]
    return pd.DataFrame(data)


def test_loading_chosen_format(tmp_path, synthetic_raw_dataset):
    """Test loading of parquet format with pandas."""
    parquet_path = tmp_path / "test_raw.parquet"
    synthetic_raw_dataset.to_parquet(parquet_path, index=False)
    assert os.path.exists(parquet_path)

    loaded_df = pd.read_parquet(parquet_path)
    assert len(loaded_df) == len(synthetic_raw_dataset)
    assert "title" in loaded_df.columns
    assert "id" in loaded_df.columns


def test_missing_text_normalization():
    """Test text cleaning handles None, empty strings, and Markdown/HTML cleanly."""
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""
    assert clean_text("   \n\n  ") == ""

    # HTML tag stripping
    raw_html = "<details><summary>Traceback</summary><code>Error: 404</code></details>"
    cleaned = clean_text(raw_html)
    assert "<" not in cleaned and ">" not in cleaned
    assert "Traceback" in cleaned
    assert "Error: 404" in cleaned

    # Markdown image stripping
    raw_md = "Look at this error: ![screenshot](https://example.com/err.png) crash!"
    assert "![" not in clean_text(raw_md)
    assert "crash!" in clean_text(raw_md)

    # Technical identifiers preserved
    code_text = "Calling __iter__() with v1.2.3 failed at /usr/local/lib."
    assert clean_text(code_text) == code_text


def test_label_mapping_and_conflicts():
    """Test transparent taxonomy resolution for single, multi, unmapped, and conflicting tags."""
    # Single category mappings
    assert resolve_target_label(["bug"])[0] == "BUG"
    assert resolve_target_label(["dataset bug"])[0] == "BUG"
    assert resolve_target_label(["enhancement"])[0] == "FEATURE"
    assert resolve_target_label(["dataset request"])[0] == "FEATURE"
    assert resolve_target_label(["documentation"])[0] == "DOCUMENTATION"
    assert resolve_target_label(["question"])[0] == "QUESTION"
    assert resolve_target_label(["generic discussion"])[0] == "QUESTION"
    assert resolve_target_label(["maintenance"])[0] == "OTHER"

    # Multi-label with orthogonal non-type tag
    assert resolve_target_label(["bug", "good first issue"])[0] == "BUG"
    assert resolve_target_label(["enhancement", "vision"])[0] == "FEATURE"

    # Conflicting type tags
    cls, reason = resolve_target_label(["bug", "enhancement"])
    assert cls == "CONFLICTING"
    assert "Conflicting categories" in reason

    # Unmapped tags (only triage/area tags)
    cls, reason = resolve_target_label(["good first issue", "hacktoberfest"])
    assert cls == "UNMAPPED"

    # Empty labels
    cls, reason = resolve_target_label([])
    assert cls == "UNLABELED"


def test_duplicate_and_pull_request_filtering(tmp_path, synthetic_raw_dataset):
    """Test pipeline deduplication on stable ID and pull request exclusion."""
    raw_path = tmp_path / "raw.parquet"
    synthetic_raw_dataset.to_parquet(raw_path, index=False)

    out_clean_pq = tmp_path / "cleaned.parquet"
    out_clean_csv = tmp_path / "cleaned.csv"
    out_sup_pq = tmp_path / "sup.parquet"
    out_sup_csv = tmp_path / "sup.csv"

    cleaned_df, sup_df = preprocess_data(
        str(raw_path),
        str(out_clean_pq),
        str(out_clean_csv),
        str(out_sup_pq),
        str(out_sup_csv),
    )

    # Initial had 9 rows:
    # - 1 duplicate ID (id 1001 appears twice) -> 8 unique
    # - 1 Pull Request (id 1002) -> 7 pure issues
    assert len(cleaned_df) == 7
    assert 1002 not in cleaned_df["id"].values  # PR must be excluded
    assert cleaned_df["id"].nunique() == 7      # No duplicate IDs

    # Supervised subset excludes CONFLICTING (id 1004), UNLABELED (id 1006)
    # Remaining: 1001 (BUG), 1003 (FEATURE), 1005 (DOCUMENTATION), 1007 (QUESTION), 1008 (OTHER) -> 5
    assert len(sup_df) == 5
    expected_classes = {"BUG", "FEATURE", "DOCUMENTATION", "QUESTION", "OTHER"}
    assert set(sup_df["target_class"].unique()) == expected_classes


def test_leakage_fields_exclusion(tmp_path, synthetic_raw_dataset):
    """Verify that NO post-triage leakage fields exist in the cleaned feature allowlist."""
    raw_path = tmp_path / "raw.parquet"
    synthetic_raw_dataset.to_parquet(raw_path, index=False)

    out_clean_pq = tmp_path / "cleaned.parquet"
    out_clean_csv = tmp_path / "cleaned.csv"
    out_sup_pq = tmp_path / "sup.parquet"
    out_sup_csv = tmp_path / "sup.csv"

    cleaned_df, sup_df = preprocess_data(
        str(raw_path),
        str(out_clean_pq),
        str(out_clean_csv),
        str(out_sup_pq),
        str(out_sup_csv),
    )

    for field in EXCLUDED_LEAKAGE_FIELDS:
        assert field not in cleaned_df.columns, f"Target leakage field '{field}' leaked into cleaned features!"
        assert field not in sup_df.columns, f"Target leakage field '{field}' leaked into supervised features!"

    # Allowed features must all be present
    for col in ["id", "number", "repository", "author_association", "created_at", "text", "target_class"]:
        assert col in cleaned_df.columns
        assert col in sup_df.columns


def test_deterministic_cleaning(tmp_path, synthetic_raw_dataset):
    """Ensure running preprocessing twice produces identical output."""
    raw_path = tmp_path / "raw.parquet"
    synthetic_raw_dataset.to_parquet(raw_path, index=False)

    c1, s1 = preprocess_data(
        str(raw_path),
        str(tmp_path / "c1.parquet"),
        str(tmp_path / "c1.csv"),
        str(tmp_path / "s1.parquet"),
        str(tmp_path / "s1.csv"),
        seed=42,
    )

    c2, s2 = preprocess_data(
        str(raw_path),
        str(tmp_path / "c2.parquet"),
        str(tmp_path / "c2.csv"),
        str(tmp_path / "s2.parquet"),
        str(tmp_path / "s2.csv"),
        seed=42,
    )

    pd.testing.assert_frame_equal(c1, c2)
    pd.testing.assert_frame_equal(s1, s2)


def test_real_research_dataset_integrity():
    """
    Validates the actual processed dataset files generated on disk
    to guarantee real research dataset compliance without synthetic substitution.
    """
    clean_path = "c:/Projects/data/processed/github_issues_cleaned.parquet"
    sup_path = "c:/Projects/data/processed/github_issues_supervised.parquet"

    assert os.path.exists(clean_path), "github_issues_cleaned.parquet does not exist"
    assert os.path.exists(sup_path), "github_issues_supervised.parquet does not exist"

    df_clean = pd.read_parquet(clean_path)
    df_sup = pd.read_parquet(sup_path)

    # Real counts check
    assert len(df_clean) == 3115, f"Expected 3,115 pure issues, got {len(df_clean)}"
    assert len(df_sup) == 1459, f"Expected 1,459 supervised issues, got {len(df_sup)}"

    # Check zero nulls in essential NLP features
    assert df_clean["text"].isna().sum() == 0
    assert (df_clean["text_char_len"] == 0).sum() == 0
    assert df_sup["text"].isna().sum() == 0

    # Check classes
    expected_classes = {"BUG", "FEATURE", "QUESTION", "DOCUMENTATION", "OTHER"}
    assert set(df_sup["target_class"].unique()) == expected_classes

    # Check exact class frequencies match EDA report
    counts = df_sup["target_class"].value_counts().to_dict()
    assert counts["BUG"] == 771
    assert counts["FEATURE"] == 615
    assert counts["QUESTION"] == 31
    assert counts["DOCUMENTATION"] == 25
    assert counts["OTHER"] == 17
