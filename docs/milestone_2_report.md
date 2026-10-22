# Milestone 2 Progress Report: Dataset Collection, Preprocessing & EDA

**Project Name:** GitHub Project Issue Classifier: AI-Powered Issue Classification, Prioritization & Routing System  
**Author:** Priyanshu Yadav (S25CSEU0725) | Batch-25  
**Institution:** College Progress Report / Mid-Term Viva Deliverable  
**Date:** October 15, 2026  
**Status:** Complete & Verified  

---

## 1. Executive Summary & Objectives
Milestone 2 establishes the end-to-end data engineering, quality audit, and exploratory analysis foundation for the GitHub Project Issue Classifier. Building directly upon the architecture defined in Milestone 1, this milestone implements a tested, deterministic data pipeline that acquires real GitHub issues, eliminates data quality anomalies, resolves source labels into a defensible multi-class taxonomy, prevents target leakage, and delivers empirical insights for upcoming machine learning model development.

### Primary Objectives Delivered:
1. **Source Verification:** Evaluated 4 candidate data sources and verified `helmo/github-issues` (Apache 2.0, 7,540 records, 12.66 MB Parquet) as the optimal source.
2. **Quality Audit:** Formally audited raw data, identifying and resolving a 100-row duplication block and separating 4,325 pull requests from 3,115 pure issues.
3. **Leakage Prevention:** Established a strict feature allowlist and completely excluded post-triage attributes (`state`, `comments`, `assignees`, `closed_at`, etc.).
4. **Target Taxonomy:** Developed a transparent mapping policy defining 5 defensible classes (`BUG`, `FEATURE`, `QUESTION`, `DOCUMENTATION`, `OTHER`), yielding 1,459 supervised records and 1,493 unlabeled records for unsupervised extensions.
5. **Exploratory Data Analysis:** Generated 11 publication-grade analytical visualizations and detailed interpretations.
6. **Split Readiness & Automated Tests:** Completed rigorous split-readiness validation and achieved 100% test pass rate across 7 pipeline unit and integration tests.

---

## 2. Data Source Verification & Provenance

### Evaluation of Candidate Data Sources
As mandated by the Milestone 2 protocol, four candidate repositories were inspected:

| Candidate Source | Accessibility | Size / Format | Usability & Provenance | Selected? |
| :--- | :--- | :--- | :--- | :--- |
| **`helmo/github-issues`** (Hugging Face) | Accessible (Public, No Auth) | 12.66 MB Parquet | Extracted directly from `huggingface/datasets` via GitHub API; Apache 2.0; 7,540 records. | **YES (Primary)** |
| **`noamaanMulla-03/datasets-issues`** (Hugging Face) | Accessible (Public, No Auth) | 39.33 MB Parquet | Contains 8,084 records; larger size due to full comment text strings; Apache 2.0. | Verified Alternative |
| **`ansnadeem/aic-framework`** (GitHub) | Accessible (Repo Only) | Code only | Only contains extraction notebooks; raw dataset not committed; subject to 60 req/hr rate limits. | Ineligible (No Data) |
| **`ansnadeem/github-top-repository-issues`** (Kaggle) | Requires Auth | Kaggle Archive | Requires user API credentials; automated ingestion blocked without credential setup. | Ineligible (Blocked) |

**Selected Primary Source:** [`helmo/github-issues`](https://huggingface.co/datasets/helmo/github-issues) was selected due to its stable Parquet format, manageable footprint, Apache 2.0 license, and alignment with Milestone 1's scale target (2,000–10,000 issues).

---

## 3. Data Ingestion & Quality Audit Findings

The automated audit script (`scripts/inspect_dataset.py`) revealed several critical characteristics of real GitHub data:

```
+-------------------------------------------------------------+
| Raw Parquet Records: 7,540 rows, 33 columns                 |
+-------------------------------------------------------------+
                            |
           [Deduplication on Stable ID]
                            v
+-------------------------------------------------------------+
| Unique Entities: 7,440 records (100 duplicate rows removed) |
+-------------------------------------------------------------+
                            |
           [Pull Request Filtering (is_pr == True)]
                            v
+-------------------------------------------------------------+
| Pure GitHub Issues: 3,115 records (4,325 PRs excluded)      |
+-------------------------------------------------------------+
                            |
           [Label Taxonomy Resolution & Validation]
                            v
+-----------------------------------+-------------------------+
| Supervised Dataset: 1,459 issues  | Unlabeled: 1,493 issues |
| BUG: 771 | FEATURE: 615           | Unmapped: 129 issues    |
| QUESTION: 31 | DOCS: 25 | OTHER: 17| Conflicting: 34 issues  |
+-----------------------------------+-------------------------+
```

### Key Findings:
1. **Raw Duplication Block:** Exactly 100 rows (indices 0..99) were identical to rows 100..199 in the raw Parquet dump. Deduplication on `id` removed this artifact deterministically.
2. **Issue vs. Pull Request Conflation:** 58.1% (4,325 records) of the raw dataset represented pull requests rather than user issues. Because PR descriptions contain commit messages and code review summaries rather than problem reports, including them would severely corrupt issue classification.
3. **Text Completeness:** 100% of pure issues have complete titles (0 nulls). Body descriptions had only 10 nulls (0.32%) and 20 empty strings, all normalized to empty strings during cleaning. Combining title and body provided 100% text coverage.

---

## 4. Target Taxonomy & Data Leakage Prevention

### Target Classification Taxonomy
To ensure scientifically valid ground truth, source labels were mapped using transparent, documented rules:

- **`BUG` (771 issues, 52.84%):** `bug`, `dataset bug`, `metric bug`.
- **`FEATURE` (615 issues, 42.15%):** `enhancement`, `dataset request`, `dataset contribution`.
- **`QUESTION` (31 issues, 2.12%):** `question`, `generic discussion`, `Dataset discussion`, `Metric discussion`.
- **`DOCUMENTATION` (25 issues, 1.71%):** `documentation`.
- **`OTHER` (17 issues, 1.17%):** `maintenance`, `refactoring`.

Issues with only orthogonal tags (`good first issue`, `vision`, `streaming`) were marked as `UNMAPPED` (129 issues). Issues with contradictory tags (`[bug, enhancement]`) were marked as `CONFLICTING` (34 issues) and excluded from single-label training.

### Rigorous Data Leakage Prevention
To simulate genuine real-time inference (when an issue is first submitted to GitHub), all post-triage fields were strictly excluded from features:

- **Creation-Time Feature Allowlist:** `clean_title`, `clean_body`, `text` (combined), `author_association`, `created_at`, `repository`, plus derived text lengths.
- **Excluded Leaky Fields:** `state`, `closed_at`, `closed_by`, `state_reason`, `comments`, `assignee`, `assignees`, `milestone`, `reactions`, `locked`, `labels`.

---

## 5. Summary of Exploratory Data Analysis (EDA)

All 11 figures were programmatically generated from the actual loaded dataset (`reports/figures/`):

1. **Figure 1 (Filtering Pipeline):** Demonstrates the step-by-step reduction from 7,540 raw rows to 1,459 clean supervised issues.
2. **Figure 2 (Class Frequency & Imbalance):** Shows the 45:1 imbalance ratio between `BUG` (771) and `OTHER` (17). Top two classes represent 95.0% of data.
3. **Figure 3 (Original Label Frequencies):** Displays the distribution of the top 20 source tags in the repository.
4. **Figure 4 (Repository Entity Breakdown):** Visualizes the relative proportions of PRs, pure issues, and supervised records.
5. **Figure 5 (Author Association Breakdown):** Reveals that external casual users (`author_association == NONE`) submit the overwhelming majority of bugs, whereas maintainers submit a higher proportion of features.
6. **Figure 6 (Missingness Analysis):** Documents that triage fields (`assignee`, `milestone`) are 89-99% missing, whereas title and body text have essentially 0% missingness.
7. **Figure 7 (Text Length Distributions):** Shows title length follows a bell curve (mean 67.4 chars, median 61 chars), while body word counts are right-skewed (median 98 words).
8. **Figure 8 (Creation Timeline):** Tracks monthly issue volume from 2020 through 2025, identifying peak activity in 2022-2023.
9. **Figure 9 (Distinctive N-Grams):** Proves strong vocabulary divergence between bugs (`traceback call`, `error message`, `fails when`) and features (`add support`, `feature request`, `allow user`).
10. **Figure 10 (Duplicate Analysis):** Quantifies exact row duplicates and near-identical titles.
11. **Figure 11 (Candidate vs. Clean Class Counts):** Compares raw tag mentions against clean filtered supervised targets.

---

## 6. Dataset Split Readiness & Evaluation Strategy

1. **Stratification Feasibility:** With 17 samples in the smallest class (`OTHER`), 5-fold stratified cross-validation is fully supported (each fold contains 3-4 instances of `OTHER`, ~123 of `FEATURE`, ~154 of `BUG`).
2. **Contamination Risk:** 483 near-duplicate title pairs (cosine similarity >= 0.85) were identified. To prevent optimistic evaluation bias, near-duplicate issues must be clustered within the same fold or removed from the test set.
3. **Temporal Split Benchmark:** Chronological splitting (80% past / 20% future at cutoff date `2023-01-26`) revealed temporal shift: earlier years were bug-heavy, while recent years are feature-heavy. Temporal evaluation will be included alongside stratified k-fold in Milestone 3.
4. **Metric Mandate:** Overall accuracy is misleading due to class imbalance. All future evaluations must report **Macro-F1, Balanced Accuracy, and per-class Precision/Recall**.

---

## 7. Verification & Automated Test Results

The automated test suite (`tests/test_data_pipeline.py`) was executed via `pytest`:

```
tests/test_data_pipeline.py::test_loading_chosen_format PASSED           [ 14%]
tests/test_data_pipeline.py::test_missing_text_normalization PASSED      [ 28%]
tests/test_data_pipeline.py::test_label_mapping_and_conflicts PASSED     [ 42%]
tests/test_data_pipeline.py::test_duplicate_and_pull_request_filtering PASSED [ 57%]
tests/test_data_pipeline.py::test_leakage_fields_exclusion PASSED        [ 71%]
tests/test_data_pipeline.py::test_deterministic_cleaning PASSED          [ 85%]
tests/test_data_pipeline.py::test_real_research_dataset_integrity PASSED [100%]

============================== 7 passed in 1.17s ==============================
```

All 7 tests passed with zero failures.

---

## 8. Next Steps for Milestone 3
1. **Feature Engineering:** Implement TF-IDF vectorization with sublinear term frequency and n-gram ranges (1, 2).
2. **Baseline Classifiers:** Train Multinomial Naive Bayes and Logistic Regression baselines.
3. **Class Imbalance Mitigation:** Compare balanced class weights (`class_weight='balanced'`) vs. SMOTE oversampling.
4. **Evaluation:** Evaluate models using 5-fold Stratified Cross-Validation reporting Macro-F1 and confusion matrices.

