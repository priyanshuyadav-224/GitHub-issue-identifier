# Milestone 2 Requirement Checklist & Verification Evidence

**Project:** GitHub Project Issue Classifier  
**Candidate:** Priyanshu Yadav (S25CSEU0725) | Batch-25  
**Review Date:** October 15, 2026  
**Overall Status:** **100% COMPLETE**  

---

## Task-by-Task Verification Matrix

| Task ID | Description | Status | Concrete Evidence / File Artifact |
| :--- | :--- | :--- | :--- |
| **Task 1** | **Data Source Verification** | **COMPLETE** | Inspected 4 candidate sources. Verified `helmo/github-issues` (Apache 2.0, 7,540 rows, 33 columns). Preserved raw Parquet file locally at `data/raw/github_issues_raw.parquet` (SHA-256: `09453eefae39e45a...`). Documented in `docs/dataset_card.md` and `data/raw/README.md`. |
| **Task 2** | **Data Ingestion Pipeline** | **COMPLETE** | Developed deterministic script `scripts/inspect_dataset.py` and `scripts/preprocess_dataset.py` with configurable CLI arguments (`--input`, `--output-cleaned`, `--output-supervised`), strict error handling, and deterministic ordering. |
| **Task 3** | **Data Quality Audit** | **COMPLETE** | Identified 100 duplicate row entries in raw data (indices 0..99 vs 100..199). Calculated PR counts (4,325) vs pure issues (3,115). Audited column missingness, text completeness (0% title nulls), and timestamp bounds (`2020-04-14` to `2025-06-13`). Saved to `reports/dataset_audit.json` and `reports/dataset_audit.md`. |
| **Task 4** | **Target Definition & Taxonomy** | **COMPLETE** | Mapped raw labels into 5 defensible classes: `BUG` (771), `FEATURE` (615), `QUESTION` (31), `DOCUMENTATION` (25), `OTHER` (17). Documented unmapped triage tags (129), unlabelled issues (1,493), and conflicting multi-label issues (34). Excluded heuristic text-inferred labels. |
| **Task 5** | **Preprocessing Pipeline** | **COMPLETE** | Implemented `scripts/preprocess_dataset.py`. Stripped HTML/Markdown images while preserving code symbols (`__iter__`), error logs, and version tokens. Combined title and body. Enforced strict feature allowlist and excluded all post-triage leakage fields (`state`, `comments`, `assignee`, `milestone`, `reactions`). Generated `data/processed/github_issues_cleaned.parquet` and `data/processed/github_issues_supervised.parquet`. |
| **Task 6** | **Exploratory Data Analysis** | **COMPLETE** | Generated 11 distinct publication-grade charts under `reports/figures/` (from actual dataset, not hard-coded values). Created comprehensive analytical summary with ML implications in `reports/eda_summary.md`. Provided interactive notebook `notebooks/milestone_2_eda.ipynb`. |
| **Task 7** | **Dataset Split Readiness** | **COMPLETE** | Validated 5-fold and 10-fold stratified split feasibility for all 5 classes. Identified 483 near-duplicate title pairs (cosine similarity >= 0.85). Benchmarked 80/20 temporal split (`cutoff: 2023-01-26`). Documented in `docs/split_readiness_report.md`. |
| **Task 8** | **Automated Testing Suite** | **COMPLETE** | Created `tests/test_data_pipeline.py`. Included synthetic test fixtures clearly labeled as synthetic. Tested loading, missing text normalization, label mapping, duplicate handling, PR filtering, target leakage exclusion, deterministic cleaning, and real dataset integrity. 7 of 7 tests passed (0 failures). |
| **Task 9** | **Required Deliverables** | **COMPLETE** | All specified deliverables exist, are non-empty, and are verified: `data/raw/README.md`, `data/processed/README.md`, `scripts/`, `notebooks/`, `reports/`, `docs/`, `tests/`, `README.md`. |
| **Task 10**| **Execution & Final Validation**| **COMPLETE** | Pipeline fully executed in workspace; all checksums and row counts verified across code, reports, figures, and tests. |

---

## Quantitative Evidence Checklist

- [x] Raw Row Count: **7,540**
- [x] Raw Column Count: **33**
- [x] Duplicate Rows Removed: **100**
- [x] Pull Requests Filtered: **4,325**
- [x] Pure GitHub Issues: **3,115**
- [x] Supervised Labeled Issues: **1,459**
  - [x] `BUG`: **771** (52.84%)
  - [x] `FEATURE`: **615** (42.15%)
  - [x] `QUESTION`: **31** (2.12%)
  - [x] `DOCUMENTATION`: **25** (1.71%)
  - [x] `OTHER`: **17** (1.17%)
- [x] Unlabeled Issues Reserved for Unsupervised Learning: **1,493** (47.93%)
- [x] Conflicting Multi-Label Issues Excluded: **34** (1.09%)
- [x] Generated EDA Figures: **11 charts** (`reports/figures/`)
- [x] Automated Tests Passed: **7 of 7** (`pytest -v tests/`)
