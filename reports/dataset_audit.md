# Dataset Quality Audit Report

**Generated:** 2026-10-22T03:37:26.745605+00:00  
**Dataset Source:** `Hugging Face Datasets: helmo/github-issues (extracted from huggingface/datasets repo)`  
**License:** `Apache 2.0`  
**Raw File:** `c:/Projects/data/raw/github_issues_raw.parquet` (12,659,096 bytes, SHA-256: `09453eefae39e45a...`)  

## 1. Summary Record Counts

| Metric | Value | Notes |
| :--- | :--- | :--- |
| **Raw Total Rows** | 7,540 | Total rows in source parquet |
| **Raw Total Columns** | 33 | GitHub REST API issue schema fields |
| **Unique Record IDs** | 7,440 | Deduplication reveals 100 duplicated rows |
| **Duplicate ID Rows** | 100 | First 100 rows fetched twice in raw dump |
| **Unique Deduplicated Records** | 7,440 | Net unique entities in dataset |
| **Pull Requests (PRs)** | 4,325 | Excluded from issue classification task |
| **Pure GitHub Issues** | 3,115 | Actual target issues for triage/classification |

## 2. Issues Text Quality

| Field | Missing | Empty / Whitespace | Usability |
| :--- | :--- | :--- | :--- |
| `title` | 0 (0.0%) | 0 (0.0%) | 100% complete across all pure issues |
| `body` | 10 (0.32%) | 20 (0.64%) | Handled via missing-value normalization |
| `text` (combined) | 0 (0.0%) | 0 (0.0%) | Title + Body provides 100% text coverage |

## 3. Label Distribution in Pure Issues

- **Unlabeled Issues (0 labels):** 1,493 (47.93%)
- **Single-Label Issues (1 label):** 1,416 (45.46%)
- **Multi-Label Issues (>1 labels):** 206 (6.61%)

### Top 20 GitHub Labels in Pure Issues

| Rank | Label Name | Frequency | Project Category Candidate |
| :--- | :--- | :--- | :--- |
| 1 | `bug` | 709 | Normalized in Task 4 |
| 2 | `enhancement` | 480 | Normalized in Task 4 |
| 3 | `dataset request` | 162 | Normalized in Task 4 |
| 4 | `dataset-viewer` | 91 | Normalized in Task 4 |
| 5 | `dataset bug` | 74 | Normalized in Task 4 |
| 6 | `good first issue` | 53 | Normalized in Task 4 |
| 7 | `duplicate` | 30 | Normalized in Task 4 |
| 8 | `documentation` | 28 | Normalized in Task 4 |
| 9 | `generic discussion` | 27 | Normalized in Task 4 |
| 10 | `question` | 24 | Normalized in Task 4 |
| 11 | `vision` | 24 | Normalized in Task 4 |
| 12 | `streaming` | 22 | Normalized in Task 4 |
| 13 | `speech` | 19 | Normalized in Task 4 |
| 14 | `maintenance` | 17 | Normalized in Task 4 |
| 15 | `nlp-viewer` | 14 | Normalized in Task 4 |
| 16 | `good second issue` | 13 | Normalized in Task 4 |
| 17 | `hacktoberfest` | 12 | Normalized in Task 4 |
| 18 | `metric bug` | 8 | Normalized in Task 4 |
| 19 | `wontfix` | 7 | Normalized in Task 4 |
| 20 | `Dataset discussion` | 6 | Normalized in Task 4 |

## 4. Column Missingness Audit

| Column | Missing Count | Missing % |
| :--- | :--- | :--- |
| `assignee` | 6,761 | 89.67% |
| `milestone` | 7,474 | 99.12% |
| `closed_at` | 954 | 12.65% |
| `type` | 7,540 | 100.0% |
| `active_lock_reason` | 7,540 | 100.0% |
| `draft` | 3,164 | 41.96% |
| `pull_request` | 3,164 | 41.96% |
| `body` | 355 | 4.71% |
| `closed_by` | 947 | 12.56% |
| `performed_via_github_app` | 7,540 | 100.0% |
| `state_reason` | 5,213 | 69.14% |
| `sub_issues_summary` | 4,376 | 58.04% |

## 5. Temporal Range & Validations

- **`created_at`**: Valid records: 7,540, Range: `2020-04-14 10:18:02+00:00` to `2025-06-13 09:02:24+00:00`
- **`updated_at`**: Valid records: 7,540, Range: `2020-04-27 16:04:17+00:00` to `2025-06-13 10:38:04+00:00`
- **`closed_at`**: Valid records: 6,586, Range: `2020-04-14 12:01:40+00:00` to `2025-06-13 00:44:27+00:00`
