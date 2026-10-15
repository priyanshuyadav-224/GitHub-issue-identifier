# Dataset Split Readiness Assessment Report

**Author:** Priyanshu Yadav (S25CSEU0725)  
**Project:** GitHub Project Issue Classifier  
**Target Evaluated Dataset:** `c:/Projects/data/processed/github_issues_supervised.parquet` (1,459 supervised records)  

## 1. Class Frequencies & Stratification Feasibility

| Target Class | Total Count | % of Dataset | 5-Fold Min Count/Fold | 80/20 Test Set Expected |
| :--- | :--- | :--- | :--- | :--- |
| **`BUG`** | 771 | 52.84% | ~154 | ~154 |
| **`FEATURE`** | 615 | 42.15% | ~123 | ~123 |
| **`QUESTION`** | 31 | 2.12% | ~6 | ~6 |
| **`DOCUMENTATION`** | 25 | 1.71% | ~5 | ~5 |
| **`OTHER`** | 17 | 1.17% | ~3 | ~3 |

- **Minimum Class Size:** `OTHER` has 17 instances.
- **5-Fold Stratified Split Feasible:** `YES` (Minimum class 17 >= 5 folds).
- **10-Fold Stratified Split Feasible:** `YES` (Minimum class 17 >= 10 folds).

## 2. Duplicate & Near-Duplicate Contamination Risk

- **Exact Title Matches:** 2 issues share identical lowercase titles (e.g. repeated error messages like `ImportError: cannot import name ...`).
- **Near-Duplicate Pairs (TF-IDF Cosine Similarity >= 0.85):** 483 pairs detected.
- **Contamination Hazard:** If near-duplicate issue reports are split arbitrarily between train and test partitions, the model can memorize specific title phrases and inflate test accuracy. A deduplicated stratified split or title-grouped split must be enforced during Milestone 3 cross-validation.

## 3. Split Strategy Comparative Analysis

| Split Strategy | Feasibility | Strengths | Limitations | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Random Stratified Split** | **High** | Preserves minority class proportions (QUESTION, DOCUMENTATION, OTHER) in both train and test. | May allow near-duplicate issues into test set without grouping. | **Primary Evaluation Scheme** (with near-duplicate grouping). |
| **Temporal (Time-Based) Split** | **Moderate** | Perfectly reflects production deployment: train on past issues, test on future incoming issues. | Extreme minority classes (`OTHER`: 17) may be sparsely represented in recent test window. | **Secondary Benchmark Scheme** for temporal robustness. |
| **Repository-Grouped Split** | **N/A** | Tests cross-repo generalizability. | Current verified dataset is from single monolithic repository (`huggingface/datasets`). | Deferred to multi-repo extensions in Milestone 5. |

## 4. Temporal Split Simulation (80% Past / 20% Future)

- **Cutoff Timestamp:** `2023-01-26 12:28:09+00:00`
- **Past (Train) Partition:** 1,167 issues (`2020-05-15 01:12:06+00:00` to `2023-01-26 12:28:09+00:00`)
- **Future (Test) Partition:** 292 issues (`2023-01-26 12:28:09+00:00` to `2025-06-13 00:40:48+00:00`)

Class distribution across temporal split:

| Class | Train Partition Count | Test Partition Count |
| :--- | :--- | :--- |
| `BUG` | 706 | 65 |
| `DOCUMENTATION` | 14 | 11 |
| `FEATURE` | 411 | 204 |
| `OTHER` | 6 | 11 |
| `QUESTION` | 30 | 1 |

## 5. Strict Data Leakage & Evaluation Rules for Milestone 3

1. **No Fitting on Full Dataset:** TF-IDF vectorizers, scalers, and dimensionality reducers (PCA) must only be fit on `X_train`. Calling `.fit_transform()` on the full dataset before splitting is strictly forbidden.
2. **Target Leakage Exclusion:** Input feature vectors must contain only creation-time fields (`text`, `clean_title`, `clean_body`, `author_association`, `created_at`). Excluded fields (`state`, `comments`, `assignee`, `milestone`, `reactions`) must never enter `X`.
3. **Macro-Averaging:** Due to the 45:1 class imbalance between `BUG` (771) and `OTHER` (17), all milestone scorecards must report **Macro-F1, Balanced Accuracy, and per-class metrics** in addition to overall accuracy.
