# GitHub Project Issue Classifier

**AI-Powered Issue Classification, Prioritization & Routing System**  
*NLP + Statistical Machine Learning + Software Engineering*  
**Candidate:** Priyanshu Yadav (`S25CSEU0725`) | Batch-25  

---

## 📌 Executive Summary
Software engineering teams receive hundreds of incoming GitHub issues across bug reports, feature enhancements, documentation clarifications, and questions. Manually triaging, tagging, and routing each ticket demands repetitive engineering effort.

This project develops an interpretable, reproducible machine learning system that learns from historical GitHub issue titles and descriptions to predict issue categories, confidence scores, and routing recommendations. Rather than wrapping proprietary LLM APIs as black boxes, the first iteration deliberately uses classical statistical machine learning (TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM, and K-Means clustering) to maximize mathematical visibility, eliminate API costs, and achieve sub-millisecond inference latency.

---

## 📂 Project Directory Structure

```text
├── data/
│   ├── raw/
│   │   ├── README.md                           # Provenance & acquisition guide
│   │   └── github_issues_raw.parquet           # Raw source dump (7,540 records)
│   └── processed/
│       ├── README.md                           # Cleaned dataset documentation & schema
│       ├── github_issues_cleaned.parquet       # All pure issues (N = 3,115)
│       ├── github_issues_cleaned.csv           # CSV export of cleaned issues
│       ├── github_issues_supervised.parquet    # Labeled supervised dataset (N = 1,459)
│       └── github_issues_supervised.csv        # CSV export of supervised dataset
├── docs/
│   ├── milestone_1_report.pdf                  # Preserved Milestone 1 architecture PDF
│   ├── milestone_2_report.md                   # Milestone 2 comprehensive progress report
│   ├── milestone_2_checklist.md                # Task-by-task verification checklist
│   ├── split_readiness_report.md               # Stratified & temporal split feasibility
│   ├── dataset_card.md                         # Standard ML dataset card
│   └── viva_prep.md                            # Comprehensive viva & exam prep guide
├── notebooks/
│   └── milestone_2_eda.ipynb                   # Interactive EDA Jupyter notebook
├── reports/
│   ├── dataset_audit.json                      # Machine-readable audit output
│   ├── dataset_audit.md                        # Human-readable audit report
│   ├── eda_summary.md                          # Written EDA findings & ML implications
│   ├── milestone_2_report.pdf                  # Formal PDF progress report
│   ├── milestone_2_presentation.pptx           # Presentation slide deck with charts
│   └── figures/                                # 11 Generated publication-grade plots
│       ├── fig01_filtering_pipeline_counts.png
│       ├── fig02_class_frequency_imbalance.png
│       ├── fig03_original_label_frequency.png
│       ├── fig04_records_by_repository.png
│       ├── fig05_class_distribution_by_repo.png
│       ├── fig06_missingness_analysis.png
│       ├── fig07_text_length_distributions.png
│       ├── fig08_issue_creation_timeline.png
│       ├── fig09_ngram_frequency_analysis.png
│       ├── fig10_duplicate_analysis.png
│       └── fig11_class_counts_before_after_filtering.png
├── scripts/
│   ├── inspect_dataset.py                      # Data quality & schema audit script
│   ├── preprocess_dataset.py                   # Preprocessing & leakage firewall script
│   ├── run_eda.py                              # Visualizations & statistics generator
│   ├── assess_split_readiness.py               # Split evaluation & contamination audit
│   └── generate_pdf_and_pptx.py                # ReportLab PDF & PPTX generator
├── tests/
│   └── test_data_pipeline.py                   # 7 automated pipeline unit/integration tests
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart: Reproducing Milestone 2 in 4 Steps

### 1. Environment Setup
Ensure Python 3.10+ is installed with required data science packages:
```powershell
pip install pandas numpy pyarrow scikit-learn matplotlib seaborn pytest reportlab python-pptx
```

### 2. Step 1: Run Data Quality Audit
Audits the raw Parquet file, detects duplicate IDs, classifies PRs vs issues, and computes missingness:
```powershell
python scripts/inspect_dataset.py
```
*Outputs: `reports/dataset_audit.json` and `reports/dataset_audit.md`*

### 3. Step 2: Run Deterministic Preprocessing Pipeline
Deduplicates records, filters PRs, cleans text while preserving code tokens, excludes leaky post-triage fields, and maps labels:
```powershell
python scripts/preprocess_dataset.py
```
*Outputs: `data/processed/github_issues_cleaned.parquet` (3,115 issues) and `github_issues_supervised.parquet` (1,459 issues)*

### 4. Step 3: Run Exploratory Data Analysis (EDA)
Generates 11 publication-grade charts and the written interpretation:
```powershell
python scripts/run_eda.py
```
*Outputs: 11 PNG charts in `reports/figures/` and `reports/eda_summary.md`*

### 5. Step 4: Run Automated Tests
Executes the comprehensive pytest suite verifying loading, missing text, label mapping, leakage exclusion, and dataset integrity:
```powershell
pytest -v tests/
```
*Expected output: `7 passed` (100% pass rate).*

---

## 📊 Milestone 2 Pipeline Statistics

```
Raw Ingested Records (7,540)
       │
       ▼  [-100 Duplicate Rows Removed (indices 0..99 == 100..199)]
Unique Entities (7,440)
       │
       ▼  [-4,325 Pull Requests Excluded]
Pure GitHub Issues (3,115)
       │
       ▼  [Target Taxonomy Mapping & Resolution]
┌───────────────────────────────────────┬──────────────────────────────────────┐
│  Supervised Labeled Dataset (1,459)   │  Unsupervised Exploration (1,656)    │
│  • BUG:           771 (52.84%)        │  • UNLABELED:    1,493 (47.93%)      │
│  • FEATURE:       615 (42.15%)        │  • UNMAPPED:       129 ( 4.14%)      │
│  • QUESTION:       31 ( 2.12%)        │  • CONFLICTING:     34 ( 1.09%)      │
│  • DOCUMENTATION:  25 ( 1.71%)        │                                      │
│  • OTHER:          17 ( 1.17%)        │                                      │
└───────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 🛡️ Target Leakage Prevention Architecture

To guarantee scientific validity, the model only uses features that exist **at the exact second the author submits the issue**:

| Feature Category | Included (Allowlist) | Excluded (Leaky Post-Triage Attributes) |
| :--- | :--- | :--- |
| **Author Text** | `clean_title`, `clean_body`, `text` (combined) | `labels` (ground truth target) |
| **User Context** | `author_association` (`MEMBER`, `NONE`) | `assignee`, `assignees` (maintainer assignment) |
| **Temporal** | `created_at` (submission timestamp) | `closed_at`, `closed_by`, `state` (`closed`/`open`) |
| **Activity** | Character & word lengths | `comments` count, comment text, `reactions` (+1) |
| **Admin** | Traceability IDs (`id`, `number`, `html_url`) | `milestone`, `locked`, `state_reason` |

---

## 🎓 Viva & Exam Preparation
A dedicated, comprehensive viva preparation document has been compiled in [`docs/viva_prep.md`](docs/viva_prep.md). It includes:
- 60-second elevator pitch
- End-to-end architecture breakdown
- Mathematical formulas for TF-IDF, Naive Bayes, Laplace Smoothing, and Macro-F1
- Top 15 examiner questions with defensible model answers
- 30-second live terminal demonstration commands
