# GitHub Project Issue Classifier

**AI-Powered Issue Classification, Prioritization & Routing System**  
*Statistical Machine Learning & Natural Language Processing*  
**Candidate:** Priyanshu Yadav (`S25CSEU0725`) | Batch-25  

---

## 1. Executive Summary
Software engineering teams receive hundreds of incoming GitHub issues across bug reports, feature enhancements, documentation clarifications, and questions. Manually triaging, tagging, and routing each ticket demands repetitive engineering effort.

This project develops an interpretable, reproducible machine learning system that learns from historical GitHub issue titles and descriptions to predict issue categories, confidence scores, and routing recommendations. Rather than wrapping proprietary LLM APIs as black boxes, this system implements classical statistical machine learning (TF-IDF, Multinomial Naive Bayes, Logistic Regression, Linear SVM, Random Forest, and K-Means clustering) to maximize mathematical visibility, eliminate API costs, and achieve sub-millisecond inference latency (0.088 - 1.286 ms).

---

## 2. Statistical Machine Learning Benchmarks (Test Set N = 292)

All models were evaluated on an identical held-out test partition (stratified 80/20 train/test split, random seed = 42) across 5 target classes (`BUG`, `FEATURE`, `QUESTION`, `DOCUMENTATION`, `OTHER`):

| Model Architecture | Test Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Weighted-F1 | 5-Fold CV Acc | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear SVM (Calibrated)** | **88.70%** | **0.7060** | 0.5362 | **57.40%** | 87.42% | **90.58%** | 1.286 ms |
| **Logistic Regression (L2)** | 86.30% | 0.5458 | **0.6142** | 56.97% | 86.05% | 88.69% | **0.102 ms** |
| **Multinomial Naive Bayes** | 84.93% | 0.3387 | 0.3581 | 34.77% | 82.78% | 87.15% | **0.088 ms** |
| **Random Forest (150 trees)** | **89.38%** | 0.4909 | 0.4543 | 46.56% | **87.71%** | 88.69% | 35.776 ms |

### Key Benchmark Observations:
1. **Linear SVM (Calibrated)** achieves the strongest general balance (88.70% accuracy, 70.60% macro-precision, 57.40% macro-F1), constructing an optimal maximum-margin hyperplane in the 5,000-dimensional sparse TF-IDF space.
2. **Logistic Regression (L2 Balanced)** delivers the highest minority class recall (0.6142 macro-recall) with balanced class weights ($w_k = \frac{N}{K \cdot N_k}$), making it ideal for high-recall safety-critical triage.
3. **Multinomial Naive Bayes** is exceptionally fast (0.088 ms per query), but struggles on extreme minority classes (`QUESTION`, `DOCUMENTATION`, `OTHER`) due to feature independence assumptions under severe class imbalance (45:1).
4. **Random Forest** captures non-linear feature interactions, achieving 89.38% overall accuracy, but suffers from higher inference latency (35.78 ms).

---

## 3. Unsupervised Latent Topic Discovery (K-Means, k = 5)
Applied to 1,493 unlabeled historical issues to uncover latent problem categories without supervision:

- **Cluster 0 (Data Loaders & Streaming):** `dataset`, `request`, `add`, `feature`, `gem`, `datasets`, `stream`, `loader`
- **Cluster 1 (Build, CI & Testing):** `broken`, `ci`, `windows`, `tests`, `fails`, `release`, `ruff`, `pytest`
- **Cluster 2 (Dataset Viewer & Web):** `dataset viewer`, `viewer`, `issue`, `dataset`, `error`, `nlp`, `preview`
- **Cluster 3 (Cloud Hub & Authentication):** `dataset`, `error`, `datasets`, `loading`, `map`, `load`, `support`, `auth`
- **Cluster 4 (Core API & Execution):** `load_dataset`, `work`, `does work`, `doesn work`, `error`, `typing`, `pyarrow`

---

## 4. Project Directory Structure

```text
├── data/
│   ├── raw/
│   │   ├── README.md                           # Data provenance & source documentation
│   │   └── github_issues_raw.parquet           # Raw source dataset (7,540 records)
│   └── processed/
│       ├── README.md                           # Schema documentation & mapping rules
│       ├── github_issues_cleaned.parquet       # Deduplicated pure issues (N = 3,115)
│       └── github_issues_supervised.parquet    # Labeled supervised dataset (N = 1,459)
├── models/
│   ├── model_suite.joblib                      # Serialized multi-model inference suite
│   ├── tfidf_vectorizer.joblib                 # Fitted 5,000-dim TF-IDF feature extractor
│   └── kmeans_model.joblib                     # Fitted K-Means (k=5) clustering model
├── reports/
│   ├── figures/                                # 14 publication-grade analytical plots
│   │   ├── fig01_filtering_pipeline_counts.png
│   │   ├── fig02_class_frequency_imbalance.png
│   │   ├── fig12_model_comparison_metrics.png
│   │   ├── fig13_confusion_matrices.png
│   │   └── fig15_kmeans_cluster_projection.png
│   ├── model_evaluation_metrics.json           # Complete machine-readable benchmark report
│   ├── S25CSEU0725_Milestone_3_Report.pdf      # Formal technical report (< 10 MB)
│   └── S25CSEU0725_Milestone_3_Presentation.pptx # Slide deck (14 slides, < 10 MB)
├── scripts/
│   ├── inspect_dataset.py                      # Data quality & schema audit
│   ├── preprocess_dataset.py                   # Preprocessing & leakage firewall
│   ├── run_eda.py                              # Visualizations & statistics generator
│   ├── train_and_evaluate_models.py            # Training, cross-validation & benchmark suite
│   └── generate_milestone3_artifacts.py        # PDF report & PPTX slide deck generator
├── templates/
│   └── index.html                              # Enterprise triage & benchmark dashboard
├── tests/
│   └── test_data_pipeline.py                   # Automated pipeline test suite (7 passed)
├── app.py                                      # Production Flask REST API & web service
├── .gitignore                                  # Git exclusion configuration
└── README.md                                   # Comprehensive project documentation
```

---

## 5. Target Leakage Prevention Architecture

To guarantee scientific validity, the model only consumes attributes that exist **at the exact millisecond the author submits the issue**:

| Feature Category | Included (Allowlist) | Excluded (Quarantined Post-Triage Attributes) |
| :--- | :--- | :--- |
| **Author Text** | `clean_title`, `clean_body`, `text` (combined) | `labels` (ground truth target) |
| **User Context** | `author_association` (`MEMBER`, `NONE`) | `assignee`, `assignees` (maintainer assignment) |
| **Temporal** | `created_at` (submission timestamp) | `closed_at`, `closed_by`, `state` (`closed`/`open`) |
| **Activity** | Character & word lengths | `comments` count, comment text, `reactions` (+1) |
| **Admin** | Traceability IDs (`id`, `number`, `html_url`) | `milestone`, `locked`, `state_reason` |

---

## 6. Quickstart: Reproducing the Entire Pipeline

### 1. Environment Setup
```powershell
pip install pandas numpy pyarrow scikit-learn matplotlib seaborn pytest reportlab python-pptx flask
```

### 2. Preprocess Dataset & Apply Target Leakage Firewall
```powershell
python scripts/preprocess_dataset.py
```
*Outputs: `data/processed/github_issues_cleaned.parquet` (3,115 pure issues) and `github_issues_supervised.parquet` (1,459 supervised issues).*

### 3. Train & Benchmark Statistical ML Models
```powershell
python scripts/train_and_evaluate_models.py
```
*Trains Naive Bayes, Logistic Regression, Linear SVM, Random Forest, and K-Means. Outputs evaluation matrices and charts.*

### 4. Run Automated Tests
```powershell
pytest -v tests/
```
*Verification: 7 passed, 100% test success.*

### 5. Launch Interactive Production Web Console
```powershell
python app.py
```
*Service runs locally at: `http://127.0.0.1:5000`.*
