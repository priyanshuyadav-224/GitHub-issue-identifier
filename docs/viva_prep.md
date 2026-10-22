# Comprehensive Viva & Mid-Term Presentation Preparation Guide

**Project Title:** GitHub Project Issue Classifier (AI-Powered Issue Classification, Prioritization & Routing System)  
**Candidate:** Priyanshu Yadav (S25CSEU0725) | Batch-25  
**Course:** NLP + Statistical Machine Learning + Software Engineering  
**Milestone:** Milestone 2 (Dataset Collection, Preprocessing & EDA)  

---

## 1. Project Elevator Pitch (What to Say in the First 60 Seconds)

> *"Good morning, esteemed examiners. My project is the **GitHub Project Issue Classifier**, an AI-powered engineering system designed to automatically classify, prioritize, and route incoming software issue tickets on GitHub.*
> 
> *In large open-source repositories and engineering teams, hundreds of tickets arrive daily—ranging from critical bug reports and feature requests to documentation typos and user questions. Manually triaging each ticket wastes valuable engineering hours.*
> 
> *In **Milestone 1**, I established the system architecture, mathematical foundation, and classical ML learning path. In **Milestone 2**, which I am presenting today, I implemented the complete data engineering pipeline: I verified and acquired 7,540 real GitHub records from the foundational `huggingface/datasets` repository, diagnosed and resolved a 100-row duplication block, filtered out 4,325 pull requests, established a leak-free 5-class target taxonomy (Bug, Feature, Question, Documentation, Other) yielding 1,459 supervised issues, generated 11 analytical EDA visualizations, and passed 100% of our automated unit tests.*
> 
> *Unlike naive black-box LLM wrappers, my project uses transparent statistical ML—starting with TF-IDF, Naive Bayes, Logistic Regression, Linear SVM, and K-Means clustering—ensuring complete mathematical interpretability, low latency, zero API costs, and robust data integrity."*

---

## 2. End-to-End System Architecture: How the Project Works

The system operates across 8 planned development weeks and 5 core layers:

```
+-----------------------------------------------------------------------------------+
|                              1. INGESTION LAYER                                   |
|   GitHub REST API / Hugging Face Parquet Dump (7,540 raw records)                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        2. AUDIT & PREPROCESSING PIPELINE                          |
|   • Deduplication on ID (7,540 -> 7,440 unique)                                   |
|   • Pull Request Exclusion (4,325 PRs dropped -> 3,115 pure issues)               |
|   • Text Cleaning: Normalize Markdown/HTML, preserve code tokens (__iter__, v1.2) |
|   • Data Leakage Firewall: Drop state, closed_at, comments, assignees, milestones  |
|   • Target Resolution: Maps labels -> BUG, FEATURE, QUESTION, DOCS, OTHER         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                     3. FEATURE EXTRACTION & MODELING (M3 - M5)                     |
|   • Text Representation: Sublinear TF-IDF (Unigrams + Bigrams, 5,000 features)    |
|   • Classifiers: MultinomialNB, Logistic Regression, Linear SVM, Random Forest    |
|   • Unsupervised Clustering: K-Means (k=5) + PCA on 1,493 unlabelled issues       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        4. ROUTING & EXPLAINABILITY ENGINE                         |
|   • Confidence scoring (calibrated probabilities)                                 |
|   • Feature importance (top TF-IDF coefficients explaining why issue is a Bug)     |
|   • Automated routing recommendation (suggested engineering team / label)         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            5. SERVING & INTERFACE (M7 - M8)                       |
|   • FastAPI REST Backend (`POST /predict`)                                        |
|   • Interactive UI / Dashboard for real-time issue submission & triage            |
+-----------------------------------------------------------------------------------+
```

---

## 3. Milestone 2 Deep-Dive: Exactly What Was Done & Why

### Task 1 & 2: Why `helmo/github-issues` and Apache Parquet?
- **Why not Kaggle?** The Kaggle candidate required private API keys (`~/.kaggle/kaggle.json`), which breaks zero-setup reproducibility.
- **Why not `ansnadeem/aic-framework`?** That repository contains extraction notebooks, but no raw data file committed; scraping GitHub directly hits the 60 requests/hour unauthenticated rate limit.
- **Why `helmo/github-issues`?** It is public, Apache 2.0 licensed, 12.66 MB (manageable size), and provides raw GitHub API objects extracted from `huggingface/datasets`.
- **Why Parquet?** Columnar storage with Snappy compression; loads in milliseconds; preserves nested array/dictionary schema without corrupted text escaping common to CSVs.

### Task 3: The Data Quality Audit (A Key Story to Tell Examiners)
- When auditing the raw Parquet file, we discovered that **the first 100 rows were exact duplicates of rows 100 through 199**!
- We diagnosed that the original scraper likely paginated page 1 twice before appending page 2.
- By enforcing deduplication on `id`, we eliminated 100 duplicate rows, yielding 7,440 unique records.
- We also discovered that 4,325 records had a non-null `pull_request` dictionary. Pull requests represent code contributions, not issue tickets. Filtering them was essential to avoid severe training distribution skew.

### Task 4 & 5: Target Taxonomy and the Data Leakage Firewall
Examiners love to ask about **Data Leakage**. This is your strongest conceptual defense:

#### The Data Leakage Problem:
> *"If our model predicts the issue type when a user clicks 'Submit', what information exists at that exact millisecond?"*

- **Available:** Title, Body, Submission Timestamp, Author Association (`MEMBER`, `CONTRIBUTOR`, `NONE`).
- **NOT Available (Target Leakage):**
  - Maintainer tags (`bug`, `enhancement`) — these ARE the ground truth!
  - Issue state (`closed`) and resolution timestamp (`closed_at`) — bugs take longer to fix than questions.
  - Number of comments and comment text — comments accumulate during triage.
  - Assigned maintainer (`assignee`) — maintainers are assigned *after* reading the issue.
  - Milestone (`milestone`) — assigned during sprint planning.
  - Reactions (`+1`, `heart`) — added by the community over time.

**Our Defense:** We implemented a strict **Feature Allowlist** in `scripts/preprocess_dataset.py`. All 12 post-triage fields are explicitly stripped before feature engineering!

#### Target Taxonomy Mapping:
- **`BUG` (771):** `bug`, `dataset bug`, `metric bug`
- **`FEATURE` (615):** `enhancement`, `dataset request`, `dataset contribution`
- **`DOCUMENTATION` (25):** `documentation`
- **`QUESTION` (31):** `question`, `generic discussion`, `Dataset discussion`, `Metric discussion`
- **`OTHER` (17):** `maintenance`, `refactoring`
- **`UNLABELED` (1,493):** Issues without tags (held out for K-Means unsupervised clustering).
- **`CONFLICTING` (34):** Issues with contradictory tags (e.g. `[bug, enhancement]`). Safely excluded.

---

## 4. Key Mathematical Formulas (Be Ready to Write on Board)

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)
Used to convert raw text into high-dimensional numerical vectors:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where sublinear term frequency dampens long issue descriptions:
$$\text{TF}(t, d) = 1 + \log(\text{count}(t, d)) \quad \text{for count} > 0$$

And Inverse Document Frequency measures how rare a term is across all issues:
$$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 2. Multinomial Naive Bayes (Baseline Model)
Uses Bayes' theorem with the conditional independence assumption:

$$P(C_k \mid d) \propto P(C_k) \prod_{i=1}^{n} P(w_i \mid C_k)$$

With Laplace smoothing ($\alpha = 1.0$) to handle unseen words:
$$P(w_i \mid C_k) = \frac{\text{count}(w_i, C_k) + \alpha}{\sum_{w} \text{count}(w, C_k) + \alpha \cdot |V|}$$

### 3. Macro-Averaged F1-Score (Why Accuracy Fails)
Because `BUG` has 771 samples and `OTHER` has 17 samples (45:1 imbalance), standard accuracy is biased towards the majority class. We evaluate using **Macro-F1**:

$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k = \frac{1}{K} \sum_{k=1}^{K} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

Macro-F1 assigns equal weight to every category, penalizing models that ignore minority classes.

---

## 5. Top 15 Viva Questions & Defensible Answers

### Q1: Why did you filter out Pull Requests (PRs)? Why not train on them?
**Answer:** In the GitHub REST API, `/issues` returns both issues and pull requests. However, pull requests are developer code contributions containing diff summaries, merge commits, and code review checklists. In contrast, issue tickets are problem descriptions submitted by users. Mixing PRs into issue classification causes severe domain mismatch and makes the problem unrealistically easy because PR titles often start with `"fix:"` or `"feat:"`.

### Q2: How did you handle issues with no labels (1,493 issues)? Did you infer labels using an LLM or keyword search?
**Answer:** Absolutely not! As emphasized in our methodology, we **never infer ground-truth labels from text heuristics and then evaluate supervised models against those same inferred labels**—that creates circular reasoning and self-fulfilling bias. Instead, the 1,493 unlabeled issues are preserved in `github_issues_cleaned.parquet` as an unlabelled exploration pool for **unsupervised learning (K-Means and PCA)** in Milestone 6.

### Q3: How did you resolve multi-label issues?
**Answer:** GitHub labels encompass multiple orthogonal dimensions: issue type (`bug`), technical component (`vision`), and community triage (`good first issue`). We separated these dimensions:
1. If an issue had only orthogonal labels (e.g. `[good first issue]`), it was marked as `UNMAPPED`.
2. If an issue had a single type category alongside orthogonal tags (e.g. `[bug, dataset-viewer]`), it was mapped to `BUG`.
3. If an issue had contradictory type tags (e.g. `[bug, enhancement]`), it was marked as `CONFLICTING` (34 issues) and excluded from single-label training.

### Q4: What is data leakage and how did your pipeline prevent it?
**Answer:** Data leakage occurs when information from outside the training partition—or information that would not be available at prediction time—is introduced into the model features. For issue triage, fields like `state` (closed), `closed_at`, `comments`, and `assignee` only exist *after* human maintainers triage the ticket. We built a strict feature allowlist containing only creation-time fields (`clean_title`, `clean_body`, `author_association`, `created_at`), completely blocking post-triage leakage.

### Q5: Your dataset has a 45:1 class imbalance between BUG (771) and OTHER (17). How will you train models without them collapsing to the majority class?
**Answer:** We established a three-pronged strategy in our Split Readiness Report:
1. **Evaluation Metric:** Discard raw accuracy as a primary metric; prioritize **Macro-F1** and **Balanced Accuracy**.
2. **Loss Weighting:** Use cost-sensitive learning via scikit-learn's `class_weight='balanced'`, which assigns inverse class frequency weights to the loss function:
   $$w_k = \frac{N}{K \cdot N_k}$$
3. **Resampling:** Benchmark SMOTE (Synthetic Minority Over-sampling Technique) or random oversampling on the minority classes during training.

### Q6: Why did you choose classical ML (TF-IDF + Naive Bayes / SVM) instead of fine-tuning a transformer or calling ChatGPT?
**Answer:** Three reasons:
1. **Academic & Pedagogical Rigor:** Classical statistical ML makes the mathematics visible (loss functions, margins, conditional probabilities, sparsity) rather than hiding behind a cloud API.
2. **Computational Footprint:** TF-IDF + Logistic Regression trains in under 2 seconds on a CPU with zero GPU requirements, zero cost, and sub-millisecond inference latency suitable for high-throughput GitHub webhooks.
3. **Strong Empirical Baseline:** Decades of empirical software engineering research (e.g. Menzies et al., Herzig et al.) prove that TF-IDF with linear classifiers forms an exceptionally competitive baseline for software defect classification.

### Q7: What did your N-gram analysis reveal?
**Answer:** Figure 9 showed stark linguistic contrast. Bug reports prominently feature technical error bigrams: `traceback call`, `error message`, `typeerror expected`, `fails when`. Feature requests feature aspirational bigrams: `add support`, `feature request`, `would nice`, `allow user`. This confirms that linear separators in n-gram TF-IDF space will have strong discriminative power.

### Q8: What is near-duplicate contamination, and how does it affect your train/test split?
**Answer:** We identified 483 pairs of issues with TF-IDF cosine similarity >= 0.85 (e.g. multiple users reporting the exact same `ImportError` with identical stack traces). If one instance falls into train and the duplicate into test, the model can memorize the title and artificially inflate test accuracy. To prevent this, we enforce near-duplicate grouping in our cross-validation splits.

### Q9: What is the difference between Random Stratified Split and Temporal Split?
**Answer:** Random stratified split guarantees that minority classes (`OTHER`: 17, `DOCS`: 25) are proportionately represented in both training and test sets. Temporal split (train on 2020-2023, test on 2024-2025) reflects production reality where models predict future issues, but suffers from temporal drift (Figure 8 showed recent years had far more features than bugs). We use Stratified Split as our primary evaluation and Temporal Split as a concept drift benchmark.

### Q10: How did you preprocess technical text? Did you remove stop words and stem?
**Answer:** We deliberately avoided aggressive stemming (e.g. Porter Stemmer) and aggressive stop-word removal. In technical software text, words like `"not"`, `"cannot"`, and `"fails"` convey critical negation in bug reports. Stemming also corrupts programming terms (e.g. stemming `"streaming"` or `"dataset"` can distort library class names). We cleaned HTML tags and markdown images while strictly preserving code tokens, version numbers (`v1.2.3`), and identifiers (`__iter__`).

---

## 6. Live Viva Demonstration Commands (Run in 30 Seconds)

When the examiner asks you to demonstrate your work live in terminal, run these 3 exact commands:

### Command 1: Run Automated Test Suite (Proves complete code correctness)
```powershell
pytest -v tests/
```
*Expected output: 7 passed in ~1.17s.*

### Command 2: Run Data Preprocessing Pipeline (Proves reproducibility)
```powershell
python scripts/preprocess_dataset.py
```
*Expected output: Ingests raw data, removes 100 duplicate rows, filters 4,325 PRs, outputs 1,459 supervised issues and 3,115 pure issues.*

### Command 3: Run EDA & Visualizations (Proves figure generation)
```powershell
python scripts/run_eda.py
```
*Expected output: Generates all 11 figures into `reports/figures/` and writes `reports/eda_summary.md`.*

