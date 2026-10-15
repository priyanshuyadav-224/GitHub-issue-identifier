# Dataset Card for GitHub Project Issue Classifier

## Dataset Summary
The **GitHub Project Issue Classifier Dataset** is a curated, verified corpus of real-world public GitHub issues derived from the foundational [`huggingface/datasets`](https://github.com/huggingface/datasets) repository (via [`helmo/github-issues`](https://huggingface.co/datasets/helmo/github-issues)). The dataset is specifically structured for natural language processing (NLP) and statistical machine learning tasks, specifically automating the categorization of incoming software issues into actionable types (**BUG**, **FEATURE**, **QUESTION**, **DOCUMENTATION**, **OTHER**).

## Supported Tasks and Leaderboards
- **Multi-Class Text Classification:** Predicting the issue category from the initial title and description.
- **Unsupervised Topic Modeling / Clustering:** Discovering latent technical themes and emerging bugs within unlabelled issue text (1,493 unlabeled issues).
- **Temporal Out-of-Distribution Robustness:** Benchmarking concept drift resilience using chronological issue arrival splits.

## Languages
The text is written primarily in **English**, containing technical vocabulary, Python code snippets, stack traces, terminal outputs, error logs, library version numbers, and file system paths.

---

## Dataset Structure

### Data Instances
A typical instance in the supervised dataset comprises author-submitted text and metadata at creation time:

```json
{
  "id": 1690557262,
  "number": 5792,
  "html_url": "https://github.com/huggingface/datasets/issues/5792",
  "repository": "huggingface/datasets",
  "author_association": "NONE",
  "created_at": "2023-04-30T14:22:10+00:00",
  "clean_title": "TypeError: 'generator' object is not subscriptable in IterableDataset.take()",
  "clean_body": "### Describe the bug\n\nWhen using take(5) on an IterableDataset, it crashes with TypeError...",
  "text": "TypeError: 'generator' object is not subscriptable in IterableDataset.take()\n\n### Describe the bug...",
  "title_char_len": 74,
  "body_char_len": 412,
  "text_char_len": 488,
  "text_word_count": 58,
  "original_labels": "bug;streaming",
  "target_class": "BUG",
  "mapping_status": "Single mapped category"
}
```

### Data Fields
- `id` (*int64*): Unique integer identifier assigned by GitHub.
- `number` (*int64*): The issue number within the repository.
- `html_url` (*string*): Public web link to the original issue.
- `repository` (*string*): Repository identifier (`huggingface/datasets`).
- `author_association` (*string*): Permitted categories: `NONE`, `CONTRIBUTOR`, `MEMBER`, `COLLABORATOR`.
- `created_at` (*timestamp*): UTC timestamp when the issue was opened.
- `clean_title` (*string*): Preprocessed title string.
- `clean_body` (*string*): Preprocessed body description.
- `text` (*string*): Combined input text (`clean_title + "\n\n" + clean_body`).
- `title_char_len` (*int64*): Character count of cleaned title.
- `body_char_len` (*int64*): Character count of cleaned body.
- `text_char_len` (*int64*): Character count of combined text.
- `text_word_count` (*int64*): Word count of combined text.
- `original_labels` (*string*): Semicolon-delimited list of source GitHub labels.
- `target_class` (*string*): Standardized label: `BUG`, `FEATURE`, `QUESTION`, `DOCUMENTATION`, `OTHER`.
- `mapping_status` (*string*): Audit reason justifying class mapping.

### Data Splits
- **Supervised Dataset (N = 1,459):** Fully labeled subset for supervised model training and evaluation.
- **Full Cleaned Corpus (N = 3,115):** Includes 1,493 unlabelled issues for unsupervised clustering, active learning, and semi-supervised experiments.

---

## Dataset Creation

### Curation Rationale
In real-world software engineering, project maintainers spend excessive hours triaging incoming tickets. To train a reliable machine learning classifier, training data must mirror real conditions while avoiding **target leakage** (using triage metadata that only exists after a maintainer has read the issue).

### Source Data & Provenance
- Source: GitHub REST API issues extraction for `huggingface/datasets`.
- Dataset Provider: Hugging Face Datasets Hub (`helmo/github-issues`).
- Version Date: June 2025.
- License: Apache License 2.0.

### Annotation & Label Mapping
Annotations are derived from maintainer-assigned labels in the source repository, mapped into a 5-class taxonomy using an audited rule system:
- **`BUG` (771):** `bug`, `dataset bug`, `metric bug`.
- **`FEATURE` (615):** `enhancement`, `dataset request`, `dataset contribution`.
- **`DOCUMENTATION` (25):** `documentation`.
- **`QUESTION` (31):** `question`, `generic discussion`, `Dataset discussion`, `Metric discussion`.
- **`OTHER` (17):** `maintenance`, `refactoring`.
- **Excluded Conflicting (34):** Issues with contradictory labels (e.g. `[bug, enhancement]`).
- **Excluded Unmapped (129):** Issues with only triage/component tags (`good first issue`, `vision`).

---

## Considerations for Using the Data

### Data Leakage Prevention
To ensure valid scientific evaluation, all post-triage fields (`state`, `closed_at`, `closed_by`, `state_reason`, `comments`, `assignee`, `milestone`, `reactions`) are strictly excluded from input features. Only features available at the moment of issue creation are permitted.

### Class Imbalance
The supervised dataset exhibits a significant 45:1 imbalance ratio between `BUG` (771) and `OTHER` (17). Models must not be evaluated solely on accuracy; **Macro-F1, Balanced Accuracy, and per-class precision/recall** must be reported.

### Known Biases & Domain Specificity
1. **Repository Specificity:** The issues reflect the Python data science and machine learning ecosystem (PyTorch, Apache Arrow, Hugging Face Hub, datasets).
2. **Community Norms:** In this repository, `BUG` and `FEATURE` tickets dominate, while general usage questions are frequently redirected to forum channels, explaining the low proportion of `QUESTION` tags.
