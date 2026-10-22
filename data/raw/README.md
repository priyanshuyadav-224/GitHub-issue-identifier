# Raw Dataset Documentation

## Provenance & Overview
- **Dataset Name:** Hugging Face Datasets GitHub Issues
- **Hugging Face Hub ID:** [`helmo/github-issues`](https://huggingface.co/datasets/helmo/github-issues)
- **Primary Source Repository:** [`huggingface/datasets`](https://github.com/huggingface/datasets)
- **Retrieval Date:** 2026-10-15
- **License:** Apache 2.0
- **File Format:** Apache Parquet (`github_issues_raw.parquet`)
- **File Size:** 12,659,096 bytes (~12.07 MB)
- **SHA-256 Checksum:** `09453eefae39e45a969ab0bee72ca0e188fe79dc50403b0f2a78c39894f5d1a3`
- **Total Records:** 7,540 rows
- **Total Schema Columns:** 33 columns

## Acquisition Instructions
The raw dataset can be downloaded directly from Hugging Face Hub using the following Python command:

```python
import urllib.request

url = "https://huggingface.co/datasets/helmo/github-issues/resolve/main/data/train-00000-of-00001.parquet"
destination = "data/raw/github_issues_raw.parquet"
urllib.request.urlretrieve(url, destination)
print("Raw dataset successfully downloaded.")
```

Or via the Hugging Face `datasets` library:
```python
from datasets import load_dataset
ds = load_dataset("helmo/github-issues")
ds["train"].to_parquet("data/raw/github_issues_raw.parquet")
```

## Raw Schema Fields (33 columns)
1. `url` (str): GitHub REST API issue URL
2. `repository_url` (str): Repository API URL
3. `labels_url` (str): Labels REST URL
4. `comments_url` (str): Comments REST URL
5. `events_url` (str): Events REST URL
6. `html_url` (str): Web URL on GitHub
7. `id` (int64): Unique issue entity identifier
8. `node_id` (str): GraphQL global node ID
9. `number` (int64): Issue / PR sequence number within repository
10. `title` (str): Author's issue title
11. `user` (dict): Submitting user's profile metadata
12. `labels` (array of dicts): Associated GitHub labels (`name`, `color`, `description`)
13. `state` (str): Issue status (`open` or `closed`)
14. `locked` (bool): Administrative lock status
15. `assignee` (dict/null): Assigned maintainer profile
16. `assignees` (array of dicts): Multiple assignees
17. `milestone` (dict/null): Associated milestone release
18. `comments` (array/int): Comments count or list
19. `created_at` (timestamp): Submission timestamp (UTC)
20. `updated_at` (timestamp): Last modification timestamp (UTC)
21. `closed_at` (timestamp/null): Issue closure timestamp (UTC)
22. `author_association` (str): Relationship of author (`MEMBER`, `CONTRIBUTOR`, `NONE`, etc.)
23. `type` (float/null): Issue type field if present (sparse)
24. `active_lock_reason` (float/null): Rationale for lock
25. `draft` (float/null): PR draft indicator
26. `pull_request` (dict/null): Present only if record is a pull request
27. `body` (str/null): Main markdown description
28. `closed_by` (dict/null): User who closed the issue
29. `reactions` (dict): Summary of user emoji reactions
30. `timeline_url` (str): Timeline REST endpoint
31. `performed_via_github_app` (float/null): App bot indicator
32. `state_reason` (str/null): Reason for closure (`completed`, `not_planned`)
33. `sub_issues_summary` (dict/null): Sub-issue breakdown

## Known Limitations of Raw Data
1. **Pull Requests Included:** In the GitHub REST API, `/issues` endpoint returns both issues and pull requests. In this raw dump, 4,325 of 7,440 unique records are pull requests (identifiable by non-null `pull_request`).
2. **Duplicate Block:** The first 100 rows in the raw Parquet file are duplicate entries of rows 100 to 199.
3. **Unlabeled Records:** 1,493 pure issues have no tags assigned, reflecting real-world maintainer triage delay.

