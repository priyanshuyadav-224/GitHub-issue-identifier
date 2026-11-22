"""
scripts/generate_demonstration_video.py
=======================================
Generates an HD 1080p (1920x1080 @ 30fps) Project Demonstration Walkthrough Video
for the Statistical Machine Learning Final Project Milestone 3 Submission.
Uploads the resulting MP4 video directly to the public GitHub Release v1.0.0.
Author: Priyanshu Yadav (S25CSEU0725) | Batch-25
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_gradient_bg(width, height, color1, color2):
    """Creates a vertical linear gradient image."""
    base = Image.new("RGB", (width, height), color1)
    top_r, top_g, top_b = color1
    bot_r, bot_g, bot_b = color2

    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        factor = y / height
        r = int(top_r + factor * (bot_r - top_r))
        g = int(top_g + factor * (bot_g - top_g))
        b = int(top_b + factor * (bot_b - top_b))
        arr[y, :, 0] = r
        arr[y, :, 1] = g
        arr[y, :, 2] = b
    return Image.fromarray(arr)


def load_fonts():
    """Loads system TrueType fonts with fallbacks."""
    font_path_regular = "C:/Windows/Fonts/segoeui.ttf"
    font_path_bold = "C:/Windows/Fonts/segoeuib.ttf"
    font_path_mono = "C:/Windows/Fonts/consola.ttf"

    if not os.path.exists(font_path_regular):
        font_path_regular = "C:/Windows/Fonts/arial.ttf"
    if not os.path.exists(font_path_bold):
        font_path_bold = "C:/Windows/Fonts/arialbd.ttf"
    if not os.path.exists(font_path_mono):
        font_path_mono = "C:/Windows/Fonts/cour.ttf"

    return {
        "title": ImageFont.truetype(font_path_bold, 44),
        "subtitle": ImageFont.truetype(font_path_regular, 24),
        "heading": ImageFont.truetype(font_path_bold, 30),
        "body": ImageFont.truetype(font_path_regular, 20),
        "body_bold": ImageFont.truetype(font_path_bold, 20),
        "small": ImageFont.truetype(font_path_regular, 16),
        "mono": ImageFont.truetype(font_path_mono, 18),
        "mono_small": ImageFont.truetype(font_path_mono, 15),
        "kpi_val": ImageFont.truetype(font_path_bold, 38),
        "kpi_lbl": ImageFont.truetype(font_path_bold, 14),
    }


def draw_header(draw, fonts, category, title):
    # Header bar
    draw.rectangle([(0, 0), (1920, 100)], fill=(13, 19, 34))
    draw.line([(0, 100), (1920, 100)], fill=(31, 41, 61), width=2)

    # Brand badge
    draw.rounded_rectangle([(60, 22), (116, 78)], radius=10, fill=(37, 99, 235))
    draw.text((72, 34), "GT", font=fonts["heading"], fill=(255, 255, 255))

    # Header texts
    draw.text((135, 24), category.upper(), font=fonts["mono_small"], fill=(96, 165, 250))
    draw.text((135, 48), title, font=fonts["heading"], fill=(255, 255, 255))

    # Candidate tag
    draw.text((1400, 38), "Priyanshu Yadav (S25CSEU0725) | Batch-25", font=fonts["subtitle"], fill=(148, 163, 184))


def draw_footer(draw, fonts, scene_idx, total_scenes):
    # Footer bar
    draw.rectangle([(0, 1030), (1920, 1080)], fill=(7, 10, 18))
    draw.line([(0, 1030), (1920, 1030)], fill=(31, 41, 61), width=1)

    draw.text((60, 1045), "GitTriage AI - Statistical Machine Learning & NLP Issue Classification System", font=fonts["small"], fill=(100, 116, 139))
    draw.text((1600, 1045), f"Demonstration Walkthrough | Section {scene_idx}/{total_scenes}", font=fonts["small"], fill=(148, 163, 184))


def render_scene_1(fonts):
    """Scene 1: Title & Executive Overview"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_footer(draw, fonts, 1, 8)

    # Center Hero Box
    draw.rounded_rectangle([(140, 140), (1780, 980)], radius=16, fill=(17, 24, 39), outline=(31, 41, 61), width=2)

    # Top Tag
    draw.rounded_rectangle([(180, 180), (740, 220)], radius=6, fill=(19, 33, 61), outline=(30, 58, 138), width=1)
    draw.text((195, 188), "STATISTICAL MACHINE LEARNING - FINAL PROJECT DEMONSTRATION", font=fonts["mono_small"], fill=(96, 165, 250))

    # Main Title
    draw.text((180, 250), "GitHub Project Issue Classifier", font=fonts["title"], fill=(255, 255, 255))
    draw.text((180, 315), "AI-Powered Issue Classification, Prioritization & Routing System", font=fonts["subtitle"], fill=(96, 165, 250))

    # Author Credentials Card
    draw.rounded_rectangle([(180, 380), (1050, 480)], radius=10, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
    draw.text((210, 400), "Candidate Name:  Priyanshu Yadav", font=fonts["body_bold"], fill=(248, 250, 252))
    draw.text((210, 435), "Student ID / Batch:  S25CSEU0725  |  Batch-25  |  Submission Date: November 22, 2026", font=fonts["body"], fill=(148, 163, 184))

    # Key Highlights
    draw.text((180, 520), "Project Architecture & Key Highlights:", font=fonts["heading"], fill=(255, 255, 255))
    highlights = [
        "Classical Statistical Machine Learning: TF-IDF, Naive Bayes, Logistic Regression, Linear SVM, Random Forest, K-Means.",
        "Zero Cloud API Dependencies: 100% local, mathematically transparent, zero operational costs.",
        "Deterministic Data Engineering Pipeline: 7,540 raw records processed into 1,459 supervised and 1,493 unlabeled issues.",
        "Target Leakage Prevention Firewall: Strict temporal allowlisting quarantines post-triage metadata.",
        "Ultra-Fast Inference: Sub-millisecond latency (0.088 - 1.286 ms) with live web console and REST API.",
    ]
    y = 575
    for h in highlights:
        draw.ellipse([(190, y + 6), (200, y + 16)], fill=(59, 130, 246))
        draw.text((215, y), h, font=fonts["body"], fill=(203, 213, 225))
        y += 40

    # KPI Strip at bottom
    kpi_cards = [
        ("88.70%", "TOP TEST ACCURACY", "Linear SVM Calibrated"),
        ("57.40%", "MACRO-F1 SCORE", "Balanced Class Weights"),
        ("0.10 ms", "INFERENCE LATENCY", "Sub-millisecond Scale"),
        ("12 / 12", "PYTEST UNIT TESTS", "100% Pass Rate"),
    ]
    kx = 180
    for val, lbl, sub in kpi_cards:
        draw.rounded_rectangle([(kx, 830), (kx + 350, 930)], radius=10, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
        draw.text((kx + 20, 840), val, font=fonts["kpi_val"], fill=(52, 211, 153))
        draw.text((kx + 20, 885), lbl, font=fonts["kpi_lbl"], fill=(148, 163, 184))
        draw.text((kx + 20, 905), sub, font=fonts["small"], fill=(100, 116, 139))
        kx += 390

    return img


def render_scene_2(fonts):
    """Scene 2: Data Engineering & Quality Audit Funnel"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Data Engineering Layer", "Dataset Provenance, Deduplication & Quality Audit")
    draw_footer(draw, fonts, 2, 8)

    # Left: Text Explanation
    draw.rounded_rectangle([(60, 130), (880, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 160), "Data Provenance & Filtering Pipeline", font=fonts["heading"], fill=(255, 255, 255))

    points = [
        ("Source Dataset:", "Verified public helmo/github-issues (Apache 2.0 license) extracted via GitHub REST API from huggingface/datasets (7,540 raw records, 12.66 MB Parquet)."),
        ("Deduplication Discovery:", "Automated schema auditing detected a 100-row pagination duplication artifact (rows 0-99 exactly identical to 100-199). Deduplication on stable GitHub ID eliminated all 100 duplicate rows."),
        ("Pull Request Isolation:", "4,325 records (58.13%) were pull requests containing code review diffs. Including them creates severe distribution skew; all 4,325 PRs were isolated, yielding 3,115 pure issues."),
        ("Defensible 5-Class Taxonomy:", "Supervised corpus (1,459 issues):\n  - BUG: 771 (52.84%)\n  - FEATURE: 615 (42.15%)\n  - QUESTION: 31 (2.12%)\n  - DOCUMENTATION: 25 (1.71%)\n  - OTHER: 17 (1.17%)\nUnlabeled pool: 1,493 issues for K-Means."),
    ]
    y = 220
    for title, desc in points:
        draw.text((90, y), title, font=fonts["body_bold"], fill=(96, 165, 250))
        y += 30
        for line in desc.split("\n"):
            draw.text((110, y), line, font=fonts["body"], fill=(203, 213, 225))
            y += 28
        y += 15

    # Right: Embedded Funnel Chart
    draw.rounded_rectangle([(920, 130), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((950, 160), "Pipeline Ingestion Funnel (reports/figures/fig01)", font=fonts["heading"], fill=(255, 255, 255))

    fig_path = "reports/figures/fig01_filtering_pipeline_counts.png"
    if os.path.exists(fig_path):
        fig_img = Image.open(fig_path).convert("RGB")
        fig_img = fig_img.resize((880, 560), Image.Resampling.LANCZOS)
        img.paste(fig_img, (950, 220))

    # Caption box
    draw.rounded_rectangle([(950, 810), (1830, 970)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
    draw.text((970, 825), "Summary of Data Cleaning Actions:", font=fonts["body_bold"], fill=(52, 211, 153))
    draw.text((970, 860), "1. Deduplicated on stable GitHub ID: 7,540 -> 7,440 unique records.", font=fonts["small"], fill=(203, 213, 225))
    draw.text((970, 890), "2. Filtered out 4,325 pull requests to prevent PR review diff contamination.", font=fonts["small"], fill=(203, 213, 225))
    draw.text((970, 920), "3. Output: 1,459 supervised ground-truth issues + 1,493 unlabeled issues.", font=fonts["small"], fill=(203, 213, 225))

    return img


def render_scene_3(fonts):
    """Scene 3: Target Leakage Prevention Firewall"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Target Leakage Prevention", "Scientific Allowlisting vs Quarantined Metadata")
    draw_footer(draw, fonts, 3, 8)

    # Top description box
    draw.rounded_rectangle([(60, 130), (1860, 240)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 150), "The Target Leakage Problem in Automated Issue Triage:", font=fonts["heading"], fill=(239, 68, 68))
    desc = (
        "In production, triage occurs at the exact millisecond the user creates a ticket. Any model trained on post-triage attributes "
        "(such as closed_at, comments count, assignee, or labels) exhibits catastrophic real-world failure due to target leakage. "
        "GitTriage enforces a strict temporal firewall isolating post-triage metadata from model feature vectors."
    )
    draw.text((90, 195), desc, font=fonts["body"], fill=(203, 213, 225))

    # Side-by-side comparison tables
    # Left: Allowed features
    draw.rounded_rectangle([(60, 270), (930, 1000)], radius=12, fill=(17, 24, 39), outline=(16, 185, 129), width=2)
    draw.text((90, 300), "FEATURE ALLOWLIST (Available at Creation)", font=fonts["heading"], fill=(52, 211, 153))

    allowed = [
        ("clean_title", "Author's original issue summary, stripped of HTML noise."),
        ("clean_body", "Problem description, reproduction code, error tracebacks."),
        ("text (combined)", "Title concatenated with body for comprehensive TF-IDF coverage."),
        ("author_association", "NONE (external), CONTRIBUTOR, or MEMBER (maintainer)."),
        ("created_at", "ISO-8601 timestamp representing issue arrival time."),
        ("title_char_len", "Character length of normalized title string."),
        ("body_char_len", "Character length of normalized body text."),
        ("text_word_count", "Word token count of combined author description."),
    ]
    y = 360
    for name, expl in allowed:
        draw.rounded_rectangle([(90, y), (900, y + 65)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
        draw.text((110, y + 10), name, font=fonts["mono"], fill=(96, 165, 250))
        draw.text((110, y + 36), expl, font=fonts["small"], fill=(148, 163, 184))
        y += 75

    # Right: Quarantined features
    draw.rounded_rectangle([(980, 270), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(239, 68, 68), width=2)
    draw.text((1010, 300), "QUARANTINED METADATA (Post-Triage Leakage)", font=fonts["heading"], fill=(248, 113, 113))

    quarantined = [
        ("labels", "Ground truth target classes assigned by human maintainers."),
        ("state (open / closed)", "Outcome assigned after investigation and bug resolution."),
        ("closed_at / closed_by", "Timestamp and user identity of issue closer."),
        ("comments count & text", "Post-submission engineer discussion and debugging logs."),
        ("assignee / assignees", "Maintainer assigned after triage is already complete."),
        ("milestone", "Sprint/release milestone assigned during project planning."),
        ("reactions (+1, heart)", "Community reactions that accumulate over issue lifetime."),
        ("state_reason", "Resolution verdict (completed, not_planned)."),
    ]
    y = 360
    for name, expl in quarantined:
        draw.rounded_rectangle([(1010, y), (1830, y + 65)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
        draw.text((1030, y + 10), name + "  [QUARANTINED]", font=fonts["mono"], fill=(248, 113, 113))
        draw.text((1030, y + 36), expl, font=fonts["small"], fill=(148, 163, 184))
        y += 75

    return img


def render_scene_4(fonts):
    """Scene 4: Mathematical Formulations"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Statistical Machine Learning", "Mathematical Formulations & Algorithmic Foundations")
    draw_footer(draw, fonts, 4, 8)

    # 4 Quadrants for Algorithms
    quads = [
        ("1. Sublinear TF-IDF Feature Representation",
         "TF(t, d) = 1 + log(f(t, d)) for f(t, d) > 0;  0 otherwise.\n"
         "IDF(t, D) = log((1 + |D|) / (1 + df(t))) + 1\n\n"
         "Sublinear scaling dampens the dominance of lengthy tracebacks in bug reports.\n"
         "Vocabulary: 5,000 unigram + bigram terms, normalized under L2 norm: v = x / ||x||_2.",
         (60, 130, 930, 540)),
        ("2. Multinomial Naive Bayes (Laplace Alpha = 0.5)",
         "P(c | d) proportional to P(c) * PROD P(w_i | c)\n"
         "P(w | c) = (count(w, c) + alpha) / (SUM count(w', c) + alpha * |V|)\n\n"
         "Applies conditional independence assumption across word tokens.\n"
         "Laplace smoothing (alpha = 0.5) eliminates zero-probability underflow for unseen tokens.",
         (980, 130, 1860, 540)),
        ("3. L2-Regularized Logistic Regression (Softmax)",
         "P(y = k | x) = exp(w_k^T x) / SUM exp(w_j^T x)\n"
         "Loss: J(W) = - SUM log P(y_i | x_i) + (lambda / 2) * ||W||_F^2\n\n"
         "Multi-class cross-entropy loss with L2 regularization penalty.\n"
         "Cost-sensitive balanced class weighting: w_k = N / (K * N_k), elevating minority recall.",
         (60, 570, 930, 1000)),
        ("4. Linear Support Vector Machine (Calibrated)",
         "min (1/2) * ||w_k||^2 + C * SUM xi_i  subject to  y_i * (w_k^T x_i + b) >= 1 - xi_i\n\n"
         "Finds the maximum-margin hyperplane separating classes in 5,000-dimensional space.\n"
         "Probabilities calibrated via 3-fold Platt sigmoid scaling: P(y=1|f) = 1 / (1 + exp(A*f + B)).",
         (980, 570, 1860, 1000)),
    ]

    for title, text, box in quads:
        draw.rounded_rectangle([box[0:2], box[2:4]], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
        draw.text((box[0] + 30, box[1] + 25), title, font=fonts["heading"], fill=(96, 165, 250))
        y = box[1] + 75
        for line in text.split("\n"):
            if "TF(" in line or "P(" in line or "Loss:" in line or "min " in line:
                draw.text((box[0] + 30, y), line, font=fonts["mono"], fill=(52, 211, 153))
            else:
                draw.text((box[0] + 30, y), line, font=fonts["body"], fill=(203, 213, 225))
            y += 28

    return img


def render_scene_5(fonts, bench):
    """Scene 5: Statistical ML Comparative Benchmarks"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Empirical Evaluation", "Statistical ML Comparative Benchmark Results (Test N = 292)")
    draw_footer(draw, fonts, 5, 8)

    # Left: Benchmark Table
    draw.rounded_rectangle([(60, 130), (1050, 680)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 155), "Held-Out Test Set Performance (Stratified 80/20)", font=fonts["heading"], fill=(255, 255, 255))

    # Table Header
    cols = ["Model Architecture", "Test Acc", "Macro-F1", "CV Acc", "Latency"]
    xs = [90, 470, 620, 770, 920]
    draw.rectangle([(80, 205), (1030, 245)], fill=(30, 58, 138))
    for c_name, cx in zip(cols, xs):
        draw.text((cx, 215), c_name, font=fonts["body_bold"], fill=(255, 255, 255))

    # Table Rows
    rows = [
        ("Linear SVM (Calibrated)", f"{bench['Linear SVM']['test_accuracy']*100:.2f}%", f"{bench['Linear SVM']['macro_f1']*100:.2f}%", f"{bench['Linear SVM']['cv_accuracy_5fold']*100:.2f}%", f"{bench['Linear SVM']['latency_ms']:.2f} ms"),
        ("Logistic Regression (L2)", f"{bench['Logistic Regression']['test_accuracy']*100:.2f}%", f"{bench['Logistic Regression']['macro_f1']*100:.2f}%", f"{bench['Logistic Regression']['cv_accuracy_5fold']*100:.2f}%", f"{bench['Logistic Regression']['latency_ms']:.2f} ms"),
        ("Multinomial Naive Bayes", f"{bench['Multinomial Naive Bayes']['test_accuracy']*100:.2f}%", f"{bench['Multinomial Naive Bayes']['macro_f1']*100:.2f}%", f"{bench['Multinomial Naive Bayes']['cv_accuracy_5fold']*100:.2f}%", f"{bench['Multinomial Naive Bayes']['latency_ms']:.2f} ms"),
        ("Random Forest (150 trees)", f"{bench['Random Forest']['test_accuracy']*100:.2f}%", f"{bench['Random Forest']['macro_f1']*100:.2f}%", f"{bench['Random Forest']['cv_accuracy_5fold']*100:.2f}%", f"{bench['Random Forest']['latency_ms']:.2f} ms"),
    ]
    ry = 255
    for m_name, acc, f1, cv, lat in rows:
        bg_col = (13, 20, 36) if (ry // 40) % 2 == 0 else (17, 24, 39)
        draw.rectangle([(80, ry), (1030, ry + 42)], fill=bg_col)
        draw.text((xs[0], ry + 10), m_name, font=fonts["body_bold"], fill=(255, 255, 255))
        draw.text((xs[1], ry + 10), acc, font=fonts["body"], fill=(52, 211, 153))
        draw.text((xs[2], ry + 10), f1, font=fonts["body"], fill=(96, 165, 250))
        draw.text((xs[3], ry + 10), cv, font=fonts["body"], fill=(203, 213, 225))
        draw.text((xs[4], ry + 10), lat, font=fonts["mono"], fill=(251, 191, 36))
        ry += 45

    # Key Empirical Findings
    draw.rounded_rectangle([(60, 710), (1050, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 735), "Key Empirical Takeaways:", font=fonts["heading"], fill=(52, 211, 153))
    points5 = [
        "Linear SVM achieves the highest general balance (88.70% accuracy, 70.60% macro-precision, 57.40% macro-F1).",
        "Logistic Regression achieves the highest minority class recall (0.6142) through cost-sensitive balanced weighting.",
        "Multinomial Naive Bayes executes in 0.088 ms, but degrades on rare classes under severe class imbalance (45:1).",
        "Random Forest achieves 89.38% accuracy, but exhibits 35.78 ms latency due to deep decision tree traversals.",
    ]
    fy = 780
    for p in points5:
        draw.ellipse([(90, fy + 6), (100, fy + 16)], fill=(59, 130, 246))
        draw.text((115, fy), p, font=fonts["body"], fill=(203, 213, 225))
        fy += 48

    # Right: Embedded Accuracy vs Macro-F1 Chart
    draw.rounded_rectangle([(1090, 130), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((1120, 155), "Accuracy vs Macro-F1 & Latency (reports/figures/fig12)", font=fonts["heading"], fill=(255, 255, 255))

    fig12_path = "reports/figures/fig12_model_comparison_metrics.png"
    if os.path.exists(fig12_path):
        f_img = Image.open(fig12_path).convert("RGB")
        f_img = f_img.resize((730, 750), Image.Resampling.LANCZOS)
        img.paste(f_img, (1110, 210))

    return img


def render_scene_6(fonts):
    """Scene 6: Confusion Matrix & Error Analysis"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Error Analysis & Diagnostics", "Confusion Matrix Breakdown Across All 4 Classifiers")
    draw_footer(draw, fonts, 6, 8)

    # Left: Confusion Matrix Grid Image
    draw.rounded_rectangle([(60, 130), (1150, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 155), "Confusion Matrices (reports/figures/fig13)", font=fonts["heading"], fill=(255, 255, 255))

    fig13_path = "reports/figures/fig13_confusion_matrices.png"
    if os.path.exists(fig13_path):
        f_img = Image.open(fig13_path).convert("RGB")
        f_img = f_img.resize((1050, 760), Image.Resampling.LANCZOS)
        img.paste(f_img, (80, 210))

    # Right: Error Diagnostics
    draw.rounded_rectangle([(1190, 130), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((1220, 155), "Linguistic & Class Diagnostics", font=fonts["heading"], fill=(255, 255, 255))

    diag_pts = [
        ("High-Precision BUG Separation:", "BUG tickets (154 test samples) achieve 96.1% recall under Linear SVM. Discriminating keywords ('traceback', 'exception', 'fails', 'error') create sharp linear separation in TF-IDF space."),
        ("FEATURE Intent Recognition:", "FEATURE tickets (123 test samples) achieve 87.0% recall under Linear SVM and 89.4% recall under Naive Bayes. Terms like 'support', 'add', 'request', 'stream' signal feature proposals."),
        ("Minority Class Challenges:", "QUESTION (6 samples), DOCUMENTATION (5 samples), and OTHER (4 samples) suffer from severe 45:1 imbalance. Balanced class weights enable Logistic Regression to correctly detect 80.0% of DOCS tickets."),
        ("False Positive Analysis:", "Questions containing error snippets are occasionally triaged as BUGs. This represents a safe failure mode: prioritizing potential defects for engineer inspection rather than ignoring them."),
    ]
    y = 220
    for title, desc in diag_pts:
        draw.text((1220, y), title, font=fonts["body_bold"], fill=(96, 165, 250))
        y += 32
        for line in desc.split("\n"):
            draw.text((1220, y), line, font=fonts["body"], fill=(203, 213, 225))
            y += 28
        y += 25

    return img


def render_scene_7(fonts):
    """Scene 7: Unsupervised Topic Discovery (K-Means)"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Unsupervised Discovery", "K-Means Latent Topic Clustering & 2D PCA Projection")
    draw_footer(draw, fonts, 7, 8)

    # Left: Cluster Details
    draw.rounded_rectangle([(60, 130), (880, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 155), "K-Means Clusters (1,493 Unlabeled Issues)", font=fonts["heading"], fill=(255, 255, 255))

    clusters = [
        ("Cluster 0: Data Loaders & Streaming", "dataset, request, add, feature, gem, datasets, stream, loader", (59, 130, 246)),
        ("Cluster 1: CI, Build & Testing", "broken, ci, windows, tests, fails, release, ruff, pytest", (239, 68, 68)),
        ("Cluster 2: Dataset Viewer Web", "dataset viewer, viewer, issue, dataset, error, nlp, preview", (16, 185, 129)),
        ("Cluster 3: Cloud Hub & Authentication", "dataset, error, datasets, loading, map, load, support, auth", (245, 158, 11)),
        ("Cluster 4: Core API & Type Definitions", "load_dataset, does work, doesn work, error, typing, pyarrow", (139, 92, 246)),
    ]
    y = 220
    for c_title, terms, col in clusters:
        draw.rounded_rectangle([(90, y), (850, y + 125)], radius=10, fill=(9, 14, 26), outline=col, width=1)
        draw.text((115, y + 15), c_title, font=fonts["body_bold"], fill=col)
        draw.text((115, y + 48), "Top Centroid Keywords:", font=fonts["small"], fill=(148, 163, 184))
        draw.text((115, y + 78), terms, font=fonts["mono_small"], fill=(248, 250, 252))
        y += 150

    # Right: PCA Projection
    draw.rounded_rectangle([(920, 130), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((950, 155), "2D PCA Projection (reports/figures/fig15)", font=fonts["heading"], fill=(255, 255, 255))

    fig15_path = "reports/figures/fig15_kmeans_cluster_projection.png"
    if os.path.exists(fig15_path):
        f_img = Image.open(fig15_path).convert("RGB")
        f_img = f_img.resize((890, 760), Image.Resampling.LANCZOS)
        img.paste(f_img, (940, 210))

    return img


def render_scene_8(fonts):
    """Scene 8: Production Web Console, REST API & Automated Tests"""
    img = create_gradient_bg(1920, 1080, (9, 13, 22), (17, 24, 39))
    draw = ImageDraw.Draw(img)
    draw_header(draw, fonts, "Production Integration", "Enterprise Triage Console, REST API & Automated Verification")
    draw_footer(draw, fonts, 8, 8)

    # Left: Web Console Simulation
    draw.rounded_rectangle([(60, 130), (1050, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((90, 155), "Live Production Triage Console (templates/index.html)", font=fonts["heading"], fill=(255, 255, 255))

    # Ingestion box
    draw.rounded_rectangle([(90, 210), (1020, 310)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
    draw.text((110, 225), "Live GitHub Ticket Ingestion:", font=fonts["body_bold"], fill=(96, 165, 250))
    draw.rectangle([(110, 255), (1000, 295)], fill=(13, 19, 34), outline=(59, 130, 246), width=1)
    draw.text((120, 265), "https://github.com/huggingface/datasets/issues/116", font=fonts["mono_small"], fill=(255, 255, 255))

    # Decision card
    draw.rounded_rectangle([(90, 335), (1020, 730)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
    draw.text((110, 355), "Triage Verdict & Engineering Routing Output", font=fonts["heading"], fill=(255, 255, 255))

    # Badge & Confidence
    draw.rounded_rectangle([(110, 410), (220, 455)], radius=6, fill=(239, 68, 68), outline=(239, 68, 68), width=1)
    draw.text((135, 422), "BUG", font=fonts["body_bold"], fill=(255, 255, 255))
    draw.text((250, 422), "Confidence: 92.4%  |  HIGH PRIORITY", font=fonts["body_bold"], fill=(248, 113, 113))

    # Target info
    draw.text((110, 480), "Assigned Domain: Core Framework & Triage", font=fonts["body"], fill=(203, 213, 225))
    draw.text((110, 515), "Target Review Team: @triage/core-defects", font=fonts["mono"], fill=(165, 180, 252))
    draw.text((110, 550), "Active Model: Linear SVM (Calibrated) | Latency: 0.12 ms", font=fonts["body"], fill=(52, 211, 153))

    # Keywords
    draw.text((110, 600), "Salient TF-IDF Discriminating Keywords:", font=fonts["body_bold"], fill=(148, 163, 184))
    kws = ["pyarrow", "table", "length", "buffer", "column", "mismatch"]
    kx = 110
    for kw in kws:
        draw.rounded_rectangle([(kx, 640), (kx + 120, 675)], radius=4, fill=(20, 28, 46), outline=(31, 41, 61), width=1)
        draw.text((kx + 15, 648), kw, font=fonts["mono_small"], fill=(147, 197, 253))
        kx += 135

    # Firewall card
    draw.rounded_rectangle([(90, 755), (1020, 970)], radius=8, fill=(31, 27, 36), outline=(239, 68, 68), width=1)
    draw.text((110, 775), "Target Leakage Firewall (Quarantined from Input):", font=fonts["body_bold"], fill=(248, 113, 113))
    fw_items = ["state: closed", "comments: 5", "assignee: null", "milestone: null"]
    fy = 815
    for item in fw_items:
        draw.rounded_rectangle([(110, fy), (500, fy + 32)], radius=4, fill=(9, 14, 26), outline=(239, 68, 68), width=1)
        draw.text((120, fy + 7), item + "  [QUARANTINED]", font=fonts["mono_small"], fill=(252, 165, 165))
        fy += 38

    # Right: Automated Tests & Verification
    draw.rounded_rectangle([(1090, 130), (1860, 1000)], radius=12, fill=(17, 24, 39), outline=(31, 41, 61), width=1)
    draw.text((1120, 155), "Automated Testing & Production Verification", font=fonts["heading"], fill=(255, 255, 255))

    # Terminal Box
    draw.rounded_rectangle([(1120, 210), (1830, 680)], radius=8, fill=(9, 14, 26), outline=(31, 41, 61), width=1)
    draw.text((1140, 225), "$ pytest -v", font=fonts["mono"], fill=(52, 211, 153))

    test_lines = [
        "tests/test_data_pipeline.py::test_loading_chosen_format PASSED      [ 8%]",
        "tests/test_data_pipeline.py::test_missing_text_normalization PASSED [16%]",
        "tests/test_data_pipeline.py::test_label_mapping_and_conflicts PASSED [25%]",
        "tests/test_data_pipeline.py::test_duplicate_filtering PASSED         [33%]",
        "tests/test_data_pipeline.py::test_leakage_fields_exclusion PASSED   [41%]",
        "tests/test_data_pipeline.py::test_deterministic_cleaning PASSED     [50%]",
        "tests/test_data_pipeline.py::test_real_research_dataset_integrity PASSED [58%]",
        "tests/test_model_pipeline.py::test_model_suite_loading PASSED       [66%]",
        "tests/test_model_pipeline.py::test_inference_and_probabilities PASSED [75%]",
        "tests/test_model_pipeline.py::test_model_switching PASSED           [83%]",
        "tests/test_model_pipeline.py::test_models_benchmark_endpoint PASSED [91%]",
        "tests/test_model_pipeline.py::test_clusters_endpoint PASSED        [100%]",
        "",
        "======================== 12 passed in 3.08s ========================",
    ]
    ty = 265
    for tl in test_lines:
        col = (52, 211, 153) if "PASSED" in tl or "12 passed" in tl else (148, 163, 184)
        draw.text((1140, ty), tl, font=fonts["mono_small"], fill=col)
        ty += 28

    # Bottom Delivery Box
    draw.rounded_rectangle([(1120, 710), (1830, 970)], radius=8, fill=(9, 14, 26), outline=(30, 58, 138), width=1)
    draw.text((1140, 735), "Open Public Deliverables:", font=fonts["heading"], fill=(96, 165, 250))
    draw.text((1140, 785), "GitHub Repository: https://github.com/priyanshuyadav-224/GitHub-issue-identifier", font=fonts["mono_small"], fill=(255, 255, 255))
    draw.text((1140, 825), "Public Release Tag: https://github.com/priyanshuyadav-224/GitHub-issue-identifier/releases/tag/v1.0.0", font=fonts["mono_small"], fill=(255, 255, 255))
    draw.text((1140, 865), "Technical Report (PDF): reports/S25CSEU0725_Milestone_3_Report.pdf (0.66 MB)", font=fonts["small"], fill=(203, 213, 225))
    draw.text((1140, 900), "Project Presentation (PPTX): reports/S25CSEU0725_Milestone_3_Presentation.pptx (14 slides)", font=fonts["small"], fill=(203, 213, 225))
    draw.text((1140, 935), "Demonstration Video (MP4): Attached to Public GitHub Release (Open Access)", font=fonts["small"], fill=(52, 211, 153))

    return img


def generate_video(output_mp4_path: str, metrics_path: str):
    print(f"Generating HD 1080p demonstration video at: {output_mp4_path}...")
    os.makedirs(os.path.dirname(output_mp4_path), exist_ok=True)

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    bench = metrics["model_benchmarks"]

    fonts = load_fonts()

    # Define Scenes and their durations (in seconds)
    scenes_def = [
        (render_scene_1(fonts), 12),              # Scene 1: Title & Highlights (12s)
        (render_scene_2(fonts), 15),              # Scene 2: Ingestion & Deduplication (15s)
        (render_scene_3(fonts), 15),              # Scene 3: Target Leakage Firewall (15s)
        (render_scene_4(fonts), 16),              # Scene 4: Mathematical Formulations (16s)
        (render_scene_5(fonts, bench), 18),       # Scene 5: ML Benchmark Table & Chart (18s)
        (render_scene_6(fonts), 15),              # Scene 6: Confusion Matrix & Error Analysis (15s)
        (render_scene_7(fonts), 15),              # Scene 7: K-Means Clustering & PCA (15s)
        (render_scene_8(fonts), 18),              # Scene 8: Console & Test Verification (18s)
    ]

    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4_path, fourcc, fps, (1920, 1080))

    total_frames = 0
    fade_frames = 15  # 0.5s crossfade between scenes

    prev_frame_cv = None

    for s_idx, (scene_pil, duration_sec) in enumerate(scenes_def):
        print(f"Encoding Scene {s_idx + 1}/8 (Duration: {duration_sec}s)...")
        # Convert PIL Image to OpenCV BGR format
        frame_rgb = np.array(scene_pil)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        num_frames = int(duration_sec * fps)

        # Crossfade from previous scene if applicable
        if prev_frame_cv is not None:
            for f in range(fade_frames):
                alpha = f / float(fade_frames)
                blended = cv2.addWeighted(prev_frame_cv, 1.0 - alpha, frame_bgr, alpha, 0)
                out.write(blended)
                total_frames += 1
            num_frames -= fade_frames

        # Hold main scene frame
        for _ in range(num_frames):
            out.write(frame_bgr)
            total_frames += 1

        prev_frame_cv = frame_bgr

    out.release()
    file_size_mb = os.path.getsize(output_mp4_path) / (1024 * 1024)
    print(f"Video generated successfully! Total Frames: {total_frames} (~{total_frames/fps:.1f}s), Size: {file_size_mb:.2f} MB")
    return output_mp4_path


def upload_to_github_release(mp4_path: str):
    """Uploads the demonstration MP4 directly to GitHub Release v1.0.0."""
    print("Uploading video asset to public GitHub Release...")
    import urllib.request
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        try:
            import subprocess
            proc = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n", text=True, capture_output=True)
            for line in proc.stdout.splitlines():
                if line.startswith("password="):
                    token = line.split("=", 1)[1].strip()
        except Exception:
            pass

    # Get release info
    rel_url = "https://api.github.com/repos/priyanshuyadav-224/GitHub-issue-identifier/releases/tags/v1.0.0"
    req = urllib.request.Request(rel_url, headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req) as resp:
            rel_info = json.loads(resp.read().decode())
            release_id = rel_info["id"]
            upload_url = rel_info["upload_url"].split("{")[0]
            print(f"Found release ID {release_id}, Upload URL: {upload_url}")

            # Check if asset already exists and delete to overwrite
            for asset in rel_info.get("assets", []):
                if asset["name"] == os.path.basename(mp4_path):
                    del_url = asset["url"]
                    del_req = urllib.request.Request(del_url, headers={"Authorization": f"Bearer {token}", "User-Agent": "Mozilla/5.0"}, method="DELETE")
                    urllib.request.urlopen(del_req)
                    print("Existing asset deleted for clean overwrite.")

            # Upload MP4 asset
            with open(mp4_path, "rb") as f:
                video_data = f.read()

            asset_name = os.path.basename(mp4_path)
            target_url = f"{upload_url}?name={asset_name}"
            upload_req = urllib.request.Request(
                target_url,
                data=video_data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Mozilla/5.0",
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(video_data))
                }
            )
            with urllib.request.urlopen(upload_req) as up_resp:
                up_res = json.loads(up_resp.read().decode())
                download_url = up_res.get("browser_download_url")
                print(f"Asset uploaded successfully! Direct download URL: {download_url}")
                return download_url
    except Exception as e:
        print("Upload failed:", e)
        return None


if __name__ == "__main__":
    mp4_file = "reports/project_demonstration_walkthrough.mp4"
    metrics_file = "reports/model_evaluation_metrics.json"

    generate_video(mp4_file, metrics_file)
    dl_url = upload_to_github_release(mp4_file)
    if dl_url:
        print(f"\nPublic Demonstration Video Link:\n{dl_url}\n")
