"""
generate_pdf_and_pptx.py
========================
Generates:
1. reports/milestone_2_report.pdf (Comprehensive PDF document)
2. reports/milestone_2_presentation.pptx (Milestone progress slide deck with embedded EDA figures)
Author: Priyanshu Yadav (S25CSEU0725)
Project: GitHub Project Issue Classifier
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


def generate_pdf(output_pdf_path: str):
    print(f"Generating PDF report: {output_pdf_path}...")
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#2b6cb0"),
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#2d3748"),
        spaceAfter=6,
    )

    story = []

    # Title Banner
    story.append(Paragraph("GitHub Project Issue Classifier", title_style))
    story.append(
        Paragraph(
            "<b>Milestone 2 Report:</b> Dataset Collection, Preprocessing & Exploratory Data Analysis (EDA)<br/>"
            "<b>Author:</b> Priyanshu Yadav (S25CSEU0725) | Batch-25 | Date: October 15, 2026",
            subtitle_style,
        )
    )
    story.append(Spacer(1, 10))

    # Executive Summary
    story.append(Paragraph("1. Executive Summary & Verification Outcomes", h2_style))
    exec_text = (
        "Milestone 2 establishes a fully tested, reproducible data engineering pipeline for the "
        "GitHub Project Issue Classifier. Building upon Milestone 1's architecture, we acquired the "
        "public <b>helmo/github-issues</b> dataset (Apache 2.0, 7,540 raw records) derived from the "
        "foundational <i>huggingface/datasets</i> repository. Through automated data quality auditing, "
        "a 100-row duplication block was diagnosed and resolved, 4,325 pull requests were filtered out, "
        "and a leak-free 5-class target taxonomy was established yielding <b>1,459 supervised issues</b> "
        "and 1,493 unlabeled issues for clustering extensions."
    )
    story.append(Paragraph(exec_text, body_style))
    story.append(Spacer(1, 8))

    # Quantitative Metrics Table
    story.append(Paragraph("2. Dataset Pipeline Quantitative Summary", h2_style))
    table_data = [
        ["Pipeline Stage / Metric", "Count", "Percentage", "Engineering Action"],
        ["Raw Ingested Records", "7,540", "100.0%", "Source Parquet verified from Hugging Face"],
        ["Unique Entities (Deduplicated)", "7,440", "98.67%", "Removed 100 duplicate row entries (indices 0..99)"],
        ["Pull Requests Excluded", "4,325", "58.13%", "Filtered out to prevent code review domain skew"],
        ["Pure GitHub Issues", "3,115", "41.87%", "Legitimate problem reports retained for triage"],
        ["Supervised Labeled Dataset", "1,459", "19.35%", "Clean single-label ground truth for ML training"],
        ["-- BUG Category", "771", "52.84%", "Mapped from bug, dataset bug, metric bug"],
        ["-- FEATURE Category", "615", "42.15%", "Mapped from enhancement, dataset request"],
        ["-- QUESTION Category", "31", "2.12%", "Mapped from question, generic discussion"],
        ["-- DOCUMENTATION Category", "25", "1.71%", "Mapped from documentation tag"],
        ["-- OTHER Category", "17", "1.17%", "Mapped from maintenance, refactoring"],
        ["Unlabeled Issues Reserved", "1,493", "47.93%", "Preserved for Milestone 6 K-Means clustering"],
        ["Conflicting Multi-Label Issues", "34", "1.09%", "Excluded from single-label training (e.g. bug+feat)"],
    ]
    t = Table(table_data, colWidths=[180, 55, 65, 230])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 12))

    # Data Leakage & Feature Allowlist
    story.append(Paragraph("3. Data Leakage Prevention Architecture", h2_style))
    leak_text = (
        "To ensure valid predictive modeling at issue creation time, a strict <b>Feature Allowlist</b> "
        "was enforced. All 12 post-triage fields (issue closure <i>state</i>, <i>closed_at</i>, "
        "<i>comments</i> count and text, <i>assignee</i>, <i>milestone</i>, and <i>reactions</i>) are "
        "strictly excluded from features because they do not exist when an author submits a ticket. "
        "Only creation-time features (<i>clean_title</i>, <i>clean_body</i>, combined <i>text</i>, "
        "<i>author_association</i>, and <i>created_at</i>) are permitted."
    )
    story.append(Paragraph(leak_text, body_style))
    story.append(Spacer(1, 10))

    # Selected EDA Figures
    fig_dir = "c:/Projects/reports/figures"
    fig1 = os.path.join(fig_dir, "fig01_filtering_pipeline_counts.png")
    fig2 = os.path.join(fig_dir, "fig02_class_frequency_imbalance.png")
    if os.path.exists(fig1) and os.path.exists(fig2):
        story.append(Paragraph("4. Key Exploratory Visualizations", h2_style))
        story.append(RLImage(fig1, width=480, height=200))
        story.append(Spacer(1, 6))
        story.append(RLImage(fig2, width=480, height=210))
        story.append(Spacer(1, 10))

    # Automated Testing
    story.append(Paragraph("5. Automated Testing & Verification", h2_style))
    test_text = (
        "Automated unit and integration tests (<code>tests/test_data_pipeline.py</code>) were run using pytest. "
        "All 7 test suites passed with 100% success (0 failures): format loading, missing text normalization, "
        "label mapping, duplicate handling, pull-request filtering, leakage field exclusion, and real dataset integrity."
    )
    story.append(Paragraph(test_text, body_style))

    doc.build(story)
    print(f"PDF generated successfully at: {output_pdf_path}")


def generate_pptx(output_pptx_path: str):
    print(f"Generating PPTX presentation: {output_pptx_path}...")
    os.makedirs(os.path.dirname(output_pptx_path), exist_ok=True)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def add_header(slide, title_text, category="GITHUB PROJECT ISSUE CLASSIFIER | MILESTONE 2"):
        tx_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.9))
        tf = tx_box.text_frame
        tf.word_wrap = True
        p_cat = tf.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)

        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(0x1A, 0x20, 0x2C)

    # Slide 1: Title Slide
    s1 = prs.slides.add_slide(blank_layout)
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(11.3), Inches(3.5))
    tf1 = t_box.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "GitHub Project Issue Classifier"
    p1.font.size = Pt(36)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    p2 = tf1.add_paragraph()
    p2.text = "Milestone 2: Dataset Collection, Preprocessing & Exploratory Data Analysis"
    p2.font.size = Pt(20)
    p2.font.color.rgb = RGBColor(0x2B, 0x6C, 0xB0)
    p2.space_before = Pt(10)

    p3 = tf1.add_paragraph()
    p3.text = "Candidate: Priyanshu Yadav (S25CSEU0725) | Batch-25\nCollege Progress Report & Mid-Term Viva Presentation"
    p3.font.size = Pt(14)
    p3.font.color.rgb = RGBColor(0x71, 0x80, 0x96)
    p3.space_before = Pt(25)

    # Slide 2: Ingestion & Filtering Pipeline
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Data Ingestion & Multi-Stage Quality Filtering Funnel")
    fig1 = "c:/Projects/reports/figures/fig01_filtering_pipeline_counts.png"
    if os.path.exists(fig1):
        s2.shapes.add_picture(fig1, Inches(0.8), Inches(1.5), width=Inches(6.8))
    t_box = s2.shapes.add_textbox(Inches(7.8), Inches(1.5), Inches(4.8), Inches(5.0))
    tf2 = t_box.text_frame
    tf2.word_wrap = True
    bullets = [
        ("Source Verified:", "Acquired 7,540 raw records from huggingface/datasets (Apache 2.0)."),
        ("100 Duplicates Removed:", "Identified exact duplicate block (rows 0..99 == 100..199)."),
        ("4,325 PRs Filtered:", "Pull requests separated to eliminate commit/code-review skew."),
        ("3,115 Pure Issues:", "Complete title text (0 nulls); normalized missing body descriptions."),
        ("1,459 Supervised Subset:", "Unambiguous 5-class target instances ready for ML training."),
    ]
    for i, (head, desc) in enumerate(bullets):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.text = f"{head} {desc}"
        p.font.size = Pt(13)
        p.space_after = Pt(12)

    # Slide 3: Target Class Distribution & Imbalance
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Target Taxonomy & 45:1 Class Imbalance Analysis")
    fig2 = "c:/Projects/reports/figures/fig02_class_frequency_imbalance.png"
    if os.path.exists(fig2):
        s3.shapes.add_picture(fig2, Inches(0.8), Inches(1.5), width=Inches(6.8))
    t_box = s3.shapes.add_textbox(Inches(7.8), Inches(1.5), Inches(4.8), Inches(5.0))
    tf3 = t_box.text_frame
    tf3.word_wrap = True
    bullets3 = [
        ("Defensible 5 Classes:", "BUG (771), FEATURE (615), QUESTION (31), DOCUMENTATION (25), OTHER (17)."),
        ("Imbalance Ratio:", "45:1 ratio between majority (BUG) and minority (OTHER)."),
        ("Why Accuracy Fails:", "A naive majority classifier yields 52.8% accuracy while failing on all questions."),
        ("Milestone 3 Mandate:", "Evaluation strictly requires Macro-F1, Balanced Accuracy, and class-weighted loss."),
        ("Unsupervised Pool:", "1,493 unlabeled issues reserved for K-Means discovery in Milestone 6."),
    ]
    for i, (head, desc) in enumerate(bullets3):
        p = tf3.paragraphs[0] if i == 0 else tf3.add_paragraph()
        p.text = f"{head} {desc}"
        p.font.size = Pt(13)
        p.space_after = Pt(12)

    # Slide 4: Data Leakage Prevention Architecture
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Scientific Validity: Rigorous Data Leakage Firewall")
    t_box = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.0))
    tf4 = t_box.text_frame
    tf4.word_wrap = True
    p = tf4.paragraphs[0]
    p.text = "ALLOWED CREATION-TIME FEATURES"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x27, 0xAE, 0x60)
    for feat, note in [
        ("clean_title", "Author-submitted summary"),
        ("clean_body", "Author-submitted description"),
        ("text", "Documented combined title + body"),
        ("author_association", "NONE, CONTRIBUTOR, MEMBER"),
        ("created_at", "Submission timestamp"),
        ("title/body lengths", "Derived char and word counts"),
    ]:
        p_sub = tf4.add_paragraph()
        p_sub.text = f"• {feat}: {note}"
        p_sub.font.size = Pt(12)
        p_sub.space_after = Pt(8)

    t_box2 = s4.shapes.add_textbox(Inches(6.8), Inches(1.5), Inches(5.6), Inches(5.0))
    tf4_2 = t_box2.text_frame
    tf4_2.word_wrap = True
    p2 = tf4_2.paragraphs[0]
    p2.text = "EXCLUDED LEAKY POST-TRIAGE FIELDS"
    p2.font.bold = True
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
    for leak, note in [
        ("state (closed/open)", "Resolution determined post-triage"),
        ("closed_at / closed_by", "Closure timestamp and resolver"),
        ("comments count & text", "Discussions accumulate over time"),
        ("assignee / assignees", "Maintainers assigned during triage"),
        ("milestone", "Target release assigned by leads"),
        ("reactions (+1, hearts)", "Community votes added over time"),
    ]:
        p_sub2 = tf4_2.add_paragraph()
        p_sub2.text = f"X {leak}: {note}"
        p_sub2.font.size = Pt(12)
        p_sub2.space_after = Pt(8)

    # Slide 5: Linguistic Discovery (N-Grams) & Verification
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Linguistic Discrimination (N-Grams) & Test Verification")
    fig9 = "c:/Projects/reports/figures/fig09_ngram_frequency_analysis.png"
    if os.path.exists(fig9):
        s5.shapes.add_picture(fig9, Inches(0.8), Inches(1.5), width=Inches(7.2))
    t_box = s5.shapes.add_textbox(Inches(8.3), Inches(1.5), Inches(4.3), Inches(5.0))
    tf5 = t_box.text_frame
    tf5.word_wrap = True
    p = tf5.paragraphs[0]
    p.text = "VERIFIED TEST SUITE (100% PASS)"
    p.font.bold = True
    p.font.size = Pt(14)
    p.font.color.rgb = RGBColor(0x29, 0x80, 0xB9)
    for test_item in [
        "test_loading_chosen_format: PASSED",
        "test_missing_text_normalization: PASSED",
        "test_label_mapping_and_conflicts: PASSED",
        "test_duplicate_and_pr_filtering: PASSED",
        "test_leakage_fields_exclusion: PASSED",
        "test_deterministic_cleaning: PASSED",
        "test_real_research_dataset_integrity: PASSED",
    ]:
        p_t = tf5.add_paragraph()
        p_t.text = f"✓ {test_item}"
        p_t.font.size = Pt(11)
        p_t.space_after = Pt(6)

    prs.save(output_pptx_path)
    print(f"PPTX presentation generated successfully at: {output_pptx_path}")


if __name__ == "__main__":
    generate_pdf("c:/Projects/reports/milestone_2_report.pdf")
    generate_pptx("c:/Projects/reports/milestone_2_presentation.pptx")

