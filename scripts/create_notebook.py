"""
create_notebook.py
==================
Generates notebooks/milestone_2_eda.ipynb deterministically.
"""

import json

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Milestone 2: Dataset Collection, Preprocessing, and Exploratory Data Analysis (EDA)\n",
                "**Project:** GitHub Project Issue Classifier  \n",
                "**Student:** Priyanshu Yadav (S25CSEU0725) | Batch-25  \n",
                "**Objective:** Clean, inspect, and analyze real-world GitHub issues from the `huggingface/datasets` repository.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "# Set plot styling\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams[\"figure.figsize\"] = (10, 5)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Load Processed and Supervised Datasets\n",
                "Loading the deterministically cleaned parquet files produced by `scripts/preprocess_dataset.py`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_cleaned = pd.read_parquet(\"../data/processed/github_issues_cleaned.parquet\")\n",
                "df_supervised = pd.read_parquet(\"../data/processed/github_issues_supervised.parquet\")\n",
                "\n",
                "print(\"Pure cleaned issues:\", len(df_cleaned))\n",
                "print(\"Supervised labeled issues:\", len(df_supervised))\n",
                "df_supervised.head(3)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Target Class Distribution & Imbalance Analysis\n",
                "Auditing class counts across the 5 mapped categories."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [],
            "source": [
                "counts = df_supervised[\"target_class\"].value_counts()\n",
                "pcts = df_supervised[\"target_class\"].value_counts(normalize=True) * 100\n",
                "dist_df = pd.DataFrame({\"Count\": counts, \"Percentage (%)\": pcts.round(2)})\n",
                "print(dist_df)\n",
                "\n",
                "plt.figure(figsize=(8, 4), dpi=150)\n",
                "sns.barplot(x=counts.index, y=counts.values, palette=\"viridis\")\n",
                "plt.title(\"Target Class Distribution (Supervised Dataset)\", fontsize=13, fontweight=\"bold\")\n",
                "plt.ylabel(\"Number of Issues\")\n",
                "plt.xlabel(\"Class\")\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Text Length Analysis\n",
                "Analyzing title and body character lengths and word counts."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Text Summary Statistics:\")\n",
                "print(df_supervised[[\"title_char_len\", \"body_char_len\", \"text_word_count\"]].describe().round(1))\n",
                "\n",
                "fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=150)\n",
                "sns.histplot(df_supervised[\"title_char_len\"], bins=30, kde=True, ax=ax[0], color=\"royalblue\")\n",
                "ax[0].set_title(\"Title Character Length\")\n",
                "sns.boxplot(x=\"target_class\", y=\"text_word_count\", data=df_supervised, ax=ax[1], palette=\"Set2\", showfliers=False)\n",
                "ax[1].set_title(\"Word Count by Issue Type\")\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Temporal Distribution\n",
                "Inspecting arrival patterns over time."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [],
            "source": [
                "df_cleaned[\"created_dt\"] = pd.to_datetime(df_cleaned[\"created_at\"])\n",
                "monthly = df_cleaned.set_index(\"created_dt\").resample(\"ME\").size()\n",
                "\n",
                "plt.figure(figsize=(10, 4), dpi=150)\n",
                "monthly.plot(color=\"darkslateblue\", linewidth=2)\n",
                "plt.title(\"Monthly GitHub Issue Submissions (2020 - 2025)\", fontweight=\"bold\")\n",
                "plt.ylabel(\"Issues Count\")\n",
                "plt.xlabel(\"Creation Date\")\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.14.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("c:/Projects/notebooks/milestone_2_eda.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Created notebooks/milestone_2_eda.ipynb successfully.")
