"""
tests/test_model_pipeline.py
============================
Automated tests for Statistical Machine Learning model inference,
calibrated probability output, priority calculation, routing, and Flask API.
"""

import os
import pytest
import numpy as np
from app import app, MODEL_SUITE, VECTORIZER, CLASSES


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_model_suite_loading():
    """Verify that all 4 candidate statistical ML models are initialized."""
    expected_models = ["Logistic Regression", "Linear SVM", "Multinomial Naive Bayes", "Random Forest"]
    for m in expected_models:
        assert m in MODEL_SUITE, f"Model {m} missing from MODEL_SUITE"
    assert len(CLASSES) == 5, f"Expected 5 classes, got {len(CLASSES)}"
    assert VECTORIZER is not None


def test_inference_and_probabilities(client):
    """Verify that prediction returns valid classes, confidence, and calibrated probability distribution."""
    payload = {
        "title": "TypeError: PyArrow buffer length mismatch when loading parquet",
        "body": "Traceback: File datasets/arrow_dataset.py line 204: buffer allocation error.",
        "author_association": "NONE",
        "model": "Linear SVM"
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert data["status"] == "success"
    assert data["prediction"] in CLASSES
    assert 0.0 <= data["confidence"] <= 100.0
    assert len(data["probabilities"]) == 5
    prob_sum = sum(data["probabilities"].values())
    assert pytest.approx(prob_sum, abs=0.02) == 1.0
    assert data["priority"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert "routing_area" in data
    assert "routing_team" in data
    assert len(data["keywords"]) > 0


def test_model_switching(client):
    """Verify that user can select between different statistical ML models."""
    sample = {
        "title": "Feature: Add audio dataset streaming support",
        "body": "Proposing torchaudio iterable chunk loading."
    }
    for model_name in ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVM", "Random Forest"]:
        sample["model"] = model_name
        res = client.post("/api/predict", json=sample)
        assert res.status_code == 200
        data = res.get_json()
        assert data["model_used"] == model_name
        assert data["prediction"] in CLASSES


def test_models_benchmark_endpoint(client):
    """Verify that /api/models returns complete comparative benchmark data."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.get_json()
    assert "available_models" in data
    assert len(data["available_models"]) >= 4
    assert "benchmark_summary" in data


def test_clusters_endpoint(client):
    """Verify that /api/clusters returns K-Means topic clusters."""
    res = client.get("/api/clusters")
    assert res.status_code == 200
    data = res.get_json()
    assert "clusters" in data
    assert len(data["clusters"]) == 5
