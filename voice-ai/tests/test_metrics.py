"""
Unit tests for VAJRA Voice AI — evaluation/metrics.py

Covers:
    - MetricsReport dataclass
    - compute_metrics with perfect, random, and edge-case inputs
    - compute_eer (Equal Error Rate) computation
    - compute_roc_auc (ROC-AUC) computation
    - Empty / degenerate input handling
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the voice-ai package root is importable.
_VOICE_AI_ROOT = str(Path(__file__).resolve().parent.parent)
if _VOICE_AI_ROOT not in sys.path:
    sys.path.insert(0, _VOICE_AI_ROOT)

from evaluation.metrics import MetricsReport, compute_eer, compute_metrics, compute_roc_auc


# ═══════════════════════════════════════════════════════════════════════════
# compute_metrics
# ═══════════════════════════════════════════════════════════════════════════
class TestComputeMetrics:

    def test_perfect_predictions(self):
        labels = np.array([0, 0, 1, 1, 1])
        preds = np.array([0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8, 0.95])
        report = compute_metrics(labels, preds, scores)
        assert isinstance(report, MetricsReport)
        assert report.accuracy == 1.0
        assert report.precision == 1.0
        assert report.recall == 1.0
        assert report.f1_score == 1.0
        assert report.num_samples == 5

    def test_all_wrong_predictions(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([1, 1, 0, 0])
        report = compute_metrics(labels, preds)
        assert report.accuracy == 0.0
        assert report.precision == 0.0
        assert report.recall == 0.0
        assert report.f1_score == 0.0

    def test_class_distribution(self):
        labels = np.array([0, 0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1, 1])
        report = compute_metrics(labels, preds)
        assert report.class_distribution == {0: 3, 1: 2}
        assert report.num_samples == 5

    def test_empty_input(self):
        labels = np.array([])
        preds = np.array([])
        report = compute_metrics(labels, preds)
        assert report.num_samples == 0
        assert report.accuracy == 0.0

    def test_roc_auc_with_scores(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        report = compute_metrics(labels, preds, scores)
        assert report.roc_auc > 0.9

    def test_roc_auc_without_scores(self):
        labels = np.array([0, 1])
        preds = np.array([0, 1])
        report = compute_metrics(labels, preds)
        assert report.roc_auc == 0.0  # No scores provided


# ═══════════════════════════════════════════════════════════════════════════
# compute_eer
# ═══════════════════════════════════════════════════════════════════════════
class TestComputeEER:

    def test_perfect_separation(self):
        """Perfectly separable: EER should be 0."""
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        eer = compute_eer(labels, scores)
        assert eer == pytest.approx(0.0, abs=0.05)

    def test_random_scores(self):
        """Random scores → EER near 0.5."""
        rng = np.random.default_rng(42)
        n = 1000
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        scores = rng.random(n)
        eer = compute_eer(labels, scores)
        assert 0.3 < eer < 0.7

    def test_empty_input(self):
        assert compute_eer(np.array([]), np.array([])) == 0.0

    def test_single_class(self):
        labels = np.array([1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7])
        assert compute_eer(labels, scores) == 0.0

    def test_return_value_in_range(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.3, 0.4, 0.6, 0.7, 0.5, 0.8])
        eer = compute_eer(labels, scores)
        assert 0.0 <= eer <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# compute_roc_auc
# ═══════════════════════════════════════════════════════════════════════════
class TestComputeROCAUC:

    def test_perfect_separation(self):
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        auc = compute_roc_auc(labels, scores)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_random_scores(self):
        rng = np.random.default_rng(42)
        n = 1000
        labels = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
        scores = rng.random(n)
        auc = compute_roc_auc(labels, scores)
        assert 0.4 < auc < 0.6

    def test_inverse_scores(self):
        """Inverted scores → AUC near 0."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        auc = compute_roc_auc(labels, scores)
        assert auc < 0.1

    def test_empty_input(self):
        assert compute_roc_auc(np.array([]), np.array([])) == 0.0

    def test_single_class(self):
        labels = np.array([0, 0, 0])
        scores = np.array([0.5, 0.6, 0.7])
        assert compute_roc_auc(labels, scores) == 0.0

    def test_return_value_clamped(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.3, 0.7, 0.2, 0.8])
        auc = compute_roc_auc(labels, scores)
        assert 0.0 <= auc <= 1.0
