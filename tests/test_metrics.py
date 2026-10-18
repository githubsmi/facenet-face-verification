import numpy as np
import pytest

from evaluation.metrics import (
    choose_threshold,
    compute_metrics,
    save_confusion_plot,
    save_roc_plot,
)


def test_choose_threshold_returns_finite_value():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])

    threshold = choose_threshold(labels, scores)

    assert np.isfinite(threshold)
    assert 0.0 <= threshold <= 1.0


def test_choose_threshold_requires_both_classes():
    labels = np.array([1, 1, 1])
    scores = np.array([0.7, 0.8, 0.9])

    with pytest.raises(ValueError, match="both classes"):
        choose_threshold(labels, scores)


def test_compute_metrics_returns_expected_values():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    threshold = 0.35

    metrics = compute_metrics(labels, scores, threshold)

    assert metrics["num_pairs"] == 4
    assert metrics["true_positive"] == 2
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1_score"] == pytest.approx(0.8)
    assert metrics["roc_auc"] == pytest.approx(0.75)
    assert metrics["far"] == pytest.approx(0.5)
    assert metrics["frr"] == pytest.approx(0.0)


def test_compute_metrics_rejects_empty_input():
    with pytest.raises(ValueError, match="equal, nonzero lengths"):
        compute_metrics([], [], 0.5)


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal, nonzero lengths"):
        compute_metrics([0, 1], [0.2], 0.5)


def test_compute_metrics_requires_both_classes():
    with pytest.raises(ValueError, match="both positive and negative"):
        compute_metrics([1, 1], [0.7, 0.9], 0.5)


def test_save_roc_plot_creates_file(tmp_path):
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    output_path = tmp_path / "plots" / "roc_curve.png"

    save_roc_plot(labels, scores, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_confusion_plot_creates_file(tmp_path):
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    output_path = tmp_path / "plots" / "confusion_matrix.png"

    save_confusion_plot(labels, scores, 0.5, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
