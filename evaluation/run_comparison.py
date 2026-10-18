
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve

from evaluation.dataset import load_lfw_pairs
from evaluation.metrics import (
    choose_threshold,
    compute_metrics,
    save_confusion_plot,
    save_roc_plot,
)
from evaluation.run_evaluation import (
    EmbeddingCache,
    save_json,
    save_predictions,
    save_skipped_pairs,
    score_pairs,
)
from src.facenet_model import load_models


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "comparison"

CONFIGS = {
    "resize_1_0": 1.0,
    "resize_0_5": 0.5,
    "resize_1_5": 1.5,
}


def evaluate_configuration(name, resize, cache):
    print(f"\n{'=' * 60}")
    print(f"Configuration: {name} | resize={resize}")
    print(f"{'=' * 60}")

    dev_pairs, dev_labels, dev_indices = load_lfw_pairs(
        subset="train",
        random_state=42,
        resize=resize,
        return_indices=True,
    )

    dev_indices, dev_labels, dev_scores, dev_skipped = score_pairs(
        dev_pairs,
        dev_labels,
        dev_indices,
        cache,
        f"{name}_development",
    )

    if len(dev_labels) == 0 or len(np.unique(dev_labels)) < 2:
        raise RuntimeError(
            f"{name}: development data must contain both classes."
        )

    threshold = float(choose_threshold(dev_labels, dev_scores))
    print(f"Development threshold: {threshold:.6f}")

    test_pairs, test_labels, test_indices = load_lfw_pairs(
        subset="test",
        random_state=43,
        resize=resize,
        return_indices=True,
    )

    test_indices, test_labels, test_scores, test_skipped = score_pairs(
        test_pairs,
        test_labels,
        test_indices,
        cache,
        f"{name}_test",
    )

    if len(test_labels) == 0:
        raise RuntimeError(f"{name}: no valid test pairs.")

    return {
        "name": name,
        "resize": resize,
        "threshold": threshold,
        "dev_total": len(dev_pairs),
        "dev_indices": dev_indices,
        "dev_labels": dev_labels,
        "dev_scores": dev_scores,
        "dev_skipped": dev_skipped,
        "test_total": len(test_pairs),
        "test_indices": test_indices,
        "test_labels": test_labels,
        "test_scores": test_scores,
        "test_skipped": test_skipped,
    }


def save_comparison_roc(labels, results, output_path):
    plt.figure(figsize=(8, 6))

    for result in results:
        fpr, tpr, _ = roc_curve(
            labels,
            result["common_scores"],
        )
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f'{result["name"]} (AUC = {roc_auc:.4f})',
        )

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Face Verification ROC Comparison")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare FaceNet input resize settings on LFW."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Number of CPU threads to use.",
    )
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be at least 1")

    torch.set_num_threads(args.threads)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading pretrained FaceNet and MTCNN...")
    mtcnn, model = load_models()
    cache = EmbeddingCache(model, mtcnn)

    results = []

    for name, resize in CONFIGS.items():
        result = evaluate_configuration(name, resize, cache)
        results.append(result)

    cache.save()

    common_indices = set(results[0]["test_indices"].tolist())

    for result in results[1:]:
        common_indices.intersection_update(
            result["test_indices"].tolist()
        )

    common_indices = np.asarray(
        sorted(common_indices),
        dtype=np.int64,
    )

    if len(common_indices) == 0:
        raise RuntimeError(
            "No shared test pairs were successfully processed."
        )

    for result in results:
        positions = {
            int(index): position
            for position, index in enumerate(result["test_indices"])
        }

        selected_positions = np.asarray(
            [positions[int(index)] for index in common_indices],
            dtype=np.int64,
        )

        result["common_labels"] = result["test_labels"][
            selected_positions
        ]
        result["common_scores"] = result["test_scores"][
            selected_positions
        ]

    common_labels = results[0]["common_labels"]

    for result in results[1:]:
        if not np.array_equal(
            common_labels,
            result["common_labels"],
        ):
            raise RuntimeError(
                "Test labels do not match for shared pair indices."
            )

    if len(np.unique(common_labels)) < 2:
        raise RuntimeError(
            "The common test subset must contain both classes."
        )

    summary_rows = []

    for result in results:
        name = result["name"]
        resize = result["resize"]
        threshold = result["threshold"]

        output = OUTPUT_DIR / name
        output.mkdir(parents=True, exist_ok=True)

        labels = result["common_labels"]
        scores = result["common_scores"]

        metrics = compute_metrics(labels, scores, threshold)

        test_skipped_count = len({
            item["pair_index"]
            for item in result["test_skipped"]
        })
        dev_skipped_count = len({
            item["pair_index"]
            for item in result["dev_skipped"]
        })

        metrics.update({
            "configuration": name,
            "resize": resize,
            "model": "InceptionResnetV1 pretrained on VGGFace2",
            "dataset": "Labeled Faces in the Wild (LFW)",
            "threshold_source": "development subset",
            "development_pairs_total": result["dev_total"],
            "development_pairs_used": len(result["dev_labels"]),
            "development_pairs_skipped": dev_skipped_count,
            "development_coverage": (
                len(result["dev_labels"]) / result["dev_total"]
            ),
            "test_pairs_total": result["test_total"],
            "test_pairs_used_individual": len(result["test_labels"]),
            "test_pairs_skipped_individual": test_skipped_count,
            "test_coverage_individual": (
                len(result["test_labels"]) / result["test_total"]
            ),
            "common_test_pairs_used": len(common_indices),
            "common_test_coverage": (
                len(common_indices) / result["test_total"]
            ),
        })

        save_json(metrics, output / "metrics.json")

        save_predictions(
            common_indices,
            labels,
            scores,
            threshold,
            output / "test_predictions.csv",
        )

        save_roc_plot(
            labels,
            scores,
            output / "roc_curve.png",
        )

        save_confusion_plot(
            labels,
            scores,
            threshold,
            output / "confusion_matrix.png",
        )

        save_skipped_pairs(
            result["dev_skipped"] + result["test_skipped"],
            output / "skipped_pairs.csv",
        )

        summary_rows.append(metrics)

        print(f"\n{name} results on common test pairs:")
        print(json.dumps(metrics, indent=2))

    summary_path = OUTPUT_DIR / "comparison_metrics.csv"

    with summary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    save_comparison_roc(
        common_labels,
        results,
        OUTPUT_DIR / "comparison_roc.png",
    )

    report = {
        "experiment": "Controlled LFW input resize comparison",
        "model": "InceptionResnetV1 pretrained on VGGFace2",
        "detector": "MTCNN",
        "configurations": CONFIGS,
        "development_random_state": 42,
        "test_random_state": 43,
        "threshold_selection": "Development subset only",
        "comparison_method": (
            "Metrics calculated on the intersection of test pairs "
            "successfully processed by both configurations."
        ),
        "test_pairs_total": results[0]["test_total"],
        "common_test_pairs_used": len(common_indices),
        "configurations_results": summary_rows,
    }

    save_json(
        report,
        OUTPUT_DIR / "comparison_report.json",
    )

    print("\nComparison complete.")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Common test pairs: {len(common_indices)}")
    print(f"Metrics table: {summary_path}")


if __name__ == "__main__":
    main()
