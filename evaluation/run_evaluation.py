
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import sklearn
from importlib.metadata import version

from evaluation.dataset import load_lfw_pairs
from evaluation.metrics import (
    choose_threshold,
    compute_metrics,
    save_confusion_plot,
    save_roc_plot,
)
from src.facenet_model import get_embedding, load_models


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "cache" / "embeddings.pkl"
RESULTS_DIR = ROOT / "results"
RESIZE = 1.0


class EmbeddingCache:
    """Cache embeddings to avoid repeating FaceNet inference."""

    def __init__(self, model, mtcnn):
        self.model = model
        self.mtcnn = mtcnn
        self.cache = {}
        self.dirty = False

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        if CACHE_PATH.exists():
            try:
                with CACHE_PATH.open("rb") as file:
                    saved = pickle.load(file)

                if isinstance(saved, dict):
                    self.cache = saved
                    print(f"Loaded {len(self.cache)} cached embeddings.")

            except (OSError, EOFError, pickle.UnpicklingError, AttributeError):
                print("Cache could not be read; starting a new cache.")

    @staticmethod
    def prepare_image(image_array):
        array = np.asarray(image_array)

        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError(
                f"Expected an RGB image, got shape {array.shape}"
            )

        if np.issubdtype(array.dtype, np.floating):
            array = np.nan_to_num(
                array,
                nan=0.0,
                posinf=255.0,
                neginf=0.0,
            )

            if array.size and array.max() <= 1.0:
                array = array * 255.0

        array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def get(self, image_array):
        image_array = self.prepare_image(image_array)

        # Include model and preprocessing configuration in the cache key.
        image_hash = hashlib.sha256(image_array.tobytes()).hexdigest()
        key = f"facenet-vggface2-mtcnn160-margin20-v1:{image_hash}"

        if key in self.cache:
            return self.cache[key]

        pil_image = Image.fromarray(image_array)

        with torch.inference_mode():
            embedding = get_embedding(
                pil_image,
                self.mtcnn,
                self.model,
            )

        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)

        if embedding.size != 512:
            raise ValueError(
                f"Expected a 512-dimensional embedding, got {embedding.size}"
            )

        if not np.all(np.isfinite(embedding)):
            raise ValueError("Embedding contains non-finite values.")

        self.cache[key] = embedding
        self.dirty = True

        if len(self.cache) % 50 == 0:
            self.save()

        return embedding

    def save(self):
        if not self.dirty:
            return

        temporary_path = CACHE_PATH.with_suffix(".tmp")

        with temporary_path.open("wb") as file:
            pickle.dump(self.cache, file)

        temporary_path.replace(CACHE_PATH)
        self.dirty = False

        print(f"Saved embedding cache ({len(self.cache)} images).")


def cosine_similarity(embedding_a, embedding_b):
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
    )



def score_pairs(pairs, labels, pair_indices, cache, subset_name):
    valid_indices = []
    valid_labels = []
    scores = []
    skipped = []

    for position, (image_a, image_b) in enumerate(pairs):
        original_index = int(pair_indices[position])
        label = int(labels[position])

        embedding_a = None
        embedding_b = None

        # Test image 1 and image 2 separately.
        try:
            embedding_a = cache.get(image_a)
        except (ValueError, RuntimeError, OSError) as exc:
            skipped.append({
                "subset": subset_name,
                "pair_index": original_index,
                "label": label,
                "failed_image": "image_1",
                "reason": str(exc),
            })

        try:
            embedding_b = cache.get(image_b)
        except (ValueError, RuntimeError, OSError) as exc:
            skipped.append({
                "subset": subset_name,
                "pair_index": original_index,
                "label": label,
                "failed_image": "image_2",
                "reason": str(exc),
            })

        # Skip the pair if either image failed.
        if embedding_a is None or embedding_b is None:
            continue

        try:
            score = cosine_similarity(embedding_a, embedding_b)

            if not np.isfinite(score):
                raise ValueError("Non-finite similarity score")

        except (ValueError, RuntimeError, OSError) as exc:
            skipped.append({
                "subset": subset_name,
                "pair_index": original_index,
                "label": label,
                "failed_image": "similarity_calculation",
                "reason": str(exc),
            })
            continue

        valid_indices.append(original_index)
        valid_labels.append(label)
        scores.append(score)

        if (position + 1) % 25 == 0 or position + 1 == len(pairs):
            print(
                f"{subset_name}: processed {position + 1}/{len(pairs)} "
                f"pairs; valid={len(scores)}, "
                f"skipped={len(set(s['pair_index'] for s in skipped))}"
            )

    return (
        np.asarray(valid_indices, dtype=np.int64),
        np.asarray(valid_labels, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        skipped,
    )


def save_json(data, path):
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, allow_nan=False)


def save_predictions(pair_indices, labels, scores, threshold, output_path):
    predictions = (scores >= threshold).astype(np.int64)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "original_pair_index",
            "actual_label",
            "similarity",
            "prediction",
        ])

        for index, label, score, prediction in zip(
            pair_indices, labels, scores, predictions
        ):
            writer.writerow([
                int(index),
                int(label),
                f"{score:.8f}",
                int(prediction),
            ])



def save_skipped_pairs(skipped, output_path):
    # Always rewrite this file to prevent stale records from earlier runs.
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "subset",
            "pair_index",
            "label",
            "failed_image",
            "reason",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FaceNet on LFW image pairs."
    )
    parser.add_argument(
        "--max-dev-pairs",
        type=int,
        default=None,
        help="Limit development pairs for a quick smoke test.",
    )
    parser.add_argument(
        "--max-test-pairs",
        type=int,
        default=None,
        help="Limit test pairs for a quick smoke test.",
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

    for name in ("max_dev_pairs", "max_test_pairs"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")

    torch.set_num_threads(args.threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading FaceNet models...")
    mtcnn, model = load_models()
    cache = EmbeddingCache(model, mtcnn)

    dev_pairs, dev_labels, dev_indices = load_lfw_pairs(
        subset="train",
        max_pairs=args.max_dev_pairs,
        random_state=42,
        resize=RESIZE,
        return_indices=True,
    )

    dev_indices, dev_labels, dev_scores, dev_skipped = score_pairs(
        dev_pairs,
        dev_labels,
        dev_indices,
        cache,
        "development",
    )

    if len(dev_labels) == 0 or len(np.unique(dev_labels)) < 2:
        raise RuntimeError(
            "Development evaluation needs valid pairs from both classes. "
            "Try a larger --max-dev-pairs value."
        )

    threshold = choose_threshold(dev_labels, dev_scores)
    print(f"Selected threshold from development data: {threshold:.6f}")

    test_pairs, test_labels, test_indices = load_lfw_pairs(
        subset="test",
        max_pairs=args.max_test_pairs,
        random_state=43,
        resize=RESIZE,
        return_indices=True,
    )

    total_test_pairs = len(test_pairs)

    test_indices, test_labels, test_scores, test_skipped = score_pairs(
        test_pairs,
        test_labels,
        test_indices,
        cache,
        "test",
    )

    cache.save()

    if len(test_labels) == 0 or len(np.unique(test_labels)) < 2:
        raise RuntimeError(
            "Test evaluation needs valid pairs from both classes. "
            "Try a larger --max-test-pairs value."
        )


    metrics = compute_metrics(test_labels, test_scores, threshold)

    metrics.update({
        "threshold_source": "development subset",
        "model": "InceptionResnetV1 pretrained on VGGFace2",
        "dataset": "Labeled Faces in the Wild (LFW)",
        "resize": RESIZE,
        "development_pairs_total": len(dev_pairs),
        "development_pairs_used": len(dev_labels),
        "development_pairs_skipped": len(
            {item["pair_index"] for item in dev_skipped}
        ),
        "development_coverage": (
            len(dev_labels) / len(dev_pairs) if len(dev_pairs) else 0.0
        ),
        "test_pairs_total": total_test_pairs,
        "test_pairs_used": len(test_labels),
        "test_pairs_skipped": len(
            {item["pair_index"] for item in test_skipped}
        ),
        "test_coverage": (
            len(test_labels) / total_test_pairs if total_test_pairs else 0.0
        ),
        "python_torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "facenet_pytorch_version": version("facenet-pytorch"),
    })

    save_json(metrics, RESULTS_DIR / "metrics.json")

    with (RESULTS_DIR / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    save_roc_plot(
        test_labels,
        test_scores,
        RESULTS_DIR / "roc_curve.png",
    )

    save_confusion_plot(
        test_labels,
        test_scores,
        threshold,
        RESULTS_DIR / "confusion_matrix.png",
    )

    save_predictions(
        test_indices,
        test_labels,
        test_scores,
        threshold,
        RESULTS_DIR / "test_predictions.csv",
    )

    all_skipped = dev_skipped + test_skipped

    save_skipped_pairs(
        all_skipped,
        RESULTS_DIR / "skipped_pairs.csv",
    )

    print("\nEvaluation complete.")
    print(json.dumps(metrics, indent=2))
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
