
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_lfw_pairs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_HOME = PROJECT_ROOT / "data" / "raw"


def load_lfw_pairs(
    subset="test",
    max_pairs=None,
    random_state=42,
    resize=1.0,
    return_indices=False,
):
    """
    Load LFW image pairs for face verification.

    Labels:
        1 = same person
        0 = different people

    If return_indices=True, also return the original pair indices.
    """

    if subset not in ("train", "test"):
        raise ValueError("subset must be 'train' or 'test'")

    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")

    if resize <= 0:
        raise ValueError("resize must be greater than zero")

    DATA_HOME.mkdir(parents=True, exist_ok=True)

    print(f"Loading LFW {subset} pairs...")

    dataset = fetch_lfw_pairs(
        subset=subset,
        data_home=str(DATA_HOME),
        funneled=True,
        resize=resize,
        color=True,
        download_if_missing=True,
    )

    pairs = np.asarray(dataset.pairs)
    labels = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(len(pairs), dtype=np.int64)

    if len(pairs) != len(labels):
        raise RuntimeError(
            "Number of image pairs and labels does not match."
        )

    if len(pairs) == 0:
        raise RuntimeError("The dataset contains no image pairs.")

    if max_pairs is not None and max_pairs < len(pairs):
        rng = np.random.default_rng(random_state)
        selected = rng.choice(
            len(pairs),
            size=max_pairs,
            replace=False,
        )
        selected.sort()

        pairs = pairs[selected]
        labels = labels[selected]
        indices = indices[selected]

    print(f"Subset: {subset}")
    print(f"Pairs loaded: {len(pairs)}")
    print(f"Same-person pairs: {int(np.sum(labels == 1))}")
    print(f"Different-person pairs: {int(np.sum(labels == 0))}")
    print(f"Image pair array shape: {pairs.shape}")

    if return_indices:
        return pairs, labels, indices

    return pairs, labels
