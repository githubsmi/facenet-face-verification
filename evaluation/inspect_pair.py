
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import load_lfw_pairs


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "failed_examples"


def main():
    parser = argparse.ArgumentParser(
        description="Save an LFW image pair for visual inspection."
    )
    parser.add_argument(
        "--subset",
        choices=["train", "test"],
        required=True,
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
    )
    args = parser.parse_args()

    pairs, labels, indices = load_lfw_pairs(
        subset=args.subset,
        resize=1.0,
        return_indices=True,
    )

    matches = np.where(indices == args.index)[0]

    if len(matches) == 0:
        raise ValueError(
            f"Pair index {args.index} was not found in {args.subset}."
        )

    position = int(matches[0])
    pair = pairs[position]
    label = int(labels[position])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for side, image_array in enumerate(pair, start=1):
        image_array = np.asarray(image_array)

        # Convert normalized float pixels (0–1) to image pixels (0–255).
        if (
            np.issubdtype(image_array.dtype, np.floating)
            and image_array.max() <= 1.0
        ):
            image_array = image_array * 255.0

        image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)

        path = OUTPUT_DIR / (
            f"{args.subset}_pair_{args.index}_image_{side}.png"
        )
        image.save(path)
        print(f"Saved: {path}")

    label_text = "same person" if label == 1 else "different people"
    print(f"Subset: {args.subset}")
    print(f"Original pair index: {args.index}")
    print(f"Label: {label} ({label_text})")


if __name__ == "__main__":
    main()
