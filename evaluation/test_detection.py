
from evaluation.dataset import load_lfw_pairs
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
import torch

pairs, labels, indices = load_lfw_pairs(
    subset="train",
    resize=1.0,
    return_indices=True,
)

position = int(np.where(indices == 39)[0][0])
image_array = pairs[position][0]
image_array = np.clip(
    image_array * 255, 0, 255
).astype(np.uint8)
image = Image.fromarray(image_array)

detector = MTCNN(
    image_size=160,
    margin=0,
    keep_all=True,
    device=torch.device("cpu"),
)

for factor in [1, 2, 3]:
    resized = image.resize(
        (image.width * factor, image.height * factor),
        Image.Resampling.LANCZOS,
    )

    boxes, probabilities = detector.detect(resized)

    print(f"\nResize factor: {factor}x")
    print("Detected boxes:", boxes)
    print("Detection probabilities:", probabilities)
