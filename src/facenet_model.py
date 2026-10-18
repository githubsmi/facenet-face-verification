
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torch.nn.functional import cosine_similarity


DEVICE = torch.device("cpu")


def load_models():
    """Load the pretrained face detector and FaceNet model."""

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        device=DEVICE,
    )

    model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(DEVICE)

    return detector, model


def get_embedding(image: Image.Image, detector, model):
    """Return a 512-dimensional embedding for one detected face."""

    image = image.convert("RGB")
    face = detector(image)

    if face is None:
        raise ValueError(
            "No face detected. Please upload a clearer image "
            "with one visible face."
        )

    face = face.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        embedding = model(face)

    return embedding


def compare_faces(image1, image2, detector, model):
    """Compare two face embeddings using cosine similarity."""

    embedding1 = get_embedding(image1, detector, model)
    embedding2 = get_embedding(image2, detector, model)

    score = cosine_similarity(
        embedding1,
        embedding2,
        dim=1,
    ).item()

    return score
