
import torch
import pytest
from PIL import Image

from src import facenet_model


class FakeDetector:
    def __init__(self, face):
        self.face = face
        self.received_mode = None

    def __call__(self, image):
        self.received_mode = image.mode
        return self.face


class FakeModel:
    def __call__(self, face):
        assert face.shape == (1, 3, 160, 160)
        return torch.ones((1, 512))


def test_get_embedding_returns_512_dimensional_embedding():
    face = torch.rand(3, 160, 160)
    detector = FakeDetector(face)
    model = FakeModel()
    image = Image.new("RGB", (200, 200))

    embedding = facenet_model.get_embedding(
        image, detector, model
    )

    assert embedding.shape == (1, 512)
    assert torch.isfinite(embedding).all()


def test_get_embedding_converts_image_to_rgb():
    face = torch.rand(3, 160, 160)
    detector = FakeDetector(face)
    model = FakeModel()
    image = Image.new("L", (200, 200))

    facenet_model.get_embedding(
        image, detector, model
    )

    assert detector.received_mode == "RGB"


def test_get_embedding_raises_error_when_no_face_detected():
    detector = FakeDetector(None)
    model = FakeModel()
    image = Image.new("RGB", (200, 200))

    with pytest.raises(ValueError, match="No face detected"):
        facenet_model.get_embedding(
            image, detector, model
        )


def test_compare_faces_returns_cosine_similarity(monkeypatch):
    embeddings = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.8, 0.6, 0.0]]),
    ]

    def fake_get_embedding(image, detector, model):
        return embeddings.pop(0)

    monkeypatch.setattr(
        facenet_model,
        "get_embedding",
        fake_get_embedding,
    )

    image1 = Image.new("RGB", (100, 100))
    image2 = Image.new("RGB", (100, 100))

    score = facenet_model.compare_faces(
        image1, image2, object(), object()
    )

    assert score == pytest.approx(0.8, abs=1e-6)


def test_compare_faces_returns_float(monkeypatch):
    embeddings = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0]]),
    ]

    def fake_get_embedding(image, detector, model):
        return embeddings.pop(0)

    monkeypatch.setattr(
        facenet_model,
        "get_embedding",
        fake_get_embedding,
    )

    image = Image.new("RGB", (100, 100))

    score = facenet_model.compare_faces(
        image, image, object(), object()
    )

    assert isinstance(score, float)
    assert score == pytest.approx(1.0)
