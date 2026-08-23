"""Unit tests for the model, dataset transforms, training step, and serving API."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import (
    get_dataset_info,
    get_eval_transform,
    get_transforms,
    resolve_image_size,
)
from src.model import SimpleCNN, get_model
from src.train import evaluate, train_one_epoch


CIFAR_CLASSES = get_dataset_info("cifar10")["classes"]


def _rgb_image(size: int = 32) -> Image.Image:
    return Image.new("RGB", (size, size), color=(128, 64, 32))


def _gray_image(size: int = 28) -> Image.Image:
    return Image.new("L", (size, size), color=90)


def _write_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_simple_cnn_forward_shape():
    model = SimpleCNN(in_channels=3, num_classes=10)
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_simple_cnn_fashion_mnist_shape():
    model = SimpleCNN(in_channels=1, num_classes=10)
    out = model(torch.randn(2, 1, 28, 28))
    assert out.shape == (2, 10)


def test_resnet18_forward_shape():
    model = get_model("resnet18", num_classes=10, pretrained=False, in_channels=3)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 10)


def test_resnet18_grayscale_replaces_conv1():
    model = get_model("resnet18", num_classes=10, pretrained=False, in_channels=1)
    assert model.conv1.in_channels == 1
    out = model(torch.randn(1, 1, 224, 224))
    assert out.shape == (1, 10)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="Unknown architecture"):
        get_model("not_a_model")


def test_dataset_info_and_aliases():
    cifar = get_dataset_info("cifar10")
    assert cifar["in_channels"] == 3
    assert len(cifar["classes"]) == 10
    fashion = get_dataset_info("fashion-mnist")
    assert fashion["in_channels"] == 1
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_dataset_info("imagenet")


def test_eval_transform_cifar_simple_cnn():
    tensor = get_eval_transform("cifar10", "simple_cnn")(_rgb_image(32))
    assert tensor.shape == (3, 32, 32)


def test_eval_transform_resnet_resizes():
    tensor = get_eval_transform("cifar10", "resnet18")(_rgb_image(32))
    assert tensor.shape == (3, 224, 224)
    assert resolve_image_size("cifar10", "resnet18") == 224


def test_eval_transform_fashion_mnist():
    tensor = get_eval_transform("fashion_mnist", "simple_cnn")(_gray_image(28))
    assert tensor.shape == (1, 28, 28)


def test_train_transforms_include_augmentation():
    train_t = get_transforms(train=True, dataset_name="cifar10")
    eval_t = get_transforms(train=False, dataset_name="cifar10")
    assert len(train_t.transforms) > len(eval_t.transforms)


def test_train_one_epoch_respects_max_batches():
    model = SimpleCNN(in_channels=3, num_classes=10)
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss, acc = train_one_epoch(
        model, loader, optimizer, criterion, torch.device("cpu"), max_batches=1
    )
    assert loss >= 0
    assert 0.0 <= acc <= 1.0


def test_train_one_epoch_and_evaluate():
    model = SimpleCNN(in_channels=3, num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = torch.randint(0, 10, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cpu")

    train_loss, train_acc = train_one_epoch(model, loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, loader, criterion, device)

    assert train_loss >= 0
    assert val_loss >= 0
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0


def _tiny_checkpoint(path: Path, in_channels: int = 3, dataset_name: str = "cifar10") -> Path:
    model = get_model("simple_cnn", num_classes=10, pretrained=False, in_channels=in_channels)
    info = get_dataset_info(dataset_name)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "architecture": "simple_cnn",
            "num_classes": 10,
            "in_channels": in_channels,
            "dataset_name": dataset_name,
            "image_size": info["image_size"],
            "classes": info["classes"],
        },
        path,
    )
    return path


def test_checkpoint_roundtrip(tmp_path: Path):
    ckpt = _tiny_checkpoint(tmp_path / "model.pt")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = get_model(
        architecture=payload["architecture"],
        num_classes=payload["num_classes"],
        pretrained=False,
        in_channels=payload["in_channels"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 10)
    assert payload["classes"] == CIFAR_CLASSES


def test_health_without_model(monkeypatch):
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", "this/does/not/exist.pt")
    from src import serve

    serve._state["model"] = None
    from fastapi.testclient import TestClient

    with TestClient(serve.app, raise_server_exceptions=False) as client:
        response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["model_loaded"] is False


def test_predict_with_loaded_checkpoint(tmp_path: Path, monkeypatch):
    ckpt = _tiny_checkpoint(tmp_path / "model.pt")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(ckpt))
    from src import serve

    serve.load_model(str(ckpt))
    from fastapi.testclient import TestClient

    with TestClient(serve.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        png = _write_png(_rgb_image(32))
        response = client.post("/predict", files={"file": ("cat.png", png, "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in CIFAR_CLASSES
    assert len(body["probabilities"]) == 10
    assert pytest.approx(sum(body["probabilities"].values()), abs=1e-3) == 1.0


def test_predict_rejects_non_image(tmp_path: Path, monkeypatch):
    ckpt = _tiny_checkpoint(tmp_path / "model.pt")
    monkeypatch.setenv("MODEL_CHECKPOINT_PATH", str(ckpt))
    from src import serve

    serve.load_model(str(ckpt))
    from fastapi.testclient import TestClient

    with TestClient(serve.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
    assert response.status_code == 400


def test_training_config_is_valid():
    import yaml

    config_path = Path("configs/training_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert config["model"]["architecture"] in {"simple_cnn", "resnet18"}
    assert config["training"]["early_stopping_patience"] >= 1
    assert config["output"]["model_name"].endswith(".pt")
    json.dumps({"ok": True})  # sanity: stdlib json is what train.py logs with
