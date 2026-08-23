"""Data loading for CIFAR-10 and Fashion-MNIST."""
from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASET_INFO: dict[str, dict[str, Any]] = {
    "cifar10": {
        "builder": datasets.CIFAR10,
        "classes": [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ],
        "in_channels": 3,
        "image_size": 32,
        "mean": [0.4914, 0.4822, 0.4465],
        "std": [0.2470, 0.2435, 0.2616],
    },
    "fashion_mnist": {
        "builder": datasets.FashionMNIST,
        "classes": [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ],
        "in_channels": 1,
        "image_size": 28,
        "mean": [0.2860],
        "std": [0.3530],
    },
}

_ALIASES = {
    "cifar-10": "cifar10",
    "cifar_10": "cifar10",
    "fashionmnist": "fashion_mnist",
    "fashion-mnist": "fashion_mnist",
}


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """Return metadata for a supported dataset (classes, channels, size, stats)."""
    name = _ALIASES.get(dataset_name.lower(), dataset_name.lower())
    if name not in DATASET_INFO:
        supported = ", ".join(sorted(DATASET_INFO))
        raise ValueError(f"Unknown dataset: {dataset_name!r}. Choose one of: {supported}.")
    return DATASET_INFO[name]


def resolve_image_size(
    dataset_name: str,
    architecture: str = "simple_cnn",
    image_size: int | None = None,
) -> int:
    """Native size for simple_cnn; 224 for ResNet-18 unless overridden."""
    if image_size is not None:
        return int(image_size)
    if architecture.lower() == "resnet18":
        return 224
    return int(get_dataset_info(dataset_name)["image_size"])


def get_transforms(
    train: bool = True,
    dataset_name: str = "cifar10",
    architecture: str = "simple_cnn",
    image_size: int | None = None,
) -> transforms.Compose:
    info = get_dataset_info(dataset_name)
    size = resolve_image_size(dataset_name, architecture, image_size)
    ops: list = []
    if size != info["image_size"]:
        ops.append(transforms.Resize(size))
    if train:
        ops.append(transforms.RandomHorizontalFlip())
        ops.append(transforms.RandomCrop(size, padding=4))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=info["mean"], std=info["std"]),
        ]
    )
    return transforms.Compose(ops)


def get_eval_transform(
    dataset_name: str,
    architecture: str = "simple_cnn",
    image_size: int | None = None,
) -> transforms.Compose:
    """Inference-time transform used by src/serve.py."""
    return get_transforms(
        train=False,
        dataset_name=dataset_name,
        architecture=architecture,
        image_size=image_size,
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
    dataset_name: str = "cifar10",
    architecture: str = "simple_cnn",
    image_size: int | None = None,
    pin_memory: bool = False,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    info = get_dataset_info(dataset_name)
    builder = info["builder"]
    train_dataset = builder(
        root=data_dir,
        train=True,
        download=download,
        transform=get_transforms(
            train=True,
            dataset_name=dataset_name,
            architecture=architecture,
            image_size=image_size,
        ),
    )
    val_dataset = builder(
        root=data_dir,
        train=False,
        download=download,
        transform=get_eval_transform(
            dataset_name=dataset_name,
            architecture=architecture,
            image_size=image_size,
        ),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
