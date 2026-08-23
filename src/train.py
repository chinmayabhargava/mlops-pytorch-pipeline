"""Training loop for the image classifier.

Reads hyperparameters from configs/training_config.yaml, logs per-epoch
metrics as JSON lines on stdout, saves the best checkpoint, and supports
early stopping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import get_dataloaders, get_dataset_info, resolve_image_size
from src.model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError(f"Config file is empty: {config_path}")
    return config


def resolve_config_path(cli_path: str | None = None) -> Path:
    if cli_path:
        path = Path(cli_path)
        if not path.is_absolute():
            if (Path.cwd() / path).exists():
                path = Path.cwd() / path
            elif (ROOT / path).exists():
                path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {cli_path}")
        return path.resolve()
    for candidate in (Path("/app/configs/training_config.yaml"), ROOT / "configs" / "training_config.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find configs/training_config.yaml. Pass --config PATH."
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_under_root(path: str | Path) -> Path:
    """Resolve relative paths against the project root, not the current working directory."""
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, correct / total


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
    meta: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            **meta,
        },
        path,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an image classifier.")
    parser.add_argument("--config", default=None, help="Path to training_config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Limit train/val batches per epoch (useful for a smoke run)",
    )
    args = parser.parse_args(argv)

    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))

    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    output_cfg = config.get("output", {})

    dataset_name = data_cfg.get("dataset_name", "cifar10")
    architecture = model_cfg.get("architecture", "simple_cnn")
    info = get_dataset_info(dataset_name)
    num_classes = int(model_cfg.get("num_classes", len(info["classes"])))
    in_channels = int(info["in_channels"])
    image_size = resolve_image_size(
        dataset_name,
        architecture,
        data_cfg.get("image_size"),
    )
    classes = list(info["classes"])

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"event": "start", "device": str(device), "config": str(config_path)}), flush=True)

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", False)),
        in_channels=in_channels,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=str(resolve_under_root(data_cfg.get("data_dir", "./data"))),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 0)),
        dataset_name=dataset_name,
        architecture=architecture,
        image_size=image_size,
        pin_memory=device.type == "cuda",
        download=bool(data_cfg.get("download", True)),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = int(train_cfg.get("early_stopping_patience", 5))
    checkpoint_dir = resolve_under_root(output_cfg.get("checkpoint_dir", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / output_cfg.get("model_name", "model.pt")
    max_batches = args.max_batches if args.max_batches is not None else train_cfg.get("max_batches")
    if max_batches is not None:
        max_batches = int(max_batches)

    checkpoint_meta = {
        "architecture": architecture,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "dataset_name": dataset_name,
        "image_size": image_size,
        "classes": classes,
    }

    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 10))
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_batches=max_batches
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, max_batches=max_batches
        )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "train_loss": round(train_loss, 4),
                    "train_accuracy": round(train_acc, 4),
                    "val_loss": round(val_loss, 4),
                    "val_accuracy": round(val_acc, 4),
                }
            ),
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_acc,
                checkpoint_meta,
            )
            print(json.dumps({"event": "checkpoint_saved", "path": str(save_path)}), flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(json.dumps({"event": "early_stopping", "epoch": epoch + 1}), flush=True)
                break

    print(
        json.dumps({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
