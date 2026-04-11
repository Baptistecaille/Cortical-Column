"""
Entrainement MNIST avec la colonne corticale v3.

Cette version evite torchvision et lit directement les fichiers IDX deja
present dans le projet.

Usage:
    .venv/bin/python train_mnist_v3.py --train-limit 1000 --test-limit 200 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cortical_column_v3 import ColumnV3, IDX_V3


DEFAULT_MNIST_ROOT = Path(
    "/Users/baptistecaillerie/Documents/Cortical Column/Corical column code/data/MNIST/raw"
)


def _read_idx_images(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    magic, count, rows, cols = struct.unpack(">IIII", raw[:16])
    if magic != 2051:
        raise ValueError(f"Magic image IDX invalide pour {path}: {magic}")
    pixels = torch.tensor(bytearray(raw[16:]), dtype=torch.uint8)
    return pixels.view(count, rows, cols).float() / 255.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    raw = path.read_bytes()
    magic, count = struct.unpack(">II", raw[:8])
    if magic != 2049:
        raise ValueError(f"Magic label IDX invalide pour {path}: {magic}")
    labels = torch.tensor(bytearray(raw[8:]), dtype=torch.uint8)
    return labels.view(count).long()


class LocalMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, train: bool = True, limit: int | None = None, seed: int = 0):
        split = "train" if train else "t10k"
        self.images = _read_idx_images(root / f"{split}-images-idx3-ubyte")
        self.labels = _read_idx_labels(root / f"{split}-labels-idx1-ubyte")
        if limit is not None and limit < len(self.labels):
            rng = random.Random(seed)
            indices = list(range(len(self.labels)))
            rng.shuffle(indices)
            selected = torch.tensor(indices[:limit], dtype=torch.long)
            self.images = self.images[selected]
            self.labels = self.labels[selected]

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.labels[idx]


@dataclass
class EvalStats:
    loss: float
    accuracy: float


class MNISTV3Classifier(nn.Module):
    """
    Harness de classification autour de ColumnV3.

    L'image est convertie en drives sensoriels/top-down de dimension n_mc, puis
    on effectue quelques pas de scan avec petites vitesses pour faire vivre L6.
    """

    def __init__(
        self,
        n_mc: int = 32,
        k: int = 6,
        dt: float = 0.1,
        scan_steps: int = 3,
        hidden_dim: int = 64,
        local_column_learning: bool = False,
    ):
        super().__init__()
        self.n_mc = n_mc
        self.scan_steps = scan_steps
        self.local_column_learning = local_column_learning
        self.column = ColumnV3(n_mc=n_mc, k=k, dt=dt)
        self.classifier = nn.Sequential(
            nn.Linear(4 * n_mc, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 10),
        )

    def _scan_velocities(self, device: torch.device) -> List[torch.Tensor]:
        base = [
            torch.tensor([0.20, 0.00], device=device),
            torch.tensor([0.00, 0.20], device=device),
            torch.tensor([0.15, 0.15], device=device),
            torch.tensor([-0.10, 0.10], device=device),
        ]
        return base[: self.scan_steps]

    def _image_to_drives(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = image.reshape(1, 1, -1)
        transposed_flat = image.transpose(0, 1).contiguous().reshape(1, 1, -1)
        sensory_drive = F.interpolate(flat, size=self.n_mc, mode="linear", align_corners=False).view(-1) * 80.0
        topdown_drive = F.interpolate(transposed_flat, size=self.n_mc, mode="linear", align_corners=False).view(-1) * 40.0
        return sensory_drive, topdown_drive

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            image = image.unsqueeze(0)
        batch = image.shape[0]
        logits = []
        device = image.device

        for b in range(batch):
            sensory_drive, topdown_drive = self._image_to_drives(image[b])
            self.column.reset_state()
            with torch.no_grad():
                out = None
                for velocity in self._scan_velocities(device):
                    out = self.column(
                        sensory_drive,
                        mu=topdown_drive,
                        velocity=velocity,
                        learn=self.local_column_learning and self.training,
                        prospective_steps=3,
                    )

            assert out is not None
            feature = torch.cat(
                [
                    out["y_ff"],
                    out["y_fb"],
                    out["r_6"],
                    out["eps_pos"] + out["eps_neg"],
                ],
                dim=0,
            )
            logits.append(self.classifier(feature.detach()))

        return torch.stack(logits, dim=0)


def iterate_minibatches(dataset: Sequence[tuple[torch.Tensor, torch.Tensor]], batch_size: int) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for start in range(0, len(dataset), batch_size):
        images = []
        labels = []
        for idx in range(start, min(start + batch_size, len(dataset))):
            image, label = dataset[idx]
            images.append(image)
            labels.append(label)
        yield torch.stack(images), torch.tensor(labels, dtype=torch.long)


def evaluate(model: MNISTV3Classifier, dataset: Sequence[tuple[torch.Tensor, torch.Tensor]], batch_size: int, device: torch.device) -> EvalStats:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in iterate_minibatches(dataset, batch_size=batch_size):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.shape[0]
    return EvalStats(loss=total_loss / total, accuracy=total_correct / total)


def train(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cpu")
    train_dataset = LocalMNISTDataset(args.mnist_root, train=True, limit=args.train_limit, seed=args.seed)
    test_dataset = LocalMNISTDataset(args.mnist_root, train=False, limit=args.test_limit, seed=args.seed)

    model = MNISTV3Classifier(
        n_mc=args.n_mc,
        k=args.k,
        dt=args.dt,
        scan_steps=args.scan_steps,
        hidden_dim=args.hidden_dim,
        local_column_learning=args.local_column_learning,
    ).to(device)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(optim_params, lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best = {"test_accuracy": 0.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        shuffled = [(train_dataset.images[i], train_dataset.labels[i]) for i in indices]
        for images, labels in iterate_minibatches(shuffled, batch_size=args.batch_size):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * labels.shape[0]
            total_correct += (logits.argmax(dim=-1) == labels).sum().item()
            total_seen += labels.shape[0]

        train_loss = total_loss / total_seen
        train_acc = total_correct / total_seen
        test_stats = evaluate(model, test_dataset, batch_size=args.batch_size, device=device)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_stats.loss,
            "test_accuracy": test_stats.accuracy,
        }
        history.append(epoch_stats)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"test_loss={test_stats.loss:.4f} test_acc={test_stats.accuracy:.3f}"
        )

        if test_stats.accuracy >= best["test_accuracy"]:
            best = epoch_stats
            torch.save(model.state_dict(), args.output_dir / "mnist_v3_best.pt")

    results = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "best": best,
        "history": history,
    }
    with (args.output_dir / "mnist_v3_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test d'apprentissage MNIST avec ColumnV3")
    parser.add_argument("--mnist-root", type=Path, default=DEFAULT_MNIST_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("./artifacts"))
    parser.add_argument("--train-limit", type=int, default=1000)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-mc", type=int, default=32)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--scan-steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-column-learning", action="store_true")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = train(args)
    print("best_test_acc:", round(results["best"]["test_accuracy"], 4))
    print("artifacts:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
