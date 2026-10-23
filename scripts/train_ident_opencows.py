"""在 OpenCows2020 的官方 known/unknown 划分上重训个体识别器，并测开集拒识。

与仓库自带的 10 类分类器不同，这里用数据集自带的 75-25.json 等划分：一部分牛只作为已知类
训练，另一部分牛只只在测试时出现，用来衡量"能不能把没见过的牛判成 unknown"。

用法：
    python scripts/train_ident_opencows.py --zip OpenCows2020.zip --epochs 20 --device cuda
"""

import argparse
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from scripts.opencows import extract_identification, ident_split

IMAGE_SIZE = 224
FALSE_POSITIVE_BUDGET = 0.05


class FaceDataset(Dataset):
    def __init__(self, items: list[tuple[pathlib.Path, int]], transform) -> None:
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), label


def collect_items(images_dir: pathlib.Path, split: str, classes: list[str], label_of: dict[str, int]) -> list:
    items = []
    for class_dir in classes:
        for path in sorted((images_dir / split / class_dir).glob("*.jpg")):
            label = label_of.get(class_dir)
            if label is not None:
                items.append((path, label))
    return items


def build_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluate = transforms.Compose(
        [transforms.Resize(IMAGE_SIZE + 32), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(), normalize]
    )
    return train, evaluate


def max_probabilities(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[float], list[int]]:
    model.eval()
    probs: list[float] = []
    hits: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            softmax = torch.softmax(logits, dim=1)
            confidence, predicted = softmax.max(dim=1)
            probs.extend(confidence.cpu().tolist())
            hits.extend((predicted.cpu() == labels).int().tolist())
    return probs, hits


def auroc(known_scores: list[float], unknown_scores: list[float]) -> float:
    """已知类得分 vs 未见类得分的排序 AUC，用来衡量拒识可分性。"""
    pairs = [(score, 1) for score in known_scores] + [(score, 0) for score in unknown_scores]
    pairs.sort(key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        rank_sum += average_rank * sum(1 for _, label in pairs[index : end + 1] if label == 1)
        index = end + 1
    positives = len(known_scores)
    negatives = len(unknown_scores)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def tpr_at_fpr(known_scores: list[float], unknown_scores: list[float], budget: float) -> tuple[float, float]:
    """给定误拒预算（把已知牛误判成 unknown 的比例），返回 (接受率, 阈值)。"""
    ordered = sorted(known_scores)
    cut = max(int(len(ordered) * budget) - 1, 0)
    threshold = ordered[cut]
    accepted = sum(1 for score in unknown_scores if score < threshold) / len(unknown_scores)
    return accepted, threshold


def train(model: nn.Module, loader: DataLoader, args, device: torch.device) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.perf_counter()
        total = 0.0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * labels.size(0)
            correct += int((logits.argmax(dim=1) == labels).sum())
        scheduler.step()
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={total / len(loader.dataset):.4f} "
            f"acc={correct / len(loader.dataset):.4f} {time.perf_counter() - started:.1f}s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("opencows_work"))
    parser.add_argument("--split", default="75-25", help="splits 目录下的划分名，如 75-25")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/ident_opencows.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 张训练图，用于冒烟")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    images_dir = extract_identification(args.zip, args.workdir)
    known, unknown = ident_split(args.zip, args.split, args.fold)
    print(f"划分 {args.split} fold {args.fold}: 已知 {len(known)} 头，未见 {len(unknown)} 头")

    label_of = {name: index for index, name in enumerate(known)}
    train_items = collect_items(images_dir, "train", known, label_of)
    if args.limit:
        train_items = train_items[: args.limit]
    known_test_items = collect_items(images_dir, "test", known, label_of)
    unknown_test_paths = [
        path for class_dir in unknown for path in sorted((images_dir / "test" / class_dir).glob("*.jpg"))
    ]
    print(f"训练 {len(train_items)} 张 / 已知测试 {len(known_test_items)} 张 / 未见测试 {len(unknown_test_paths)} 张")

    train_transform, eval_transform = build_transforms()
    train_loader = DataLoader(
        FaceDataset(train_items, train_transform), batch_size=args.batch, shuffle=True, num_workers=0
    )
    known_loader = DataLoader(FaceDataset(known_test_items, eval_transform), batch_size=args.batch, num_workers=0)
    unknown_loader = DataLoader(
        FaceDataset([(path, -1) for path in unknown_test_paths], eval_transform), batch_size=args.batch, num_workers=0
    )

    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(known))
    model.to(device)

    train(model, train_loader, args, device)

    known_probs, known_hits = max_probabilities(model, known_loader, device)
    unknown_probs, _ = max_probabilities(model, unknown_loader, device)
    accepted, threshold = tpr_at_fpr(known_probs, unknown_probs, FALSE_POSITIVE_BUDGET)
    top1 = sum(known_hits) / len(known_hits)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "classes": known, "split": args.split, "fold": args.fold},
        args.out,
    )
    metrics = {
        "dataset": f"OpenCows2020 identification ({args.split} fold {args.fold})",
        "known_classes": len(known),
        "unknown_classes": len(unknown),
        "train_images": len(train_items),
        "known_test_images": len(known_test_items),
        "unknown_test_images": len(unknown_test_paths),
        "closed_set_top1": round(top1, 4),
        "open_set_auroc": round(auroc(known_probs, unknown_probs), 4),
        "unknown_rejection_at_5pct_false_reject": round(accepted, 4),
        "threshold": round(threshold, 4),
        "epochs": args.epochs,
        "device": str(device),
        "checkpoint": str(args.out),
    }
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
