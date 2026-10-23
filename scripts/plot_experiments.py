"""重画 docs/ 下的两张图：开集拒识曲线与域偏移对比。

开集曲线直接从识别 checkpoint 计算；域偏移图里的四个 mAP50 是 docs/experiments.md 记录的结果，
由 scripts/evaluate*.py 与 scripts/train_*_opencows.py 产出，这里只负责画图。

需要 matplotlib：
    uv pip install matplotlib

用法：
    python scripts/plot_experiments.py --zip OpenCows2020.zip --workdir opencows_work
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0

from scripts.opencows import extract_identification, ident_split
from scripts.train_ident_opencows import FaceDataset, build_transforms, collect_items, max_probabilities

CURVE_STEPS = 200
MARKED_FALSE_REJECT_RATES = (0.05, 0.10)
COLORS = {"75-25": "#2f6fed", "50-50": "#e0662a"}
DOMAIN_GAP = {
    "own split\n(leaky)": 0.941,
    "own split\nre-trained\nOpenCows f0": 0.974,
    "OpenCows w.\n-> own test": 0.355,
    "own w.\n-> OpenCows\n(zero-shot)": 0.055,
}
DOMAIN_COLORS = ["#b9c2d6", "#2f6fed", "#e0a24a", "#d1495b"]


def load_model(checkpoint: pathlib.Path, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def score_split(zip_path, workdir, split: str, device: torch.device, checkpoints: pathlib.Path):
    model = load_model(checkpoints / f"ident_{split}_f0.pt", device)
    images = extract_identification(zip_path, workdir)
    known, unknown = ident_split(zip_path, split, 0)
    label_of = {name: index for index, name in enumerate(known)}
    known_items = collect_items(images, "test", known, label_of)
    unknown_items = [(path, -1) for cow in unknown for path in sorted((images / "test" / cow).glob("*.jpg"))]
    _, evaluate = build_transforms()
    known_probs, _ = max_probabilities(model, DataLoader(FaceDataset(known_items, evaluate), batch_size=64), device)
    unknown_probs, _ = max_probabilities(model, DataLoader(FaceDataset(unknown_items, evaluate), batch_size=64), device)
    return split, len(known), len(unknown), known_probs, unknown_probs


def rejection_curve(known_probs, unknown_probs):
    rates = [step / CURVE_STEPS for step in range(CURVE_STEPS + 1)]
    false_reject = [sum(1 for score in known_probs if score < rate) / len(known_probs) for rate in rates]
    reject = [sum(1 for score in unknown_probs if score < rate) / len(unknown_probs) for rate in rates]
    return false_reject, reject


def plot_open_set(rows, out: pathlib.Path) -> None:
    fig, axes = plt.subplots(figsize=(6.2, 4.6), dpi=160)
    for split, known_cows, unknown_cows, known_probs, unknown_probs in rows:
        xs, ys = rejection_curve(known_probs, unknown_probs)
        axes.plot(xs, ys, color=COLORS[split], lw=2, label=f"{split} ({known_cows} known / {unknown_cows} unknown)")
        for rate in MARKED_FALSE_REJECT_RATES:
            index = min(range(len(xs)), key=lambda i: abs(xs[i] - rate))
            axes.plot([xs[index]], [ys[index]], "o", color=COLORS[split], ms=5)
            axes.annotate(
                f"{ys[index]:.2f}",
                (xs[index], ys[index]),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=8,
                color=COLORS[split],
            )
    axes.plot([0, 1], [0, 1], ls=":", color="#999", lw=1, label="random")
    axes.set_xlabel("False reject rate (known cow rejected)")
    axes.set_ylabel("Rejection rate of unseen cows")
    axes.set_title("Open-set rejection with softmax confidence")
    axes.set_xlim(0, 0.35)
    axes.set_ylim(0, 1.0)
    axes.grid(alpha=0.3)
    axes.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "openset_curve.png")
    plt.close(fig)


def plot_domain_gap(out: pathlib.Path) -> None:
    fig, axes = plt.subplots(figsize=(6.2, 4.0), dpi=160)
    bars = axes.bar(list(DOMAIN_GAP), list(DOMAIN_GAP.values()), color=DOMAIN_COLORS, width=0.62)
    for bar, value in zip(bars, DOMAIN_GAP.values()):
        axes.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)
    axes.set_ylabel("mAP50")
    axes.set_ylim(0, 1.12)
    axes.set_title("Detection mAP50 depends on which farm the data came from")
    axes.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "domain_gap.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("opencows_work"))
    parser.add_argument("--checkpoints", type=pathlib.Path, default=pathlib.Path("outputs"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("docs"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [score_split(args.zip, args.workdir, split, device, args.checkpoints) for split in ("75-25", "50-50")]
    plot_open_set(rows, args.out)
    plot_domain_gap(args.out)
    for name in ("openset_curve.png", "domain_gap.png"):
        print(f"{args.out / name}: {(args.out / name).stat().st_size} 字节")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
