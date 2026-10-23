"""在 OpenCows2020 的官方 10 折划分上重训检测器，得到无泄漏的 mAP。

仓库自带的权重是在 Roboflow 随机增强划分上训练的，test 帧 98% 与 train 同源。这里改用数据集自带
的 10-fold-CV.json：按牛只身份分组，测试帧里的牛没在训练集出现过。

用法：
    python scripts/train_detect_opencows.py --zip OpenCows2020.zip --epochs 30 --device cuda
"""

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from app.config import PRETRAINED_MODEL_PATH
from scripts.opencows import prepare_detection

DEFAULT_WEIGHTS = PRETRAINED_MODEL_PATH["n"]


def pick_weights(explicit: pathlib.Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    return str(DEFAULT_WEIGHTS) if DEFAULT_WEIGHTS.exists() else "yolov8n.pt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=pathlib.Path, required=True)
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("opencows_work"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--weights", type=pathlib.Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 张训练图，用于冒烟")
    parser.add_argument("--project", type=pathlib.Path, default=pathlib.Path("runs/opencows"))
    parser.add_argument("--name", default="detect_fold0")
    args = parser.parse_args()

    data_yaml = prepare_detection(args.zip, args.workdir, args.fold, args.limit)
    print(f"训练数据: {data_yaml}")

    model = YOLO(pick_weights(args.weights))
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        plots=False,
        val=True,
    )

    metrics = YOLO(str(args.project / args.name / "weights" / "best.pt")).val(
        data=str(data_yaml), split="val", imgsz=args.imgsz, batch=8, device=args.device, verbose=False, plots=False
    )
    box = metrics.box
    print(
        json.dumps(
            {
                "dataset": f"OpenCows2020 detection (official fold {args.fold})",
                "images": len(box.image_metrics),
                "mAP50": round(float(box.map50), 4),
                "mAP50-95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
