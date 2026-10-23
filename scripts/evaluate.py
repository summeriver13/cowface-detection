"""评测随项目发布的检测器与分类器，输出可复现的数字。

数据集由视频帧经 Roboflow 增强导出：test 的 40 段视频帧里有 39 段同时出现在 train，
所以检测指标反映的是"同源增强划分"上的表现，不是跨场景泛化能力。脚本会把重叠率一起打印出来。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from ultralytics import YOLO

from app.config import DATASET_PATH, TRAIN_DIR, YOLO_MODEL_PATH
from app.controllers.recognizer import CowFaceRecognizer

FRAME_PATTERN = re.compile(r"frame_(\d+)_")
SPLITS = ("train", "valid", "test")
DETECTION_CONFIDENCE = 0.25
LATENCY_SAMPLES = 20


def frame_ids(images_dir: Path) -> set[str]:
    return {
        FRAME_PATTERN.match(path.name).group(1)
        for path in images_dir.glob("*")
        if FRAME_PATTERN.match(path.name)
    }


def split_summary(dataset_dir: Path) -> dict[str, dict[str, int]]:
    summary = {}
    for split in SPLITS:
        images = sorted((dataset_dir / split / "images").glob("*"))
        labels = sorted((dataset_dir / split / "labels").glob("*.txt"))
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "frames": len(frame_ids(dataset_dir / split / "images")),
        }
    test_frames = frame_ids(dataset_dir / "test" / "images")
    summary["leakage"] = {
        "test_frames": len(test_frames),
        "test_frames_also_in_train": len(test_frames & frame_ids(dataset_dir / "train" / "images")),
        "test_frames_also_in_valid": len(test_frames & frame_ids(dataset_dir / "valid" / "images")),
    }
    return summary


def write_local_yaml(dataset_dir: Path) -> Path:
    """仓库里那份 dataset.yaml 写的是导出机器的绝对路径，这里按当前目录重写一份。"""
    dump = dataset_dir / "dataset.local.yaml"
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "nc: 1",
        "names: [cow_face]",
    ]
    dump.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dump


def evaluate_detector(dataset_dir: Path) -> dict[str, float]:
    model = YOLO(str(YOLO_MODEL_PATH))
    metrics = model.val(
        data=str(write_local_yaml(dataset_dir)),
        split="test",
        imgsz=640,
        batch=8,
        device="cpu",
        verbose=False,
        plots=False,
    )
    box = metrics.box

    samples = sorted((dataset_dir / "test" / "images").glob("*"))[:LATENCY_SAMPLES]
    started = time.perf_counter()
    for sample in samples:
        model.predict(str(sample), imgsz=640, device="cpu", verbose=False)
    latency_ms = (time.perf_counter() - started) * 1000 / len(samples)

    return {
        "mAP50": round(float(box.map50), 4),
        "mAP50-95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "cpu_latency_ms_per_image": round(latency_ms, 1),
    }


def evaluate_classifier(classification_dir: Path, recognizer: CowFaceRecognizer) -> dict:
    per_class = {}
    correct_total = 0
    total = 0
    for class_dir in sorted(p for p in (classification_dir / "test").iterdir() if p.is_dir()):
        correct = 0
        images = sorted(class_dir.glob("*"))
        for image_path in images:
            image = recognizer.transform(Image.open(image_path).convert("RGB"))
            with torch.inference_mode():
                probabilities = torch.softmax(
                    recognizer.classifier_model(image.unsqueeze(0).to(recognizer.device)), dim=1
                )
            if recognizer.class_names[probabilities.argmax().item()] == class_dir.name:
                correct += 1
        per_class[class_dir.name] = {"correct": correct, "total": len(images)}
        correct_total += correct
        total += len(images)

    return {
        "classes": len(per_class),
        "images": total,
        "top1_accuracy": round(correct_total / total, 4) if total else 0.0,
        "per_class": per_class,
    }


def evaluate_open_set(classification_dir: Path, recognizer: CowFaceRecognizer) -> dict:
    """训练集之外的牛：模型会把它们判成已知个体，还是给出低置信度？"""
    report = {}
    for cow_dir in sorted(
        path for path in classification_dir.iterdir() if path.is_dir() and path.name not in ("train", "test")
    ):
        predictions = []
        for image_path in sorted(cow_dir.glob("*")):
            image = recognizer.transform(Image.open(image_path).convert("RGB"))
            with torch.inference_mode():
                probabilities = torch.softmax(
                    recognizer.classifier_model(image.unsqueeze(0).to(recognizer.device)), dim=1
                )
            confidence, index = probabilities.max(dim=1)
            predictions.append((recognizer.class_names[index.item()], float(confidence.item())))

        counts = {}
        for name, _ in predictions:
            counts[name] = counts.get(name, 0) + 1
        report[cow_dir.name] = {
            "images": len(predictions),
            "mean_max_probability": round(sum(c for _, c in predictions) / len(predictions), 3),
            "predicted_classes": len(counts),
            "top_prediction": list(max(counts.items(), key=lambda item: item[1])),
        }
    return report


def save_preview(dataset_dir: Path, output: Path, recognizer: CowFaceRecognizer) -> None:
    import cv2

    sample = sorted((dataset_dir / "test" / "images").glob("*"))[0]
    image = cv2.imread(str(sample))
    for detection in recognizer.detect_and_recognize(image):
        x1, y1, x2, y2 = detection["box"]
        label = f"{detection['class']} ({detection['class_conf']})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - height - 8), (x1 + width, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"预览图: {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--json", type=Path, help="把结果同时写进 json")
    parser.add_argument("--preview", type=Path, help="保存一张带检测框的样例图")
    args = parser.parse_args()

    classification_dir = args.dataset.parent / "dataset_for_classification"
    recognizer = CowFaceRecognizer(TRAIN_DIR)

    report = {
        "splits": split_summary(args.dataset),
        "detector": evaluate_detector(args.dataset),
        "classifier": evaluate_classifier(classification_dir, recognizer),
        "open_set": evaluate_open_set(classification_dir, recognizer),
    }

    if args.preview:
        save_preview(args.dataset, args.preview, recognizer)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
