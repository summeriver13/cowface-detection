"""在 OpenCows2020 上评测检测器的跨数据集表现。

OpenCows2020 是布里斯托大学发布的荷斯坦奶牛数据集（2.1 GiB，Non-Commercial Government
Licence），检测部分 7,043 张帧配 darknet/YOLO 格式标签，003707 之后的帧是合成图。
这个脚本只取真实帧，把它们抽到一个工作目录里，然后用项目自带的权重跑一次 `model.val`。

用法：
    python scripts/evaluate_opencows.py --zip ~/Downloads/OpenCows2020.zip --workdir /tmp/opencows
下载地址：https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip
"""

import argparse
import json
import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from app.config import YOLO_MODEL_PATH

REAL_FRAME_LIMIT = 3707
IMAGES_MARKER = "/detection_and_localisation/images/"
LABELS_MARKER = "/labels-darknet/"


def is_real_frame(name: str) -> bool:
    stem = pathlib.Path(name).stem
    return stem.isdigit() and int(stem) <= REAL_FRAME_LIMIT


def prepare(zip_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
    images_dir = workdir / "images"
    labels_dir = workdir / "labels"
    data_yaml = workdir / "data.yaml"
    if data_yaml.exists() and any(images_dir.glob("*.jpg")):
        print(f"复用已有工作目录: {workdir}")
        return data_yaml

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir()]
        images = {pathlib.Path(i.filename).stem: i for i in entries if IMAGES_MARKER in i.filename}
        labels = {pathlib.Path(i.filename).stem: i for i in entries if LABELS_MARKER in i.filename}

        kept = 0
        for stem, info in sorted(images.items()):
            if not is_real_frame(info.filename) or stem not in labels:
                continue
            (images_dir / f"{stem}.jpg").write_bytes(archive.read(info))
            (labels_dir / f"{stem}.txt").write_bytes(archive.read(labels[stem]))
            kept += 1

    print(f"抽取真实帧 {kept} 张 -> {workdir}")
    data_yaml.write_text(
        f"path: {workdir.as_posix()}\nval: images\ntrain: images\nnc: 1\nnames: [cow_face]\n",
        encoding="utf-8",
    )
    return data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=pathlib.Path, required=True, help="OpenCows2020.zip")
    parser.add_argument("--workdir", type=pathlib.Path, default=pathlib.Path("opencows_eval"))
    args = parser.parse_args()

    data_yaml = prepare(args.zip, args.workdir)
    model = YOLO(str(YOLO_MODEL_PATH))
    metrics = model.val(data=str(data_yaml), split="val", imgsz=640, batch=8, device="cpu", verbose=False, plots=False)
    box = metrics.box
    print(
        json.dumps(
            {
                "dataset": "OpenCows2020 (real frames)",
                "images": len(box.image_metrics),
                "mAP50": round(float(box.map50), 4),
                "mAP50-95": round(float(box.map), 4),
                "precision": round(float(box.mp), 4),
                "recall": round(float(box.mr), 4),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
