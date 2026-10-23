"""OpenCows2020 的公共读取逻辑：解包、官方划分、工作目录准备。

数据集下载：https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip
许可：Non-Commercial Government Licence（仅研究与演示，不可商用）。
"""

import json
import pathlib
import shutil
import zipfile

ZIP_ROOT = "10m32xl88x2b61zlkkgz3fml17/"
IDENT_ROOT = ZIP_ROOT + "identification/"
DETECTION_ROOT = ZIP_ROOT + "detection_and_localisation/"
IDENT_TRAIN = IDENT_ROOT + "images/train/"
IDENT_TEST = IDENT_ROOT + "images/test/"
DETECTION_IMAGES = DETECTION_ROOT + "images/"
DETECTION_LABELS = DETECTION_ROOT + "labels-darknet/"


def read_json(zip_path: pathlib.Path, member: str) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        return json.loads(archive.read(member))


def ident_split(zip_path: pathlib.Path, name: str, fold: int) -> tuple[list[str], list[str]]:
    """返回 (known, unknown) 两个牛只编号列表，编号是目录名，如 "025"。"""
    splits = read_json(zip_path, f"{IDENT_ROOT}splits/{name}.json")
    entry = splits[str(fold)]
    return sorted(entry["known"]), sorted(entry["unknown"])


def detection_split(zip_path: pathlib.Path, fold: int) -> tuple[list[str], list[str]]:
    """官方 10 折划分，按牛只身份分组，返回 (train 文件名, test 文件名)。"""
    splits = read_json(zip_path, f"{DETECTION_ROOT}10-fold-CV.json")
    entry = splits[str(fold)]
    return sorted(entry["train"]), sorted(entry["test"])


def extract_identification(zip_path: pathlib.Path, workdir: pathlib.Path) -> pathlib.Path:
    """把 identification/images 解到 workdir，返回解出来的 images 目录。"""
    images_dir = workdir / "ident_images"
    marker = images_dir / ".done"
    if marker.exists():
        return images_dir

    if images_dir.exists():
        shutil.rmtree(images_dir)
    with zipfile.ZipFile(zip_path) as archive:
        kept = 0
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith(IDENT_ROOT + "images/"):
                continue
            relative = pathlib.Path(info.filename[len(IDENT_ROOT + "images/") :])
            if relative.name.startswith(".") or len(relative.parts) < 2:
                continue
            target = images_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
            kept += 1
    images_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(kept), encoding="utf-8")
    print(f"解出识别图片 {kept} 张 -> {images_dir}")
    return images_dir


def prepare_detection(
    zip_path: pathlib.Path,
    workdir: pathlib.Path,
    fold: int,
    limit: int | None = None,
) -> pathlib.Path:
    """按官方 10 折划分把图片与 darknet 标签铺到 YOLO 目录，返回 data.yaml。"""
    data_yaml = workdir / "det" / "data.yaml"
    if data_yaml.exists():
        print(f"复用已有检测数据集: {data_yaml}")
        return data_yaml

    train_names, test_names = detection_split(zip_path, fold)
    if limit:
        train_names = train_names[:limit]
        test_names = test_names[: max(limit // 8, 16)]

    root = workdir / "det"
    with zipfile.ZipFile(zip_path) as archive:
        images = {
            pathlib.Path(info.filename).name: info
            for info in archive.infolist()
            if info.filename.startswith(DETECTION_IMAGES) and info.filename.endswith(".jpg")
        }
        labels = {
            pathlib.Path(info.filename).stem: info
            for info in archive.infolist()
            if info.filename.startswith(DETECTION_LABELS) and info.filename.endswith(".txt")
        }
        for split, names in (("train", train_names), ("val", test_names)):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            kept = 0
            for name in names:
                stem = pathlib.Path(name).stem
                if name not in images or stem not in labels:
                    continue
                (root / "images" / split / name).write_bytes(archive.read(images[name]))
                (root / "labels" / split / f"{stem}.txt").write_bytes(archive.read(labels[stem]))
                kept += 1
            print(f"{split}: {kept} 张")

    data_yaml.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: [cow_face]\n",
        encoding="utf-8",
    )
    return data_yaml
