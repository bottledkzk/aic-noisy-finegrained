"""Read separate image manifests and prepare the training-only JPEG cache."""

import csv
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
VIEWS = ((1.0, False), (1.0, True), (0.85, False), (0.85, True), (0.70, False))
ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class ImageInputs:
    keys: list[str]
    paths: list[Path]
    classes: list[str]
    labels: np.ndarray | None


def read_inputs(file, root, *, labeled):
    root = Path(root).resolve(strict=True)
    keys, paths, raw_labels = [], [], []
    with Path(file).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "label"} if labeled else {"path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV columns: {', '.join(sorted(required))}")
        for row in reader:
            key = (row["path"] or "").strip()
            relative = Path(key)
            if not key or relative.is_absolute():
                raise ValueError("Image paths must be relative to the data root")
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing image or path outside the data root: {key}")
            keys.append(key)
            paths.append(path)
            if labeled:
                label = (row["label"] or "").strip()
                if not label:
                    raise ValueError(f"Empty training label for {key}")
                raw_labels.append(label)
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("The manifest must contain at least one image and unique paths")
    classes = sorted(set(raw_labels))
    if not labeled:
        return ImageInputs(keys, paths, [], None)
    if len(classes) < 2:
        raise ValueError("Training needs at least two classes")
    mapping = {item: index for index, item in enumerate(classes)}
    labels = np.array([mapping[item] for item in raw_labels], dtype=np.int64)
    return ImageInputs(keys, paths, classes, labels)


def training_transform(size, cache_size):
    return transforms.Compose([
        transforms.Resize(max(cache_size, size + 16), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def grouping_transform(size):
    return transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])


class FiveViews:
    def __init__(self, size, patch=32):
        self.size = size
        self.patch = patch

    def __call__(self, image):
        size = self.size
        bicubic = InterpolationMode.BICUBIC
        canvas = TF.resize(image, size, interpolation=bicubic)
        width, height = canvas.size
        tensors = []
        for fraction, flipped in VIEWS:
            view = canvas
            if fraction < 1.0:
                crop_w = max(self.patch, round(width * fraction))
                crop_h = max(self.patch, round(height * fraction))
                view = TF.crop(view, (height - crop_h) // 2, (width - crop_w) // 2,
                               crop_h, crop_w)
                view = TF.resize(view, [height, width], interpolation=bicubic)
            if flipped:
                view = TF.hflip(view)
            if view.size != (size, size):
                view = TF.resize(view, [size, size], interpolation=bicubic)
            tensors.append(TF.normalize(TF.to_tensor(view), MEAN, STD))
        return torch.stack(tensors)


class Images(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB")), index


def jpeg_cache(paths, folder, size, workers):
    names = [f"{i:06d}.jpg" for i in range(len(paths))]
    sources = []
    for path in paths:
        info = path.stat()
        sources.append({"path": str(path), "size": info.st_size, "modified_ns": info.st_mtime_ns})
    description = {"short_side": size, "quality": 85, "sources": sources}
    if folder.exists():
        prior = json.loads((folder / "inputs.json").read_text(encoding="utf-8"))
        if prior != description or not all((folder / name).is_file() for name in names):
            raise ValueError("The existing JPEG cache belongs to different or incomplete inputs")
        return [folder / name for name in names]

    with tempfile.TemporaryDirectory(prefix="image-cache-", dir=folder.parent) as temp:
        target = Path(temp) / "images"
        target.mkdir()

        def prepare(item):
            path, name = item
            with Image.open(path) as image:
                image = image.convert("RGB")
                width, height = image.size
                if width <= height:
                    shape = (size, max(1, round(height * size / width)))
                else:
                    shape = (max(1, round(width * size / height)), size)
                image.resize(shape, Image.Resampling.BICUBIC).save(target / name, quality=85)

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for _ in pool.map(prepare, zip(paths, names, strict=True)):
                pass
        (target / "inputs.json").write_text(json.dumps(description, ensure_ascii=False), encoding="utf-8")
        target.rename(folder)
    return [folder / name for name in names]
