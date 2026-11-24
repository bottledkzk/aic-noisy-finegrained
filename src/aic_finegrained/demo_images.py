"""Generate a tiny noisy-label dataset and exercise the complete image pipeline."""

import csv
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig

from .infer import infer
from .train import train


def make_inputs(root, seed):
    rng = np.random.default_rng(seed)
    images = root / "images"
    images.mkdir()
    names = [f"item-{i:02d}.png" for i in range(18)]
    labels = []
    for i, name in enumerate(names):
        group = i // 2
        label = group % 3
        height, width = 48 + group % 5, 52 + group % 7
        y, x = np.ogrid[:height, :width]
        bands = (y, x, x + y)[label]
        stripe = (bands + group) % 12 < 5
        pixels = np.where(stripe, 168, 75)[..., None] + rng.integers(-18, 19, (height, width, 3))
        Image.fromarray(pixels.astype(np.uint8)).save(images / name)
        observed = (label + 1) % 3 if i in (1, 8) else label
        labels.append(f"class-{observed}")

    with (root / "train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label"])
        writer.writerows(zip(names[:12], labels[:12], strict=True))
    (root / "query.csv").write_text("path\n" + "\n".join(names[12:]) + "\n", encoding="utf-8")
    np.save(root / "groups.npy", np.repeat(np.arange(6), 2))

    text = CLIPTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_attention_heads=2,
                          num_hidden_layers=1, max_position_embeddings=8, bos_token_id=1, eos_token_id=2)
    vision = CLIPVisionConfig(hidden_size=24, intermediate_size=32, num_attention_heads=3,
                              num_hidden_layers=1, image_size=32, patch_size=16)
    torch.manual_seed(seed)
    CLIPModel(CLIPConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         projection_dim=8)).save_pretrained(root / "base")


def run_demo(out, seed):
    root = Path(out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    threads = torch.get_num_threads()
    torch.set_num_threads(min(threads, 2))
    try:
        make_inputs(root, seed)
        common = dict(data_root=root / "images", train_csv=root / "train.csv", base_model=root / "base",
                      device="cpu", size=48, batch=4, workers=0)
        work = root / "training"
        train(Namespace(**common, stage="all", epochs=9, seed=seed, cache_size=64,
                        work=work, feature_batch=4, target_k=60, group_ids=root / "groups.npy"))
        infer(Namespace(**common, query_csv=root / "query.csv", checkpoint=work / "student.pt",
                        out=root / "predictions.csv", canonical_out=root / "canonical.npy",
                        bank_indices=None, neighbor_k=60, graph_k=10, method="sot", rho=9.0,
                        temperature=1.0))
    finally:
        torch.set_num_threads(threads)
