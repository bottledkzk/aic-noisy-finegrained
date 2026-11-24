"""Train the five-stage recipe on a user-provided labeled image collection."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import jpeg_cache, read_inputs, training_transform
from .groupcache import build_groups, posterior, soften
from .losses import OnlineTeacher, RobustSchedule, objective
from .model import VisualClassifier, execution_device, extract_features, image_loader, precision

STAGES = (
    "ce", "groupcache", "sp2", "teacher", "student",
)
PARENTS = {"sp2": "groupcache", "teacher": "sp2", "student": "sp2"}


def model_for(base, classes, size, device):
    model = VisualClassifier(base, classes)
    patch = model.clip.config.vision_config.patch_size
    if size < patch or size % patch:
        raise ValueError(f"Input size must be a positive multiple of the CLIP patch size ({patch})")
    return model.to(device)


def restore(model, path, inputs, size, *, require_public_checkpoint=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint.get("meta")
    if meta is None:
        eta = checkpoint.get("eta", {})
        if (require_public_checkpoint or checkpoint.get("mode") != "full" or
                eta.get("head_in") != "proj" or eta.get("img_size") != size or
                eta.get("n_fit") != len(inputs.keys) or eta.get("engineering_smoke", False)):
            raise ValueError("Incompatible model metadata for this input and visual recipe")
    elif (meta["classes"] != inputs.classes or meta["image_size"] != size or
          meta["train_keys"] != inputs.keys or meta["train_labels"] != inputs.labels.tolist()):
        raise ValueError("The checkpoint expects a different order, labels, classes or input size")
    model.load_visual_weights(checkpoint["state_dict"])


def training_groups(args, inputs, device):
    group_file = args.work / "groups.npy"
    groups = None
    if args.group_ids:
        groups = np.load(args.group_ids, allow_pickle=False)
    if group_file.exists():
        previous = np.load(group_file, allow_pickle=False)
        if groups is not None and not np.array_equal(groups, previous):
            raise ValueError("The work directory contains different training groups")
        groups = previous
    if groups is None:
        base = model_for(args.base_model, len(inputs.classes), args.size, device)
        features, _ = extract_features(base, inputs.paths, args.size, args.feature_batch,
                                       args.workers, device, five_view=False)
        groups, _ = build_groups(features.astype(np.float16), device=str(device))
    if groups.shape != (len(inputs.paths),) or not np.issubdtype(groups.dtype, np.integer):
        raise ValueError("Provide a one-dimensional integer group ID array aligned to the training CSV")
    if not group_file.exists():
        np.save(group_file, groups)
    return torch.as_tensor(groups, dtype=torch.long, device=device)


def teacher_anchors(args, inputs, teacher_stage, device):
    groups = training_groups(args, inputs, device)
    teacher = model_for(args.base_model, len(inputs.classes), args.size, device)
    restore(teacher, args.work / f"{teacher_stage}.pt", inputs, args.size,
            require_public_checkpoint=True)
    features, _ = extract_features(teacher, inputs.paths, args.size, args.feature_batch,
                                   args.workers, device)
    del teacher
    labels = torch.from_numpy(inputs.labels).to(device)
    largest_group = int(torch.unique(groups, return_counts=True)[1].max())
    k = min(args.target_k, len(labels) - largest_group)
    if k < 1:
        raise ValueError("Group-excluded targets need at least two groups")
    probs, _ = posterior(torch.from_numpy(features).to(device), labels, groups, len(inputs.classes), k=k)
    return soften(probs, labels)[0]


def build_optimizer(model, early):
    lr = 1e-5 if early else 2e-6
    vision = model.clip.vision_model
    layers = vision.encoder.layers
    position = vision.embeddings.position_embedding.weight
    projection = list(model.clip.visual_projection.parameters())
    layer_groups = [list(layer.parameters()) for layer in layers]
    reserved = {id(p) for group in [*layer_groups, projection, [position]] for p in group}
    other = [p for p in vision.parameters() if id(p) not in reserved]
    groups = [{"params": params, "lr": lr * 0.8 ** (len(layers) - 1 - i), "weight_decay": 0.0}
              for i, params in enumerate(layer_groups)]
    groups.extend([
        {"params": other, "lr": lr * 0.8 ** len(layers), "weight_decay": 0.0},
        {"params": projection, "lr": lr, "weight_decay": 0.0},
        {"params": model.head.parameters(), "lr": 3e-3 if early else 6e-4, "weight_decay": 1e-4},
        {"params": [position], "lr": 1e-6 if early else 2e-7, "weight_decay": 0.0},
    ])
    return torch.optim.AdamW(groups)


def train_stage(args, inputs, stage, device):
    path = args.work / f"{stage}.pt"
    if path.exists():
        raise FileExistsError(path)
    required = ({"groupcache": ("ce",), "sp2": ("ce", "groupcache"),
                 "teacher": ("ce", "sp2"), "student": ("teacher", "sp2")}).get(stage, ())
    for name in required:
        if not (args.work / f"{name}.pt").is_file():
            raise FileNotFoundError(f"Stage {stage} requires {name}.pt in the work directory")

    classes = len(inputs.classes)
    labels = torch.from_numpy(inputs.labels).to(device)
    if stage == "ce":
        anchors = F.one_hot(labels, classes).float()
    else:
        teacher_stage = "teacher" if stage == "student" else "ce"
        anchor_file = args.work / f"targets_{teacher_stage}.npy"
        if not anchor_file.exists():
            fresh = teacher_anchors(args, inputs, teacher_stage, device).cpu().numpy()
            with tempfile.TemporaryDirectory(dir=args.work) as temp:
                new_targets = Path(temp) / anchor_file.name
                np.save(new_targets, fresh)
                new_targets.rename(anchor_file)
        targets = np.load(anchor_file, allow_pickle=False)
        if (targets.shape != (len(labels), classes) or not np.isfinite(targets).all() or
                np.min(targets) < 0 or not np.allclose(targets.sum(1), 1, atol=2e-6, rtol=0)):
            raise ValueError(f"Invalid target probabilities in {anchor_file.name}")
        anchors = torch.as_tensor(targets, dtype=torch.float32, device=device)

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    model = model_for(args.base_model, classes, args.size, device)
    if stage in PARENTS:
        restore(model, args.work / f"{PARENTS[stage]}.pt", inputs, args.size,
                require_public_checkpoint=True)
    opt = build_optimizer(model, early=stage in ("ce", "groupcache"))
    params = [p for p in model.parameters() if p.requires_grad]
    robust = RobustSchedule(classes, device) if stage in ("teacher", "student") else None
    ema = OnlineTeacher(model, anchors) if stage in PARENTS else None
    cached = jpeg_cache(inputs.paths, args.work / "jpeg", args.cache_size, args.workers)
    loader = image_loader(cached, training_transform(args.size, args.cache_size), args.batch,
                          args.workers, device, shuffle=True)
    steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, [group["lr"] for group in opt.param_groups],
                                                     total_steps=steps, pct_start=0.2 if steps == 10 else 0.1)
    model.train()
    for epoch in range(1, args.epochs + 1):
        loss_sum = torch.zeros((), device=device)
        for images, ids in loader:
            images, ids = images.to(device, non_blocking=True), ids.to(device, non_blocking=True)
            targets = anchors[ids] if ema is None else ema.target(images, ids, epoch)
            mix = float(rng.beta(0.2, 0.2))
            mix = max(mix, 1 - mix)
            order = torch.randperm(len(ids), device=device)
            images = mix * images + (1 - mix) * images[order]
            targets = mix * targets + (1 - mix) * targets[order]
            opt.zero_grad(set_to_none=True)
            with precision(device):
                logits = model(images, feature_noise=0.03)
            if robust is None:
                loss = objective(logits, targets, 0.15, 0.0)[0]
            else:
                loss = robust.loss(logits, targets, labels[ids], epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if ema is not None:
                ema.update()
            if scheduler.last_epoch < steps - 1:
                scheduler.step()
            loss_sum += loss.detach() * len(ids)
        row = {"stage": stage, "epoch": epoch, "train_loss": float(loss_sum / len(labels))}
        if ema is not None:
            row["train_ot_column_error"] = ema.finish_epoch()["class_relative_error"]
        if robust is not None:
            robust.finish_epoch(epoch, len(labels))
            row["erce_scale"] = robust.scale
        print(json.dumps(row), flush=True)

    with tempfile.TemporaryDirectory(dir=args.work) as tmp:
        temp_file = Path(tmp) / path.name
        torch.save({
            "state_dict": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad},
            "meta": {"stage": stage, "image_size": args.size, "classes": inputs.classes,
                     "train_keys": inputs.keys, "train_labels": inputs.labels.tolist(),
                     "epoch": args.epochs, "seed": args.seed},
        }, temp_file)
        temp_file.rename(path)
    print(f"Saved {path}", flush=True)


def train(args):
    stages = STAGES if args.stage == "all" else (args.stage,)
    if any(stage in ("teacher", "student") for stage in stages) and args.epochs < 9:
        raise ValueError("SCE2 stages need at least nine epochs to include the calibrated ERCE term")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0 or args.feature_batch < 1 or args.target_k < 1:
        raise ValueError("Epochs, batch sizes and target k must be positive and workers nonnegative")
    if args.size < 1 or args.cache_size < args.size:
        raise ValueError("The JPEG cache short side must be at least the input size")
    inputs = read_inputs(args.train_csv, args.data_root, labeled=True)
    device = execution_device(args.device)
    args.work = Path(args.work).resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = args.work / ".training"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        for stage in stages:
            if (args.work / f"{stage}.pt").exists():
                raise FileExistsError(args.work / f"{stage}.pt")
        description = {"data_root": str(Path(args.data_root).resolve()),
                       "base": str(Path(args.base_model).resolve()), "train_keys": inputs.keys,
                       "train_labels": inputs.labels.tolist(), "classes": inputs.classes,
                       "image_size": args.size, "cache_size": args.cache_size, "target_k": args.target_k}
        inputs_file = args.work / "training_inputs.json"
        if inputs_file.exists():
            if json.loads(inputs_file.read_text(encoding="utf-8")) != description:
                raise ValueError("The work directory belongs to different inputs or target settings")
        else:
            inputs_file.write_text(json.dumps(description, ensure_ascii=False), encoding="utf-8")
        if args.group_ids and (args.work / "groups.npy").is_file():
            current = np.load(args.group_ids, allow_pickle=False)
            previous = np.load(args.work / "groups.npy", allow_pickle=False)
            if not np.array_equal(current, previous):
                raise ValueError("The work directory contains different training groups")
        for stage in stages:
            train_stage(args, inputs, stage, device)
    finally:
        lock.unlink()
