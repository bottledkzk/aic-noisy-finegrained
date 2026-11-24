"""Single-model image inference using a labeled training bank and one query cohort."""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import read_inputs
from .graph import exact_neighbors, mutual_weights, smooth_scores
from .model import execution_device, extract_features
from .scores import calibrate_scores, fuse_scores
from .train import model_for, restore


@torch.inference_mode()
def neighbor_probabilities(query, bank, labels, classes, k, device):
    bank = torch.from_numpy(bank).to(device)
    labels = torch.from_numpy(labels).to(device)
    probabilities = []
    for part in np.array_split(query, max(1, (len(query) + 2047) // 2048)):
        similarities = torch.from_numpy(part).to(device) @ bank.T
        values, indices = similarities.topk(k, dim=1)
        rows = torch.zeros((len(part), classes), device=device)
        rows.scatter_add_(1, labels[indices], torch.softmax(values / 0.02, dim=1))
        probabilities.append(rows.cpu().numpy())
    return np.concatenate(probabilities)


def infer(args):
    if args.batch < 1 or args.workers < 0 or args.neighbor_k < 1 or args.graph_k < 0:
        raise ValueError("Invalid batch size, worker count or neighborhood size")
    output = Path(args.out).resolve()
    scores_file = Path(args.canonical_out).resolve() if args.canonical_out else None
    if scores_file == output:
        raise ValueError("Use separate paths for predictions and canonical scores")
    for path in (output, scores_file):
        if path is not None and path.exists():
            raise FileExistsError(path)
    train = read_inputs(args.train_csv, args.data_root, labeled=True)
    query = read_inputs(args.query_csv, args.data_root, labeled=False)
    if args.bank_indices:
        bank_indices = np.load(args.bank_indices, allow_pickle=False)
        if (bank_indices.ndim != 1 or not len(bank_indices) or
                not np.issubdtype(bank_indices.dtype, np.integer) or np.min(bank_indices) < 0 or
                np.max(bank_indices) >= len(train.paths) or len(np.unique(bank_indices)) != len(bank_indices)):
            raise ValueError("Bank indices must be unique integer row positions in the training CSV")
    else:
        bank_indices = np.arange(len(train.paths))

    device = execution_device(args.device)
    classes = len(train.classes)
    model = model_for(args.base_model, classes, args.size, device)
    restore(model, args.checkpoint, train, args.size)
    paths = [train.paths[i] for i in bank_indices]
    bank, _ = extract_features(model, paths, args.size, args.batch, args.workers, device)
    features, logits = extract_features(model, query.paths, args.size, args.batch, args.workers, device)
    del model
    probabilities = neighbor_probabilities(features, bank, train.labels[bank_indices], classes,
                                            min(args.neighbor_k, len(bank)), device)
    canonical = fuse_scores(logits, probabilities)
    k = min(args.graph_k, len(query.paths) - 1)
    if k > 0:
        neighbors, similarities, _ = exact_neighbors(features, k=k, device=device)
        weights, _, _ = mutual_weights(neighbors, similarities, temperature=0.05)
        canonical = smooth_scores(canonical, weights, beta=0.1)
    q, info = calibrate_scores(canonical, args.method, args.rho, args.temperature)
    if scores_file:
        scores_file.parent.mkdir(parents=True, exist_ok=True)
        with scores_file.open("xb") as stream:
            np.save(stream, canonical)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "label"])
        writer.writerows((name, train.classes[index]) for name, index in zip(query.keys, q.argmax(1), strict=True))
    print(json.dumps({"rows": len(query.paths), "bank_rows": len(bank_indices),
                      "method": args.method, "iterations": info["iterations"], "output": str(output)}))
