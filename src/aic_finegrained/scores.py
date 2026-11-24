"""Fuse scores and compare inference-time marginal constraints."""

import numpy as np
from scipy.special import logsumexp

from .transport import solve


def fuse_scores(head_logits, neighbor_probabilities, alpha=0.2):
    head = np.asarray(head_logits, dtype=np.float32)
    neighbors = np.asarray(neighbor_probabilities, dtype=np.float32)
    if head.ndim != 2 or head.shape != neighbors.shape:
        raise ValueError("Head logits and neighbor probabilities must have equal N by C shape")
    return (head + alpha * np.log(np.maximum(neighbors, 1e-12))).astype(np.float64)


def calibrate_scores(scores, method="sot", rho=9.0, temperature=1.0):
    if method in ("sot", "none"):
        q, _, info = solve(scores, rho=rho if method == "sot" else 0.0, temperature=temperature)
        return q, info
    if method != "balanced":
        raise ValueError(f"Unknown calibration method: {method}")

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("A finite nonempty N by C score matrix is required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    kernel = scores / temperature
    kernel -= kernel.max(axis=1, keepdims=True)
    rows, classes = kernel.shape
    desired_mass = rows / classes
    column_scale = np.zeros(classes)
    for iteration in range(1, 2001):
        log_q = kernel + column_scale
        log_q -= logsumexp(log_q, axis=1, keepdims=True)
        log_mass = logsumexp(log_q, axis=0)
        error = float(np.max(np.abs(np.expm1(log_mass - np.log(desired_mass)))))
        if error <= 1e-9:
            return np.exp(log_q), {"iterations": iteration, "column_relative_error": error}
        column_scale += np.log(desired_mass) - log_mass
        column_scale -= column_scale.mean()
    raise RuntimeError("Balanced transport did not converge")


def synthetic_comparison(seed=43):
    """Create an imbalanced population and three matched score-only comparisons."""
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(4), [120, 90, 60, 30])
    scores = rng.normal(size=(len(labels), 4))
    scores[np.arange(len(labels)), labels] += 2.6
    scores[:, 0] += 0.6
    outcomes = []
    for method in ("none", "balanced", "sot"):
        q, info = calibrate_scores(scores, method)
        outcomes.append({
            "method": method,
            "synthetic_accuracy": round(float(np.mean(q.argmax(1) == labels)), 4),
            "soft_class_mass": q.sum(0).round(2).tolist(),
            "iterations": info["iterations"],
        })
    return {"seed": seed, "synthetic_class_counts": np.bincount(labels).tolist(), "results": outcomes}
