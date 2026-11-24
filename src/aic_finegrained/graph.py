"""One-step fixed-model score smoothing on an exact mutual-neighbor graph."""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy import sparse


def exact_neighbors(features, k=10, *, ids=None, device='cpu', block=512,
                    dtype=torch.float32):
    x = np.asarray(features)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) <= k or k < 1 or block < 1:
        raise ValueError('Finite nonempty features, 1 <= k < n and block >= 1 required')
    n = len(x)
    ids = np.arange(n, dtype=np.int64) if ids is None else np.asarray(ids)
    if ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != n:
        raise ValueError('Unique integer canonical identifiers required')
    # Always compute in the same manifest order, even when the caller permutes rows.
    order = np.argsort(ids, kind='stable')
    xx = torch.as_tensor(np.ascontiguousarray(x[order]), device=device, dtype=dtype)
    norms = torch.linalg.vector_norm(xx, dim=1)
    if bool((norms <= 0).any()):
        raise ValueError('Zero feature vector cannot define cosine neighbors')
    xx = xx / norms[:, None]
    neighbors = np.empty((n, k), dtype=np.int64)
    similarities = np.empty((n, k), dtype=np.float64)
    boundary_ties = 0
    start = time.perf_counter()
    first_block_seconds = None
    take = min(k + 1, n - 1)
    with torch.no_grad():
        for left in range(0, n, block):
            right = min(n, left + block)
            sim = xx[left:right] @ xx.T
            sim[torch.arange(right-left, device=device),
                torch.arange(left, right, device=device)] = -torch.inf
            vv, ii = sim.topk(take, dim=1, largest=True, sorted=True)
            values, indices = vv.cpu().numpy(), ii.cpu().numpy()
            for offset in range(right-left):
                if take > k and values[offset, k-1] == values[offset, k]:
                    # topk does not specify which equal value crosses its boundary.
                    # Resolve the full tied row by its stable manifest identifier.
                    whole = sim[offset].cpu().numpy()
                    chosen = np.lexsort((np.arange(n), -whole))[:k]
                    neighbors[left+offset] = chosen
                    similarities[left+offset] = whole[chosen]
                    boundary_ties += 1
                else:
                    rank = np.lexsort((indices[offset], -values[offset]))[:k]
                    neighbors[left+offset] = indices[offset, rank]
                    similarities[left+offset] = values[offset, rank]
            if left == 0:
                first_block_seconds = time.perf_counter() - start
                estimate = first_block_seconds * n / (right-left)
                print(f'GRAPH_FIRST_BLOCK {right-left}/{n} {first_block_seconds:.3f}s '
                      f'linear_estimate={estimate:.1f}s', flush=True)
            if right == n or (left // block) % 16 == 15:
                print(f'GRAPH_NEIGHBORS {right}/{n}', flush=True)
    result_neighbors = np.empty_like(neighbors)
    result_similarities = np.empty_like(similarities)
    result_neighbors[order] = order[neighbors]
    result_similarities[order] = similarities
    info = {'k': k, 'n': n, 'dim': x.shape[1], 'block': block,
            'device': device, 'dtype': str(dtype), 'boundary_tie_rows': boundary_ties,
            'first_block_seconds': first_block_seconds,
            'search_seconds': time.perf_counter()-start,
            'input_norm_min': float(norms.min()), 'input_norm_max': float(norms.max()),
            'tie_break': 'ascending stable original manifest row identifier'}
    del xx, sim, vv, ii
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result_neighbors, result_similarities, info


def mutual_weights(neighbors, similarities, temperature=0.05):
    ix, values = np.asarray(neighbors), np.asarray(similarities)
    if (ix.ndim != 2 or ix.shape != values.shape or ix.size == 0 or
        not np.issubdtype(ix.dtype, np.integer) or not np.isfinite(values).all() or
        not np.isfinite(temperature) or temperature <= 0):
        raise ValueError('Valid neighbor indices/similarities and positive temperature required')
    n, k = ix.shape
    rows = np.arange(n)
    if ix.min() < 0 or ix.max() >= n or np.any(ix == rows[:, None]):
        raise ValueError('Neighbor indices must be in range and exclude self')
    if np.any(np.diff(np.sort(ix, axis=1), axis=1) == 0):
        raise ValueError('Repeated neighbors rejected')
    mutual = np.any(ix[ix] == rows[:, None, None], axis=2)
    degree = mutual.sum(1)
    isolated = degree == 0
    masked = np.where(mutual, values, -np.inf)
    maximum = masked.max(1)
    maximum[isolated] = 0.0
    weights = np.where(mutual, np.exp((masked-maximum[:, None])/temperature), 0.0)
    denominator = weights.sum(1)
    denominator[isolated] = 1.0
    weights /= denominator[:, None]
    rr = np.broadcast_to(rows[:, None], ix.shape)[mutual]
    cc = ix[mutual]
    data = weights[mutual]
    rr = np.concatenate([rr, rows[isolated]])
    cc = np.concatenate([cc, rows[isolated]])
    data = np.concatenate([data, np.ones(int(isolated.sum()))])
    w = sparse.csr_matrix((data, (rr, cc)), shape=(n, n))
    assert np.isfinite(w.data).all() and np.all(w.data >= 0)
    row_error = float(np.max(np.abs(np.asarray(w.sum(1)).ravel()-1)))
    assert row_error < 1e-12
    assert np.array_equal(w.diagonal() != 0, isolated)
    edges = w.copy(); edges.setdiag(0); edges.eliminate_zeros()
    support = edges.astype(bool)
    assert (support != support.T).nnz == 0
    info = {'temperature': temperature, 'mutual_edge_count_directed': int(mutual.sum()),
            'isolated_rows': int(isolated.sum()), 'isolated_fraction': float(isolated.mean()),
            'mutual_degree_min': int(degree.min()), 'mutual_degree_max': int(degree.max()),
            'mutual_degree_mean': float(degree.mean()),
            'degree_histogram': np.bincount(degree, minlength=k+1).tolist(),
            'weight_row_abs_error': row_error}
    return w, isolated, info


def smooth_scores(scores, weights, beta=0.10):
    s = np.asarray(scores)
    if (s.ndim != 2 or not np.isfinite(s).all() or weights.shape != (len(s), len(s)) or
        not np.isfinite(beta) or not 0 <= beta <= 1):
        raise ValueError('Finite scores, matching graph, 0 <= beta <= 1 required')
    if beta == 0:
        return s.copy()
    if (not np.isfinite(weights.data).all() or np.any(weights.data < 0) or
        np.max(np.abs(np.asarray(weights.sum(1)).ravel()-1)) > 1e-12):
        raise ValueError('Graph weights must be finite, nonnegative, row-normalized')
    candidate = (1-beta)*s + beta*(weights @ s)
    # Avoid two arithmetic roundings for the explicitly preregistered identity fallback.
    isolated = weights.diagonal() == 1
    candidate[isolated] = s[isolated]
    if not np.isfinite(candidate).all():
        raise RuntimeError('Nonfinite smoothed score')
    return candidate
