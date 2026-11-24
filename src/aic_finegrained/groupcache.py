"""Automatic grouping and group-excluded training supervision."""

import time

import numpy as np
import torch

GROUP_THR = 0.98
GROUP_EPS = 1e-3


def l2_normalize_np(X, chunk=16384):
    """**确定性** L2 归一化：float64 + CPU + 成对求和 ⇒ 与设备/BLAS 无关。

    这是 D21 修复的第 ① 步。历史代码里 `l2()` 是 **float32 + numpy 除法**，
    而其它脚本用 `torch` 归一化 ⇒ 末位 ulp 不同 ⇒ 0.98 阈值下整划分漂移。
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"l2_normalize_np 需要 2-D 输入，收到 {X.shape}")
    out = np.empty_like(X)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        blk = X[s:e].astype(np.float64)
        nrm = np.sqrt(np.sum(blk * blk, axis=1))          # 成对求和，确定性
        nrm = np.where(nrm > 1e-12, nrm, 1.0)
        out[s:e] = (blk / nrm[:, None]).astype(np.float32)
    return out


def build_groups(X, thr=GROUP_THR, chunk=None, eps=GROUP_EPS, device=None,
                 normalize=True, return_diag=False):
    """近重复分组（**确定性版**，D21 修复）。

    ⚠️ 语义（与历史一致，不要改）：每张图连到它**最近的邻居**；
       若该相似度 `> thr` 则 union。阈值 0.98 极高（近乎同图）时才用并查集。

    🆕 确定性保证（2026-09-22）：
      · `normalize=True`（默认）⇒ 内部**强制** L2 归一化，调用方传 raw 也一样；
      · 判定 = 「float32 取候选超集」+「float64 精确重判」
        ⇒ 结果**与 device / BLAS / chunk 全部无关**；
      · `device=None` ⇒ 有 CUDA 用 CUDA（只为速度），**不影响结果**。

    返回：`(gid, n_edges)`；`return_diag=True` 时返回 `(gid, n_edges, diag)`。
    """
    t_start = time.time()
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    Z = np.asarray(X, dtype=np.float32)
    if normalize:
        Z = l2_normalize_np(Z)
    if Z.ndim != 2:
        raise ValueError(f"build_groups 需要 2-D 特征，收到 {Z.shape}")
    N, _D = Z.shape

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if chunk is None:                       # 让 sim 块 ≲ 2 GB；chunk 只影响内存/速度
        chunk = int(max(256, min(4096, (2 << 30) // (4 * max(N, 1)))))

    Zt = torch.from_numpy(Z).to(device)
    best = np.full(N, -1.0, dtype=np.float32)
    bidx = np.full(N, -1, dtype=np.int64)
    # ── 第 1 遍：行最大值（= 旧实现的全部计算）⇒ 决定哪些行"可能有边" ──
    with torch.no_grad():
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            sim = Zt[s:e] @ Zt.T
            sim[torch.arange(e - s, device=device),
                torch.arange(s, e, device=device)] = -2.0
            v, i = sim.max(dim=1)
            best[s:e] = v.cpu().numpy()
            bidx[s:e] = i.cpu().numpy()
            del sim

    # ── 第 2 遍：**只对** best > thr-eps 的行取"相似度超阈值"的候选 ──
    #    ⚠️ 设计要点（第一版写成"对全矩阵取候选"⇒ R2 上慢到不可用）：
    #       只有 `best > thr - eps` 的行才可能产生边；对其它行做全行扫描纯属浪费。
    #       而且这样一来，float64 精确重算的候选**只可能来自这些行**，数量极小。
    sel_idx = np.nonzero(best > (thr - eps))[0]
    rows = np.empty(0, np.int64)
    cols = np.empty(0, np.int64)
    n_border = 0
    if sel_idx.size:
        rl, cl = [], []
        with torch.no_grad():
            for s in range(0, sel_idx.size, chunk):
                sub = sel_idx[s:s + chunk]
                sub_t = torch.from_numpy(np.ascontiguousarray(sub)).to(device)
                sim = Zt[sub_t] @ Zt.T
                sim[torch.arange(sub.size, device=device), sub_t] = -2.0
                n_border += int(((sim - thr).abs() < 1e-5).sum().item())
                m = sim > (thr - eps)
                ri, ci = torch.nonzero(m, as_tuple=True)
                rl.append(sub[ri.cpu().numpy()])
                cl.append(ci.cpu().numpy())
                del sim, m
        rows = np.concatenate(rl)
        cols = np.concatenate(cl)
        del rl, cl
    del Zt
    if device == "cuda":
        torch.cuda.empty_cache()

    edges = []                                   # (row, neighbour) 精确判定后的边
    if rows.size:
        # ② 精确重算：float64 + np.sum（成对求和）⇒ 跨设备逐位一致
        S = np.empty(rows.size, dtype=np.float64)
        step = 65536
        for s in range(0, rows.size, step):
            e = min(s + step, rows.size)
            A = Z[rows[s:e]].astype(np.float64)
            B = Z[cols[s:e]].astype(np.float64)
            S[s:e] = np.sum(A * B, axis=1)
        keep = S > thr
        kr, kc, ks = rows[keep], cols[keep], S[keep]
        if kr.size:
            # rows/cols 来自 torch.nonzero ⇒ 已按 (row, col) 升序；
            # 每个 row 的候选按 col 升序 ⇒ argmax 取首个最大值 = 最小列号，无歧义。
            starts = np.flatnonzero(np.r_[True, kr[1:] != kr[:-1]])
            ends = np.r_[starts[1:], kr.size]
            edges = [(int(kr[a]), int(kc[a + int(np.argmax(ks[a:b]))]))
                     for a, b in zip(starts, ends)]

    # ③ 并查集（与历史实现逐字相同）
    parent = np.arange(N)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(N)])
    _, gid = np.unique(roots, return_inverse=True)
    n_edges = len(edges)

    diag = {"n_edges": n_edges, "n_candidates": int(rows.size),
            "n_sel_rows": int(sel_idx.size),
            "n_borderline_1e-5": int(n_border), "n_rows_with_edge": n_edges,
            "thr": float(thr), "eps": float(eps), "device": device,
            "chunk": int(chunk), "normalize": bool(normalize),
            "n": int(N), "dim": int(_D), "seconds": round(time.time() - t_start, 1),
            "best": best, "bidx": bidx}
    return (gid, n_edges, diag) if return_diag else (gid, n_edges)


def posterior(x, y, groups, classes, k=60, temperature=.02, block=512):
    n=len(x)
    assert x.ndim==2 and y.shape==groups.shape==(n,)
    assert torch.isfinite(x).all() and temperature>0
    _,counts=torch.unique(groups,return_counts=True)
    assert n-int(counts.max())>=k>0
    p=torch.empty((n,classes),dtype=x.dtype,device=x.device)
    removed=0
    for left in range(0,n,block):
        right=min(left+block,n)
        sim=x[left:right]@x.T
        same=groups[left:right,None]==groups[None,:]
        removed+=int(same.sum())-(right-left)
        sim.masked_fill_(same,-torch.inf)
        values,ix=sim.topk(k,dim=1)
        assert torch.isfinite(values).all() and not (groups[ix]==groups[left:right,None]).any()
        rows=torch.zeros((right-left,classes),dtype=x.dtype,device=x.device)
        rows.scatter_add_(1,y[ix],torch.softmax(values/temperature,dim=1))
        p[left:right]=rows
        if n>100 and (left==0 or right==n or (left//block)%32==31):
            print('GROUP_EXCLUDED_CACHE',right,n,flush=True)
    return p,removed


def soften(p,y):
    support=p.gather(1,y[:,None]).squeeze(1).clamp(0,1)
    rate=.5*(1-support)
    target=p*rate[:,None]
    target.scatter_add_(1,y[:,None],(1-rate)[:,None])
    assert torch.isfinite(target).all() and (target.sum(1)-1).abs().max()<2e-6
    assert target.gather(1,y[:,None]).min()>=.5-1e-6
    return target,support,rate
