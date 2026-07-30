from typing import Optional, Tuple, List
import math
import torch

def response_matrix(V: torch.Tensor, Z: torch.Tensor, chunk: int=0) -> torch.Tensor:
    V = V.float()
    Z = Z.float()
    if chunk and V.shape[0] > chunk:
        cols = [Z @ V[i:i + chunk].T for i in range(0, V.shape[0], chunk)]
        A = torch.cat(cols, dim=1)
    else:
        A = Z @ V.T
    return A.clamp_(-1.0, 1.0)

def soft_gap_coverage(A: torch.Tensor, tau: float) -> torch.Tensor:
    b = A.max(dim=1, keepdim=True).values
    delta = (b - A).clamp_(min=0.0)
    scaled = (delta / tau).clamp_(max=50.0)
    return torch.exp(-scaled)

def _pick_best(gains: torch.Tensor, avail: torch.Tensor, tie_tol: float) -> int:
    g = gains.clone()
    g[~avail] = float('-inf')
    gmax = g.max()
    tied = (g >= gmax - tie_tol) & avail
    return int(tied.nonzero(as_tuple=True)[0][0].item())

def _greedy_reference(score: torch.Tensor, weight: torch.Tensor, k: int, init_val: float, tie_tol: float) -> List[int]:
    M, n = score.shape
    cur = torch.full((M,), init_val, device=score.device, dtype=torch.float32)
    w = weight.float()
    avail = torch.ones(n, dtype=torch.bool, device=score.device)
    init_contrib = (w.unsqueeze(1) * score).sum(0)
    selected: List[int] = []
    while len(selected) < k:
        gains = (w.unsqueeze(1) * (score - cur.unsqueeze(1)).clamp(min=0.0)).sum(0)
        gmax = gains.masked_fill(~avail, float('-inf')).max()
        if gmax.item() <= tie_tol:
            order = torch.argsort(init_contrib.masked_fill(~avail, float('-inf')), descending=True)
            for j in order.tolist():
                if not avail[j]:
                    continue
                selected.append(j)
                avail[j] = False
                if len(selected) == k:
                    break
            break
        best = _pick_best(gains, avail, tie_tol)
        selected.append(best)
        avail[best] = False
        cur = torch.maximum(cur, score[:, best])
    return selected

def _greedy_lazy(score: torch.Tensor, weight: torch.Tensor, k: int, init_val: float, tie_tol: float) -> List[int]:
    import heapq
    M, n = score.shape
    w = weight.float()
    cur = torch.full((M,), init_val, device=score.device, dtype=torch.float32)
    init_contrib = (w.unsqueeze(1) * score).sum(0)
    g0 = (w.unsqueeze(1) * (score - cur.unsqueeze(1)).clamp(min=0.0)).sum(0)
    heap = [(-float(g0[j]), j, 0) for j in range(n)]
    heapq.heapify(heap)
    selected: List[int] = []
    picked = torch.zeros(n, dtype=torch.bool, device=score.device)
    step = 0
    sc_cols = score
    while len(selected) < k and heap:
        neg_g, j, upd = heapq.heappop(heap)
        if picked[j]:
            continue
        if upd == step:
            if heap:
                ng, nj, nu = heap[0]
                if abs(-neg_g - -ng) <= tie_tol and nj < j and (nu == step) and (not picked[nj]):
                    heapq.heappush(heap, (neg_g, j, upd))
                    heapq.heappop(heap)
            selected.append(j)
            picked[j] = True
            cur = torch.maximum(cur, sc_cols[:, j])
            step += 1
        else:
            g = float((w * (sc_cols[:, j] - cur).clamp(min=0.0)).sum())
            heapq.heappush(heap, (-g, j, step))
    if len(selected) < k:
        order = torch.argsort(init_contrib, descending=True)
        for j in order.tolist():
            if not picked[j]:
                selected.append(j)
                picked[j] = True
                if len(selected) == k:
                    break
    return selected

def kof(rho: float, n: int) -> int:
    return min(n, max(1, math.ceil(rho * n)))

def gapcover_select(V: torch.Tensor, Z: torch.Tensor, weight: torch.Tensor, k: int, objective: str='soft_gap', tau: float=0.05, backend: str='lazy', tie_tol: float=1e-10, chunk: int=0) -> torch.Tensor:
    assert V.dim() == 2 and Z.dim() == 2 and (V.shape[1] == Z.shape[1])
    n = V.shape[0]
    k = min(k, n)
    A = response_matrix(V, Z, chunk=chunk)
    if objective == 'soft_gap':
        score, init = (soft_gap_coverage(A, tau), 0.0)
    elif objective == 'direct_regret':
        score, init = (A, -1.0)
    else:
        raise ValueError(f'unknown objective {objective!r}')
    eng = _greedy_lazy if backend == 'lazy' else _greedy_reference
    sel = eng(score, weight, k, init, tie_tol)
    return torch.tensor(sorted(sel), device=V.device, dtype=torch.long)

def response_orthogonal_select(V: torch.Tensor, Z: torch.Tensor, weight: torch.Tensor, k: int, tau: float=0.05, eps: float=1e-08, chunk: int=0) -> torch.Tensor:
    n = V.shape[0]
    k = min(k, n)
    A = response_matrix(V, Z, chunk=chunk)
    C = soft_gap_coverage(A, tau)
    w = weight.float()
    sw = w.clamp(min=0).sqrt()
    R = C * sw.unsqueeze(1)
    q = (w.unsqueeze(1) * C).sum(0)
    selected: List[int] = []
    picked = torch.zeros(n, dtype=torch.bool, device=V.device)
    basis: List[torch.Tensor] = []
    first = int(torch.argmax(q).item())
    selected.append(first)
    picked[first] = True
    v = R[:, first].clone()
    nv = v.norm()
    if nv > eps:
        basis.append(v / nv)
    while len(selected) < k:
        if basis:
            B = torch.stack(basis, dim=1)
            proj = B @ (B.T @ R)
            residual_norm = (R - proj).norm(dim=0)
        else:
            residual_norm = R.norm(dim=0)
        score = q * residual_norm
        score[picked] = float('-inf')
        if float(residual_norm.masked_fill(picked, 0).max()) < eps:
            for j in torch.argsort(q, descending=True).tolist():
                if not picked[j]:
                    selected.append(j)
                    picked[j] = True
                    if len(selected) == k:
                        break
            break
        gmax = score.max()
        tied = (score >= gmax - 1e-12) & ~picked
        j = int(tied.nonzero(as_tuple=True)[0][0].item())
        selected.append(j)
        picked[j] = True
        v = R[:, j].clone()
        for _ in range(2):
            for b in basis:
                v = v - b @ v * b
        nv = v.norm()
        if nv > eps:
            basis.append(v / nv)
    return torch.tensor(sorted(selected), device=V.device, dtype=torch.long)

def predicted_u_facility_select(P: torch.Tensor, p: torch.Tensor, pool_idx: torch.Tensor, k: int, alpha: float=1.0, tau_patch: float=0.1, tie_tol: float=1e-10) -> torch.Tensor:
    Ppool = P[pool_idx].float()
    ppool = p[pool_idx].float()
    m = Ppool.shape[0]
    k = min(k, m)
    demand = ppool.clamp(min=1e-08) ** alpha
    demand = demand / demand.sum()
    cos = (Ppool @ Ppool.T).clamp(-1, 1)
    K = torch.exp(-(1.0 - cos) / tau_patch)
    sel = _greedy_reference(K * 1.0, demand, k, 0.0, tie_tol)
    return pool_idx[torch.tensor(sorted(sel), device=P.device)]

def objective_value(score: torch.Tensor, weight: torch.Tensor, sel: torch.Tensor, init_val: float=0.0) -> float:
    cur = score[:, sel].max(dim=1).values if len(sel) else torch.full((score.shape[0],), init_val, device=score.device)
    return float((weight.float() * cur.clamp(min=init_val)).sum())

def coverage_stats(A: torch.Tensor, weight: torch.Tensor, sel: torch.Tensor, tau: float) -> dict:
    C = soft_gap_coverage(A, tau)
    w = weight.float()
    cov = C[:, sel].max(dim=1).values if len(sel) else torch.zeros(A.shape[0], device=A.device)
    b = A.max(dim=1).values
    resp = A[:, sel].max(dim=1).values if len(sel) else torch.full_like(b, -1.0)
    return {'soft_coverage_wmean': float((w * cov).sum()), 'soft_coverage_mean': float(cov.mean()), 'soft_coverage_min': float(cov.min()), 'soft_coverage_p10': float(torch.quantile(cov, 0.1)), 'weighted_regret': float((w * (b - resp)).sum())}
