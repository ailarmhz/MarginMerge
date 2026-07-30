from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import gapcover as gc
import clustering as sf
FEATURE_NAMES = ['anchor_cos', 'cos_unw_centroid', 'cos_resp_centroid', 'q_resp', 'log_cluster_size', 'rel_cluster_size', 'dist_rank', 'pos_x', 'pos_y', 'is_anchor', 'local_redundancy', 'resp_wmean', 'resp_wmax', 'resp_wstd', 'resp_coverage']
N_FEAT = len(FEATURE_NAMES)
SPATIAL_AVAILABLE = False

def _norm(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-08)

def fixed_clusters(V: torch.Tensor, Z: torch.Tensor, w: torch.Tensor, k: int, tau: float=0.05, backend: str='lazy'):
    A = gc.response_matrix(V, Z)
    anchors = gc.gapcover_select(V, Z, w, k, 'soft_gap', tau, backend)
    assign = sf.assign_clusters(V, A, w, anchors, mode='embedding_cosine')
    return (A, anchors, assign)

def base_weights(V, A, w, anchors, assign, tau_response=0.05, tau_assign=0.1, importance_power=0.5, eps=1e-08):
    C = gc.soft_gap_coverage(A, tau_response)
    q = (w.unsqueeze(1) * C).sum(0)
    anchor_cos = (V * V[anchors][assign]).sum(1)
    bw = torch.exp(anchor_cos / tau_assign) * (q + eps) ** importance_power
    return (bw, q, anchor_cos)

@torch.no_grad()
def patch_features(V, A, w, anchors, assign, q, anchor_cos) -> torch.Tensor:
    n, d = V.shape
    k = anchors.numel()
    dev = V.device
    sizes = torch.bincount(assign, minlength=k).float()
    csize = sizes[assign]
    acc_u = torch.zeros(k, d, device=dev)
    acc_u.index_add_(0, assign, V)
    cen_u = _norm(acc_u)[assign]
    bw = torch.exp(anchor_cos / 0.1) * (q + 1e-08) ** 0.5
    acc_r = torch.zeros(k, d, device=dev)
    acc_r.index_add_(0, assign, bw.unsqueeze(1) * V)
    cen_r = _norm(acc_r)[assign]
    f_cos_u = (V * cen_u).sum(1)
    f_cos_r = (V * cen_r).sum(1)
    order = torch.argsort(anchor_cos, descending=True)
    rank = torch.empty(n, device=dev)
    rank[order] = torch.arange(n, device=dev).float()
    key = assign.float() * 1000000.0 - anchor_cos
    o2 = torch.argsort(key)
    pos_in = torch.empty(n, device=dev)
    start = torch.zeros(k, device=dev)
    cnt = torch.zeros(k, device=dev)
    a_sorted = assign[o2]
    run = torch.arange(n, device=dev).float()
    first = torch.full((k,), float('inf'), device=dev)
    first.scatter_reduce_(0, a_sorted, run, reduce='amin', include_self=True)
    pos_in[o2] = run - first[a_sorted]
    dist_rank = pos_in / (csize - 1).clamp_min(1)
    is_anchor = torch.zeros(n, device=dev)
    is_anchor[anchors] = 1.0
    G = V @ V.T
    same = assign.unsqueeze(0) == assign.unsqueeze(1)
    G = G.masked_fill(~same, -2.0).fill_diagonal_(-2.0)
    kk = int(min(5, max(1, int(sizes.max().item()) - 1)))
    topv = G.topk(kk, dim=1).values
    local_red = torch.where(topv > -1.9, topv, torch.zeros_like(topv)).sum(1) / (topv > -1.9).sum(1).clamp_min(1).float()
    wn = (w / w.sum()).unsqueeze(1)
    r_wmean = (wn * A).sum(0)
    r_wmax = A.max(0).values
    r_wstd = (wn * (A - r_wmean.unsqueeze(0)) ** 2).sum(0).clamp_min(0).sqrt()
    r_cov = q
    zeros = torch.zeros(n, device=dev)
    F = torch.stack([anchor_cos, f_cos_u, f_cos_r, q, torch.log1p(csize), csize / n, dist_rank, zeros, zeros, is_anchor, local_red, r_wmean, r_wmax, r_wstd, r_cov], dim=1)
    return F.float()

class WeightMLP(nn.Module):

    def __init__(self, in_dim: int=N_FEAT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        return self.net(F).squeeze(-1).clamp(-5.0, 5.0)

def reps_from_weights(V, assign, k, weight):
    denom = torch.zeros(k, device=V.device, dtype=weight.dtype).index_add_(0, assign, weight)
    alpha = weight / denom[assign].clamp_min(1e-12)
    acc = torch.zeros(k, V.shape[1], device=V.device, dtype=V.dtype).index_add_(0, assign, alpha.unsqueeze(1) * V)
    return (_norm(acc), alpha)

def response_centroid(V, Z, w, k, **kw) -> Tuple[torch.Tensor, Dict]:
    V = V.float()
    A, anchors, assign = fixed_clusters(V, Z, w, k, **kw)
    bw, q, ac = base_weights(V, A, w, anchors, assign)
    R, alpha = reps_from_weights(V, assign, anchors.numel(), bw)
    return (R, {'anchors': anchors, 'assign': assign, 'A': A, 'alpha': alpha})

def marginmerge_reps(V, Z, w, k, mlp: Optional[WeightMLP], norm_stats: Optional[Dict]=None, cache: Optional[Dict]=None, **kw):
    V = V.float()
    if cache is None:
        A, anchors, assign = fixed_clusters(V, Z, w, k, **kw)
        bw, q, ac = base_weights(V, A, w, anchors, assign)
        F = patch_features(V, A, w, anchors, assign, q, ac)
        cache = {'A': A, 'anchors': anchors, 'assign': assign, 'bw': bw, 'F': F}
    A, anchors, assign, bw, F = (cache['A'], cache['anchors'], cache['assign'], cache['bw'], cache['F'])
    kk = anchors.numel()
    if mlp is None:
        R, alpha = reps_from_weights(V, assign, kk, bw)
        return (R, {'alpha': alpha, 'h_abs': 0.0, 'cache': cache})
    if norm_stats is None:
        raise RuntimeError('marginmerge: feature-normalization stats are REQUIRED with a trained MLP')
    Fn = (F - norm_stats['mean'].to(F.device)) / norm_stats['std'].to(F.device).clamp_min(1e-06)
    h = mlp(Fn)
    weight = bw * torch.exp(h)
    R, alpha = reps_from_weights(V, assign, kk, weight)
    return (R, {'alpha': alpha, 'h': h, 'h_abs': float(h.abs().mean().detach()), 'cache': cache})

def cluster_entropy(alpha, assign, k):
    ent = torch.zeros(k, device=alpha.device).index_add_(0, assign, -(alpha * torch.log(alpha + 1e-08)))
    sizes = torch.bincount(assign, minlength=k).float().clamp_min(2)
    return ent / torch.log(sizes)
