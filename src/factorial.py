from typing import Dict, Optional
import torch
import gapcover as gc
import clustering as sf
import marginmerge as mm
ANCHOR_STRATEGIES = ('random', 'kcenter', 'coverage')
REPRESENTATIVE_STRATEGIES = ('anchor', 'uniform_centroid', 'response_centroid', 'learned')
LOSS_STRATEGIES = ('none', 'score_reconstruction', 'margin', 'full')
TAU = 0.05

def _stable_seed(base_seed: int, doc_id: int) -> int:
    return (base_seed * 1000003 + int(doc_id) * 97 + 12345) % (2 ** 31 - 1)

def select_anchors(V: torch.Tensor, Z: torch.Tensor, w: torch.Tensor, k: int, strategy: str, base_seed: int=42, doc_id: int=0) -> Dict:
    n = V.shape[0]
    dev = V.device
    assert 1 <= k <= n, f'invalid k={k} for n={n}'
    if strategy == 'coverage':
        anchors = gc.gapcover_select(V, Z, w, k, 'soft_gap', TAU, 'lazy')
        return {'anchors': anchors.to(dev).long(), 'start': None}
    if strategy == 'random':
        g = torch.Generator(device='cpu').manual_seed(_stable_seed(base_seed, doc_id))
        anchors = torch.randperm(n, generator=g)[:k].to(dev).long()
        return {'anchors': anchors, 'start': None}
    if strategy == 'kcenter':
        g = torch.Generator(device='cpu').manual_seed(_stable_seed(base_seed, doc_id))
        start = int(torch.randint(0, n, (1,), generator=g).item())
        Vn = V.float()
        sel = [start]
        d = 1.0 - (Vn @ Vn[start]).clamp(-1, 1)
        for _ in range(k - 1):
            nxt = int(torch.argmax(d).item())
            if nxt in sel:
                remaining = [i for i in range(n) if i not in sel]
                sel.extend(remaining[:k - len(sel)])
                break
            sel.append(nxt)
            d = torch.minimum(d, 1.0 - (Vn @ Vn[nxt]).clamp(-1, 1))
        anchors = torch.tensor(sel[:k], device=dev, dtype=torch.long)
        return {'anchors': anchors, 'start': start}
    raise ValueError(f'unknown anchor_strategy={strategy}')

def build_clusters(V: torch.Tensor, Z: torch.Tensor, w: torch.Tensor, k: int, anchor_strategy: str, base_seed: int=42, doc_id: int=0) -> Dict:
    V = V.float()
    A = gc.response_matrix(V, Z)
    info = select_anchors(V, Z, w, k, anchor_strategy, base_seed, doc_id)
    anchors = info['anchors']
    assert anchors.numel() == k and anchors.unique().numel() == k, 'anchors must be k distinct indices'
    assign = sf.assign_clusters(V, A, w, anchors, mode='embedding_cosine')
    sizes = torch.bincount(assign, minlength=k)
    assert int(sizes.min()) >= 1, 'empty cluster produced (should be impossible with anchor ownership)'
    bw, q, ac = mm.base_weights(V, A, w, anchors, assign)
    Fmat = mm.patch_features(V, A, w, anchors, assign, q, ac)
    return {'V': V, 'A': A, 'anchors': anchors, 'assign': assign, 'bw': bw, 'F': Fmat, 'k': k, 'start': info['start'], 'sizes': sizes}

def weights_for(cache: Dict, representative_strategy: str, mlp: Optional[mm.WeightMLP]=None, norm_stats: Optional[Dict]=None):
    V, assign, anchors, bw = (cache['V'], cache['assign'], cache['anchors'], cache['bw'])
    n = V.shape[0]
    dev = V.device
    if representative_strategy == 'anchor':
        weight = torch.zeros(n, device=dev)
        weight[anchors] = 1.0
        return (weight, None)
    if representative_strategy == 'uniform_centroid':
        return (torch.ones(n, device=dev), None)
    if representative_strategy == 'response_centroid':
        return (bw, None)
    if representative_strategy == 'learned':
        if mlp is None:
            raise RuntimeError('representative_strategy=learned requires an MLP')
        if norm_stats is None:
            raise RuntimeError('representative_strategy=learned requires feature-normalization stats')
        Fn = (cache['F'] - norm_stats['mean'].to(dev)) / norm_stats['std'].to(dev).clamp_min(1e-06)
        h = mlp(Fn)
        return (bw * torch.exp(h), h)
    raise ValueError(f'unknown representative_strategy={representative_strategy}')

def build_reps(cache: Dict, representative_strategy: str, mlp: Optional[mm.WeightMLP]=None, norm_stats: Optional[Dict]=None):
    weight, h = weights_for(cache, representative_strategy, mlp, norm_stats)
    R, alpha = mm.reps_from_weights(cache['V'], cache['assign'], cache['k'], weight)
    assert R.shape[0] == cache['k'], 'must return exactly k representatives'
    return (R, {'alpha': alpha, 'h': h, 'h_abs': float(h.abs().mean().detach()) if h is not None else 0.0})
