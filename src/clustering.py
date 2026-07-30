import torch

def _norm(x: torch.Tensor, dim: int=-1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-08)

def assign_clusters(V: torch.Tensor, A: torch.Tensor, w: torch.Tensor, anchors: torch.Tensor, mode: str='embedding_cosine', eta: float=0.75) -> torch.Tensor:
    n = V.shape[0]
    sim_emb = V @ V[anchors].T
    if mode == 'embedding_cosine':
        S = sim_emb
    else:
        sig = A - A.mean(dim=0, keepdim=True)
        sig = sig * w.clamp(min=0).sqrt().unsqueeze(1)
        sig = _norm(sig, dim=0)
        sim_resp = sig.T @ sig[:, anchors]
        S = sim_resp if mode == 'response_cosine' else eta * sim_resp + (1.0 - eta) * sim_emb
    assign = S.argmax(dim=1)
    assign[anchors] = torch.arange(anchors.numel(), device=V.device)
    return assign
