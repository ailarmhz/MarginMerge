import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch
torch.set_num_threads(1)
import factorial as fac
import marginmerge as mm
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

def _n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-08)

def bank(M=16, d=8):
    Z = _n(torch.randn(M, d)).to(DEV)
    w = torch.rand(M).to(DEV)
    return (Z, w / w.sum())

def doc(n=50, d=8):
    return _n(torch.randn(n, d)).to(DEV)

def stats(F):
    return {'mean': F.mean(0), 'std': F.std(0).clamp_min(1e-06)}

def test_anchor_strategies():
    V = doc(50, 8)
    Z, w = bank()
    k = 8
    for strat in fac.ANCHOR_STRATEGIES:
        c = fac.build_clusters(V, Z, w, k, strat, base_seed=42, doc_id=3)
        assert c['anchors'].numel() == k and c['anchors'].unique().numel() == k, strat
        assert int(c['sizes'].min()) >= 1, f'{strat} empty cluster'
        assert int(c['assign'].max()) < k and int(c['assign'].min()) >= 0
        for ci, a in enumerate(c['anchors'].tolist()):
            assert int(c['assign'][a]) == ci, f'{strat} anchor owns'
    print('PASS anchor strategies: k distinct anchors, non-empty clusters, anchor ownership (1)')

def test_determinism():
    V = doc(50, 8)
    Z, w = bank()
    k = 8
    for strat in ('random', 'kcenter', 'coverage'):
        a1 = fac.build_clusters(V, Z, w, k, strat, 42, 7)['anchors']
        a2 = fac.build_clusters(V, Z, w, k, strat, 42, 7)['anchors']
        assert torch.equal(a1, a2), f'{strat} not deterministic'
    r1 = fac.build_clusters(V, Z, w, k, 'random', 42, 7)['anchors']
    r2 = fac.build_clusters(V, Z, w, k, 'random', 42, 8)['anchors']
    assert not torch.equal(r1, r2), 'random should differ across doc_id'
    print('PASS determinism: same (seed,doc)->same anchors; random varies across doc (2)')

def test_representatives_shape_norm():
    V = doc(50, 8)
    Z, w = bank()
    k = 8
    c = fac.build_clusters(V, Z, w, k, 'coverage', 42, 0)
    mlp = mm.WeightMLP().to(DEV)
    ns = stats(c['F'])
    for rep in fac.REPRESENTATIVE_STRATEGIES:
        R, dg = fac.build_reps(c, rep, mlp if rep == 'learned' else None, ns if rep == 'learned' else None)
        assert R.shape == (k, 8), rep
        assert torch.allclose(R.norm(dim=1), torch.ones(k, device=DEV), atol=0.0001), f'{rep} not normalized'
    print('PASS representatives: exactly k, L2-normalized, all four strategies (3)')

def test_anchor_rep_equals_retained_vector():
    V = doc(40, 8)
    Z, w = bank()
    k = 6
    c = fac.build_clusters(V, Z, w, k, 'kcenter', 42, 1)
    R, _ = fac.build_reps(c, 'anchor')
    for ci, a in enumerate(c['anchors'].tolist()):
        assert torch.allclose(R[ci], V[a], atol=0.0001), ci
    print('PASS representative=anchor reproduces retained anchor vectors (4)')

def test_learned_zero_equals_response_centroid():
    V = doc(50, 8)
    Z, w = bank()
    k = 7
    c = fac.build_clusters(V, Z, w, k, 'coverage', 42, 0)
    mlp = mm.WeightMLP().to(DEV)
    with torch.no_grad():
        mlp.net[-1].weight.zero_()
        mlp.net[-1].bias.zero_()
    ns = stats(c['F'])
    R_learned, _ = fac.build_reps(c, 'learned', mlp, ns)
    R_resp, _ = fac.build_reps(c, 'response_centroid')
    assert torch.allclose(R_learned, R_resp, atol=1e-05), float((R_learned - R_resp).abs().max())
    print('PASS learned(h=0) == response_centroid exactly (5)')

def test_uniform_differs_from_response():
    V = doc(60, 8)
    Z, w = bank()
    k = 8
    c = fac.build_clusters(V, Z, w, k, 'coverage', 42, 0)
    Ru, _ = fac.build_reps(c, 'uniform_centroid')
    Rr, _ = fac.build_reps(c, 'response_centroid')
    assert (Ru - Rr).abs().max() > 0.0001, 'uniform and response centroids should differ'
    print('PASS uniform_centroid != response_centroid (6)')

def test_grad_flows_learned_only():
    V = doc(40, 8).requires_grad_(True)
    Z, w = bank()
    k = 6
    Vd = V.detach()
    c = fac.build_clusters(Vd, Z, w, k, 'coverage', 42, 0)
    mlp = mm.WeightMLP().to(DEV)
    ns = stats(c['F'])
    R, _ = fac.build_reps(c, 'learned', mlp, ns)
    q = _n(torch.randn(3, 8)).to(DEV)
    (q @ R.T).max(1).values.sum().backward()
    assert V.grad is None
    gn = sum((float(p.grad.abs().sum()) for p in mlp.parameters() if p.grad is not None))
    assert gn > 0
    print(f'PASS grad flows to MLP only (embeddings frozen), grad-norm={gn:.4f} (7)')
if __name__ == '__main__':
    test_anchor_strategies()
    test_determinism()
    test_representatives_shape_norm()
    test_anchor_rep_equals_retained_vector()
    test_learned_zero_equals_response_centroid()
    test_uniform_differs_from_response()
    test_grad_flows_learned_only()
    print('\nALL FACTORIAL CORE TESTS PASSED')
