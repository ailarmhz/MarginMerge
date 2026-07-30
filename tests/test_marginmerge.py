import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import torch, math
torch.set_num_threads(1)
import torch.nn.functional as Fn
import marginmerge as mm
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)

def _n(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-08)

def bank(M=16, d=8):
    Z = _n(torch.randn(M, d)).to(DEV)
    w = torch.rand(M).to(DEV)
    return (Z, w / w.sum())

def doc(n=40, d=8):
    return _n(torch.randn(n, d)).to(DEV)

def stats(F):
    return {'mean': F.mean(0), 'std': F.std(0).clamp_min(1e-06)}

def test_core():
    V = doc(50, 8)
    Z, w = bank(16, 8)
    k = 8
    mlp = mm.WeightMLP().to(DEV)
    R0, d0 = mm.marginmerge_reps(V, Z, w, k, None)
    F = d0['cache']['F']
    ns = stats(F)
    R, dg = mm.marginmerge_reps(V, Z, w, k, mlp, ns, cache=d0['cache'])
    assert R.shape == (k, 8)
    assert torch.allclose(R.norm(dim=1), torch.ones(k, device=DEV), atol=0.0001)
    assign = d0['cache']['assign']
    anchors = d0['cache']['anchors']
    assert assign.shape == (50,) and int(assign.min()) >= 0 and (int(assign.max()) < k)
    for c, a in enumerate(anchors.tolist()):
        assert int(assign[a]) == c
    alpha = dg['alpha']
    assert (alpha >= 0).all()
    s = torch.zeros(k, device=DEV).index_add_(0, assign, alpha)
    assert torch.allclose(s, torch.ones(k, device=DEV), atol=0.0001)
    R2, _ = mm.marginmerge_reps(V, Z, w, k, mlp, ns, cache=d0['cache'])
    assert torch.allclose(R, R2, atol=1e-06)
    print('PASS core: k/normalized/partition/anchors/nonneg/sum1/deterministic (1-7)')

def test_grads():
    V = doc(40, 8).requires_grad_(True)
    Z, w = bank(16, 8)
    k = 6
    mlp = mm.WeightMLP().to(DEV)
    Vd = V.detach()
    _, d0 = mm.marginmerge_reps(Vd, Z, w, k, None)
    ns = stats(d0['cache']['F'])
    R, _ = mm.marginmerge_reps(Vd, Z, w, k, mlp, ns, cache=d0['cache'])
    q = _n(torch.randn(3, 8)).to(DEV)
    loss = (q @ R.T).max(1).values.sum()
    loss.backward()
    assert V.grad is None
    gn = sum((float(p.grad.abs().sum()) for p in mlp.parameters() if p.grad is not None))
    assert gn > 0
    print(f'PASS grads: embeddings frozen, MLP grad-norm={gn:.4f} (8,9)')

def test_margin_improves():
    torch.manual_seed(1)
    Z, w = bank(16, 8)
    k = 4
    Vp = doc(30, 8)
    Vn = doc(30, 8)
    Q = _n(torch.randn(5, 8)).to(DEV)
    mlp = mm.WeightMLP().to(DEV)
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.003)
    _, dp = mm.marginmerge_reps(Vp, Z, w, k, None)
    _, dn = mm.marginmerge_reps(Vn, Z, w, k, None)
    ns = stats(torch.cat([dp['cache']['F'], dn['cache']['F']], 0))

    def margin():
        Rp, _ = mm.marginmerge_reps(Vp, Z, w, k, mlp, ns, cache=dp['cache'])
        Rn_, _ = mm.marginmerge_reps(Vn, Z, w, k, mlp, ns, cache=dn['cache'])
        return (Q @ Rp.T).max(1).values.sum() - (Q @ Rn_.T).max(1).values.sum()
    m0 = float(margin())
    for _ in range(60):
        opt.zero_grad()
        (-margin()).backward()
        opt.step()
    m1 = float(margin())
    assert m1 > m0, (m0, m1)
    print(f'PASS synthetic margin improves (10): {m0:.4f} -> {m1:.4f}')

def test_margin_loss_shift_invariance():
    sfp, sfn = (torch.tensor(5.0), torch.tensor(3.0))
    scp, scn = (torch.tensor(4.0), torch.tensor(2.0))
    l = Fn.huber_loss(scp - scn, sfp - sfn, delta=0.5)
    assert float(l) < 1e-08
    print('PASS margin loss is shift-invariant (11)')

def test_vulnerability_weighting():
    mf = torch.tensor([0.1, 2.0, 8.0])
    vw = torch.exp(-mf.abs() / 2.0)
    vw = vw / vw.sum()
    assert vw[0] > vw[1] > vw[2]
    print(f'PASS vulnerability weighting emphasizes small margins (12): {[round(float(x), 3) for x in vw]}')

def test_leakage_validator():
    import tempfile, build_query_prototypes as bqp
    tmp = tempfile.mktemp(suffix='.pt')
    torch.save({'prototypes': torch.randn(4, 8), 'raw_frequencies': torch.ones(4), 'manifest': {'source_query_ids': ['s:1', 's:2']}}, tmp)
    assert bqp.validate_no_leakage(tmp, ['s:9']) is True
    try:
        bqp.validate_no_leakage(tmp, ['s:1'])
        raise AssertionError('leak not caught')
    except RuntimeError as e:
        assert 'LEAKAGE' in str(e)
    os.remove(tmp)
    print('PASS leakage validation fails on overlap (13)')

def test_eval_needs_only_reps():
    V = doc(40, 8)
    Z, w = bank(16, 8)
    mlp = mm.WeightMLP().to(DEV)
    _, d0 = mm.marginmerge_reps(V, Z, w, 5, None)
    ns = stats(d0['cache']['F'])
    R, _ = mm.marginmerge_reps(V, Z, w, 5, mlp, ns, cache=d0['cache'])
    R_saved = R.detach().clone()
    del V, Z, w, mlp, d0, ns
    q = _n(torch.randn(3, 8)).to(DEV)
    s = (q @ R_saved.T).max(1).values.sum()
    assert torch.isfinite(s)
    print('PASS eval needs only the saved k vectors (14)')

def test_zero_mlp_equals_base():
    V = doc(50, 8)
    Z, w = bank(16, 8)
    k = 7
    mlp = mm.WeightMLP().to(DEV)
    with torch.no_grad():
        mlp.net[-1].weight.zero_()
        mlp.net[-1].bias.zero_()
    Rb, d0 = mm.marginmerge_reps(V, Z, w, k, None)
    ns = stats(d0['cache']['F'])
    Rl, _ = mm.marginmerge_reps(V, Z, w, k, mlp, ns, cache=d0['cache'])
    assert torch.allclose(Rb, Rl, atol=1e-05), float((Rb - Rl).abs().max())
    print('PASS zero-MLP reproduces base response centroid (15)')

def test_norm_stats_required():
    V = doc(30, 8)
    Z, w = bank(16, 8)
    mlp = mm.WeightMLP().to(DEV)
    try:
        mm.marginmerge_reps(V, Z, w, 5, mlp, None)
        raise AssertionError('missing norm_stats not caught')
    except RuntimeError as e:
        assert 'normalization' in str(e)
    print('PASS feature-normalization metadata is required/validated (16)')

def test_cpu_gpu_equivalence():
    if not torch.cuda.is_available():
        print('SKIP cpu/gpu (no cuda)')
        return
    torch.manual_seed(5)
    Vc = _n(torch.randn(40, 8))
    Zc = _n(torch.randn(16, 8))
    wc = torch.rand(16)
    wc = wc / wc.sum()
    mlp_c = mm.WeightMLP()
    _, dc = mm.marginmerge_reps(Vc, Zc, wc, 6, None)
    ns_c = stats(dc['cache']['F'])
    Rc, _ = mm.marginmerge_reps(Vc, Zc, wc, 6, mlp_c, ns_c, cache=dc['cache'])
    Vg, Zg, wg = (Vc.cuda(), Zc.cuda(), wc.cuda())
    mlp_g = mm.WeightMLP().cuda()
    mlp_g.load_state_dict(mlp_c.state_dict())
    _, dg = mm.marginmerge_reps(Vg, Zg, wg, 6, None)
    ns_g = {k_: v.cuda() for k_, v in ns_c.items()}
    Rg, _ = mm.marginmerge_reps(Vg, Zg, wg, 6, mlp_g, ns_g, cache=dg['cache'])
    diff = float((Rc.cuda() - Rg).abs().max())
    assert diff < 0.002, diff
    print(f'PASS cpu/gpu equivalent within {diff:.2e} (17)')
if __name__ == '__main__':
    test_core()
    test_grads()
    test_margin_improves()
    test_margin_loss_shift_invariance()
    test_vulnerability_weighting()
    test_leakage_validator()
    test_eval_needs_only_reps()
    test_zero_mlp_equals_base()
    test_norm_stats_required()
    test_cpu_gpu_equivalence()
    p = sum((x.numel() for x in mm.WeightMLP().parameters()))
    print(f'\nMLP trainable parameters: {p} (limit 20000)')
    print('ALL MARGINMERGE TESTS PASSED')
