import math, torch
import gapcover as gc
torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def _norm(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-08)

def rand_bank(M=16, d=8):
    Z = _norm(torch.randn(M, d))
    w = torch.rand(M)
    w = w / w.sum()
    return (Z.to(DEV), w.to(DEV))

def rand_doc(n=40, d=8):
    return _norm(torch.randn(n, d)).to(DEV)

def test_basic():
    V = rand_doc(50, 8)
    Z, w = rand_bank(16, 8)
    for rho in [0.2, 0.1, 0.05]:
        k = gc.kof(rho, 50)
        assert k == min(50, max(1, math.ceil(rho * 50)))
        Vc = V.clone()
        sel = gc.gapcover_select(V, Z, w, k, backend='reference')
        assert sel.numel() == k and sel.unique().numel() == k
        assert int(sel.min()) >= 0 and int(sel.max()) < 50
        assert torch.equal(V, Vc)
        sel2 = gc.gapcover_select(V, Z, w, k, backend='reference')
        assert torch.equal(sel, sel2)
    A = gc.response_matrix(V, Z)
    Cf = gc.soft_gap_coverage(A, 0.05)
    s32 = gc.gapcover_select(V.float(), Z.float(), w, 10, backend='reference')
    s16 = gc.gapcover_select(V.half(), Z.half(), w, 10, backend='reference')
    o32 = gc.objective_value(Cf, w, s32)
    o16 = gc.objective_value(Cf, w, s16)
    assert abs(o32 - o16) < 0.001, (o32, o16)
    print('PASS test_basic')

def test_monotonic():
    V = rand_doc(30, 8)
    Z, w = rand_bank(16, 8)
    A = gc.response_matrix(V, Z)
    C = gc.soft_gap_coverage(A, 0.05)
    prev_obj, prev_reg = (-1, 1000000000.0)
    for k in range(1, 31):
        sel = gc.gapcover_select(V, Z, w, k, backend='reference')
        obj = gc.objective_value(C, w, sel)
        reg = gc.coverage_stats(A, w, sel, 0.05)['weighted_regret']
        assert obj >= prev_obj - 1e-06, (k, obj, prev_obj)
        assert reg <= prev_reg + 1e-06, (k, reg, prev_reg)
        prev_obj, prev_reg = (obj, reg)
    sel = gc.gapcover_select(V, Z, w, 30, backend='reference')
    st = gc.coverage_stats(A, w, sel, 0.05)
    assert st['soft_coverage_min'] > 1 - 0.0001
    assert abs(st['weighted_regret']) < 0.0001
    print('PASS test_monotonic')

def test_duplicate_modes():
    d = 8
    modes = _norm(torch.randn(3, d))
    Z = modes.clone().to(DEV)
    w = torch.ones(3).to(DEV) / 3
    patches = [modes[0]] * 5 + [modes[1], modes[2]]
    V = _norm(torch.stack(patches)).to(DEV)
    sel = set(gc.gapcover_select(V, Z, w, 3, tau=0.05, backend='reference').tolist())
    assert 5 in sel and 6 in sel, sel
    print('PASS test_duplicate_modes')

def test_geometric_outlier():
    d = 8
    modes = _norm(torch.randn(3, d))
    Z = modes.to(DEV)
    w = torch.ones(3).to(DEV) / 3
    outlier = _norm(torch.randn(1, d))
    while (outlier @ modes.T).abs().max() > 0.3:
        outlier = _norm(torch.randn(1, d))
    V = _norm(torch.cat([modes, outlier], 0)).to(DEV)
    sel = gc.gapcover_select(V, Z, w, 3, backend='reference').tolist()
    assert 3 not in sel, ('gapcover chose the useless outlier', sel)
    print('PASS test_geometric_outlier')

def test_near_argmax_substitute():
    d = 8
    m1 = _norm(torch.randn(1, d))[0]
    others = _norm(torch.randn(2, d))
    Z = _norm(torch.stack([m1, others[0], others[1]])).to(DEV)
    w = torch.ones(3).to(DEV) / 3
    near = _norm(m1 + 0.02 * torch.randn(d))
    V = _norm(torch.stack([m1, near, others[0], others[1]])).to(DEV)
    sel = gc.gapcover_select(V, Z, w, 3, backend='reference').tolist()
    assert 2 in sel and 3 in sel, sel
    assert not (0 in sel and 1 in sel), ('took both near-duplicates', sel)
    print('PASS test_near_argmax_substitute')

def test_lazy_vs_reference():
    for t in range(20):
        n = torch.randint(15, 60, (1,)).item()
        d = 8
        M = torch.randint(8, 24, (1,)).item()
        V = rand_doc(n, d)
        Z, w = rand_bank(M, d)
        k = max(1, n // 4)
        A = gc.response_matrix(V, Z)
        C = gc.soft_gap_coverage(A, 0.05)
        sr = gc.gapcover_select(V, Z, w, k, backend='reference')
        sl = gc.gapcover_select(V, Z, w, k, backend='lazy')
        assert sr.unique().numel() == k and sl.unique().numel() == k
        oref, olazy = (gc.objective_value(C, w, sr), gc.objective_value(C, w, sl))
        assert abs(oref - olazy) < 0.0001, (t, oref, olazy)
    print('PASS test_lazy_vs_reference')

def test_regret_objective():
    V = rand_doc(30, 8)
    Z, w = rand_bank(16, 8)
    sel = gc.gapcover_select(V, Z, w, 10, objective='direct_regret', backend='reference')
    assert sel.unique().numel() == 10
    full = gc.gapcover_select(V, Z, w, 30, objective='direct_regret', backend='reference')
    A = gc.response_matrix(V, Z)
    assert gc.coverage_stats(A, w, full, 0.05)['weighted_regret'] < 0.0001
    print('PASS test_regret_objective')

def test_leakage_validator():
    import build_query_prototypes as bqp, tempfile, os
    man = {'source_query_ids': ['s:1', 's:2', 's:3']}
    tmp = tempfile.mktemp(suffix='.pt')
    torch.save({'prototypes': torch.randn(4, 8), 'raw_frequencies': torch.ones(4), 'manifest': man}, tmp)
    assert bqp.validate_no_leakage(tmp, ['s:9', 's:10']) is True
    try:
        bqp.validate_no_leakage(tmp, ['s:2', 's:9'])
        raise AssertionError('leakage not detected')
    except RuntimeError as e:
        assert 'LEAKAGE' in str(e)
    os.remove(tmp)
    import inspect
    params = list(inspect.signature(gc.gapcover_select).parameters)
    assert 'query' not in params and 'q' not in params, params
    print('PASS test_leakage_validator')
if __name__ == '__main__':
    test_basic()
    test_monotonic()
    test_duplicate_modes()
    test_geometric_outlier()
    test_near_argmax_substitute()
    test_lazy_vs_reference()
    test_regret_objective()
    test_leakage_validator()
    print('\nALL GAPCOVER TESTS PASSED')
