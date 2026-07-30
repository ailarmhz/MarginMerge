import sys, os, json, math
import numpy as np, torch
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
BASE = os.environ.get('CATTS_RES_INHARNESS', 'outputs/baselines_inharness')
CENT = os.environ.get('SAP_OUT', f'{BASE}/sap_centrality')
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3
MEM = [0.2, 0.1, 0.05]
NSEED = 3
RENDERED = {'arxivqa', 'docvqa', 'infovqa', 'tatdqa', 'tabfquad'}

def run(name):
    cf = f'{CENT}/{name}.pt'
    if not os.path.exists(cf):
        return None
    cent = torch.load(cf)
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    qs = torch.load(f'{CACHE}/{name}/qs.pt')
    ps = torch.load(f'{CACHE}/{name}/ps.pt')
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    test_docs = [docs[i] for i in perm[:nte]]
    test_docs = [d for d in test_docs if d in cent]
    if not test_docs:
        return None
    col = {d: j for j, d in enumerate(test_docs)}
    mD = len(test_docs)
    eval_q = [qi for qi, g in enumerate(gold) if g in col]
    gcol = torch.tensor([col[gold[qi]] for qi in eval_q])
    eq = [qs[qi].to(torch.bfloat16) for qi in eval_q]
    Ne = len(eq)
    Lmax = max((q.shape[0] for q in eq))
    Dd = eq[0].shape[1]
    Qpad = torch.zeros(Ne, Lmax, Dd, dtype=torch.bfloat16, device=DEV)
    Qm = torch.zeros(Ne, Lmax, device=DEV)
    for r, q in enumerate(eq):
        Qpad[r, :q.shape[0]] = q.to(DEV)
        Qm[r, :q.shape[0]] = 1.0
    Pg = {d: ps[d].to(DEV).to(torch.bfloat16) for d in test_docs}

    def ndcg(keep_fn):
        S = torch.empty(Ne, mD, device=DEV)
        for d in test_docs:
            Pp = Pg[d][keep_fn(d)]
            s_ = (Qpad @ Pp.T).max(dim=2)[0]
            S[:, col[d]] = (s_.float() * Qm).sum(dim=1)
        order = torch.argsort(S.cpu(), dim=1, descending=True)
        v = []
        for i in range(Ne):
            pos = (order[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1
            v.append(1.0 / math.log2(pos + 1) if pos <= 5 else 0.0)
        return float(np.mean(v))

    def kf(rho, S):
        return max(1, int(round(rho * S)))
    out = {'slice': name, 'n_test_docs': mD, 'n_eval_q': Ne, 'sap': {}, 'random': {}}
    for rho in MEM:

        def sap_keep(d, rho=rho):
            c = cent[d].to(DEV)
            S = Pg[d].shape[0]
            if c.shape[0] != S:
                c = c[:S] if c.shape[0] > S else torch.cat([c, torch.zeros(S - c.shape[0], device=DEV)])
            return torch.topk(c, kf(rho, S)).indices.sort().values
        out['sap'][str(rho)] = round(ndcg(sap_keep), 4)
        rv = []
        for seed in range(NSEED):
            g = torch.Generator(device=DEV).manual_seed(seed * 100003)

            def rk(d, g=g, rho=rho):
                S = Pg[d].shape[0]
                return torch.randperm(S, generator=g, device=DEV)[:kf(rho, S)].sort().values
            rv.append(ndcg(rk))
        out['random'][str(rho)] = {'mean': round(float(np.mean(rv)), 4), 'std': round(float(np.std(rv)), 4)}
        beat = out['sap'][str(rho)] > out['random'][str(rho)]['mean']
        print(f"[{name}] mem={rho} SAP={out['sap'][str(rho)]:.4f} random={out['random'][str(rho)]['mean']:.4f} {('SAP>rand' if beat else 'SAP<=rand')}", flush=True)
    return out

def main():
    slices = sys.argv[1:]
    res = {}
    for s in slices:
        r = run(s)
        if r:
            res[s] = r
        else:
            print(f'[{s}] no centrality yet, skip', flush=True)
    json.dump(res, open(f'{BASE}/sap_eval.json', 'w'), indent=2)
    print('\n=== SAP VALIDATION (rendered, beats random @10%/5%?) ===', flush=True)
    for s, r in res.items():
        if s not in RENDERED:
            continue
        for rho in ['0.1', '0.05']:
            b = r['sap'][rho] > r['random'][rho]['mean']
            print(f"  {s:<10} @{rho}: {('BEATS' if b else 'loses')} random (SAP {r['sap'][rho]} vs {r['random'][rho]['mean']})", flush=True)
    print('SAP_EVAL DONE', flush=True)
if __name__ == '__main__':
    main()
