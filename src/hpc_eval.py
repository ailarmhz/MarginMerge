import sys, os, json, math
import numpy as np, torch
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
OUT = os.environ.get('CATTS_RES_COMPLETE', 'outputs/baselines_complete')
os.makedirs(OUT, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3
MEM = [0.2, 0.1, 0.05]

def run(name):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    qs = torch.load(f'{CACHE}/{name}/qs.pt')
    ps = torch.load(f'{CACHE}/{name}/ps.pt')
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    test_docs = [docs[i] for i in perm[:nte]]
    col = {d: j for j, d in enumerate(test_docs)}
    mD = len(test_docs)
    eval_q = [qi for qi, g in enumerate(gold) if g in col]
    gcol = torch.tensor([col[gold[qi]] for qi in eval_q])
    Pg = {d: ps[d].to(DEV).to(torch.bfloat16) for d in test_docs}
    eq = [qs[qi].to(torch.bfloat16) for qi in eval_q]
    Ne = len(eq)
    Lmax = max((q.shape[0] for q in eq))
    Dd = eq[0].shape[1]
    Qpad = torch.zeros(Ne, Lmax, Dd, dtype=torch.bfloat16, device=DEV)
    Qm = torch.zeros(Ne, Lmax, device=DEV)
    for r, q in enumerate(eq):
        Qpad[r, :q.shape[0]] = q.to(DEV)
        Qm[r, :q.shape[0]] = 1.0
    Qmb = Qm.bool()

    def kf(rho, S):
        return max(1, int(round(rho * S)))

    def ndcg_at(rho):
        Sc = torch.empty(Ne, mD, device=DEV)
        for d in test_docs:
            P = Pg[d]
            Sp = P.shape[0]
            sim = Qpad @ P.T
            rel = sim.masked_fill(~Qmb.unsqueeze(2), -10000.0).max(1).values
            k = kf(rho, Sp)
            kth = rel.topk(k, dim=1).values[:, -1:]
            keep = rel >= kth
            simk = sim.masked_fill(~keep.unsqueeze(1), -10000.0)
            Sc[:, col[d]] = (simk.max(2).values.float() * Qm).sum(1)
        order = torch.argsort(Sc.cpu(), dim=1, descending=True)
        v = []
        for i in range(Ne):
            pos = (order[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1
            v.append(1.0 / math.log2(pos + 1) if pos <= 5 else 0.0)
        return round(float(np.mean(v)), 4)
    out = {'slice': name, 'n_test_docs': mD, 'n_eval_q': len(eval_q), 'hpc_qcond': {}}
    for rho in MEM:
        out['hpc_qcond'][str(rho)] = ndcg_at(rho)
        print(f"[{name}] HPC(query-cond) mem={rho} nDCG@5={out['hpc_qcond'][str(rho)]}", flush=True)
    return out

def main():
    res = {}
    for s in sys.argv[1:]:
        res[s] = run(s)
    json.dump(res, open(f'{OUT}/hpc_eval.json', 'w'), indent=2)
    print('HPC DONE', flush=True)
if __name__ == '__main__':
    main()
