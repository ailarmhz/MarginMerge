import sys, os, json, math
import torch, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
RES = os.environ.get('CATTS_RES', 'outputs/baselines_memory')
CKPT = f'{RES}/ckpt'
os.makedirs(CKPT, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3
KNN = 8
MEM = [1.0, 0.5, 0.2, 0.1, 0.05]
NSEED = 3
COUNT_METHODS = ['random', 'merge_tome', 'importance', 'top_likelihood', 'kcenter']

def load(s):
    m = json.load(open(f'{CACHE}/{s}/meta.json'))
    return (m, torch.load(f'{CACHE}/{s}/qs.pt'), torch.load(f'{CACHE}/{s}/ps.pt'))

@torch.no_grad()
def feats_label(P, qtoks):
    S = P.shape[0]
    G = (P @ P.T).float()
    eye = torch.eye(S, device=DEV, dtype=torch.bool)
    c = P.float().mean(0)
    c = c / (c.norm() + 1e-08)
    cos_c = P.float() @ c
    Gm = G.masked_fill(eye, float('-inf'))
    mx = Gm.max(dim=1)[0]
    mn = (G.sum(1) - 1.0) / max(S - 1, 1)
    kk = min(KNN, S - 1)
    knn = torch.topk(Gm, kk, dim=1)[0].mean(dim=1) if kk > 0 else torch.zeros(S, device=DEV)
    pos = torch.arange(S, device=DEV).float() / max(S - 1, 1)
    feats = torch.stack([cos_c, mn, mx, knn, pos, 1 - pos], dim=1)
    U = torch.zeros(S, device=DEV)
    if qtoks.shape[0] > 0:
        U[torch.unique((qtoks @ P.T).argmax(dim=1))] = 1.0
    return (feats, U)

def kf(rho, S):
    return max(1, int(round(rho * S)))

def fps(P, pool_idx, k, start):
    pool = P[pool_idx].float()
    M = pool.shape[0]
    if k >= M:
        return pool_idx
    sel = [start]
    d = 1.0 - (pool @ pool[start]).clamp(-1, 1)
    for _ in range(k - 1):
        nxt = int(torch.argmax(d))
        sel.append(nxt)
        d = torch.minimum(d, 1.0 - (pool @ pool[nxt]).clamp(-1, 1))
    return pool_idx[torch.tensor(sel, device=DEV)]

def _tome_round(cur, remove):
    n = cur.shape[0]
    ia = torch.arange(0, n, 2, device=DEV)
    ib = torch.arange(1, n, 2, device=DEV)
    A = cur[ia].float()
    B = cur[ib].float()
    nb = B.shape[0]
    sim = A @ B.T
    mx, mb = sim.max(dim=1)
    order = mx.argsort(descending=True)
    remove = min(remove, A.shape[0])
    merge_a = order[:remove]
    keep_a = order[remove:]
    reps = B.clone()
    cnt = torch.ones(nb, device=DEV)
    reps.index_add_(0, mb[merge_a], A[merge_a])
    cnt.index_add_(0, mb[merge_a], torch.ones(remove, device=DEV))
    reps = reps / cnt.unsqueeze(1)
    out = torch.cat([A[keep_a], reps], 0)
    return out / out.norm(dim=1, keepdim=True).clamp_min(1e-08)

def merge_tome(P, rho):
    target = kf(rho, P.shape[0])
    cur = P.float()
    while cur.shape[0] > target:
        remove = min(cur.shape[0] - target, cur.shape[0] // 2)
        if remove <= 0:
            break
        cur = _tome_round(cur, remove)
    return cur.to(torch.bfloat16)

def quant_int8(P):
    scale = P.abs().amax(dim=1, keepdim=True).clamp_min(1e-08) / 127.0
    q = torch.round(P / scale).clamp(-127, 127)
    return (q * scale).to(torch.bfloat16)

def run(name):
    m, qs, ps = load(name)
    gold = m['gold']
    bydoc = {}
    for qi, g in enumerate(gold):
        bydoc.setdefault(g, []).append(qi)
    docs = sorted(bydoc.keys())
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    test_docs = [docs[i] for i in perm[:nte]]
    train_set = set((docs[i] for i in perm[nte:]))
    feat = {}
    U = {}
    Xtr, ytr = ([], [])
    for d in docs:
        P = ps[d].to(DEV).to(torch.bfloat16)
        qtoks = torch.cat([qs[qi].to(DEV).to(torch.bfloat16) for qi in bydoc[d]], 0)
        f, u = feats_label(P, qtoks)
        feat[d] = (P, f, u)
        if d in train_set:
            Xtr.append(f.cpu().numpy())
            ytr.append(u.cpu().numpy())
    Xtr = np.concatenate(Xtr)
    ytr = np.concatenate(ytr)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, class_weight='balanced').fit(scaler.transform(Xtr), ytr)
    proba = {}
    for d in test_docs:
        P, f, u = feat[d]
        U[d] = u
        proba[d] = torch.tensor(clf.predict_proba(scaler.transform(f.cpu().numpy()))[:, 1], device=DEV)
    col = {d: j for j, d in enumerate(test_docs)}
    mD = len(test_docs)
    eval_q = [qi for d in test_docs for qi in bydoc[d]]
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

    def ndcg(vec_fn):
        S = torch.empty(Ne, mD, device=DEV)
        for d in test_docs:
            Pp = vec_fn(d)
            s_ = (Qpad @ Pp.T).max(dim=2)[0]
            S[:, col[d]] = (s_.float() * Qm).sum(dim=1)
        order = torch.argsort(S.cpu(), dim=1, descending=True)
        v = []
        for i in range(Ne):
            pos = (order[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1
            v.append(1.0 / math.log2(pos + 1) if pos <= 5 else 0.0)
        return float(np.mean(v))

    def keep_idx(method, d, rho, seed):
        P, f, u = feat[d]
        n = P.shape[0]
        k = kf(rho, n)
        if method == 'random':
            g = torch.Generator(device=DEV).manual_seed(seed * 100003 + d)
            return torch.randperm(n, generator=g, device=DEV)[:k].sort().values
        if method == 'importance':
            return torch.topk(f[:, 1], k).indices.sort().values
        if method == 'top_likelihood':
            return torch.topk(proba[d], k).indices.sort().values
        if method == 'kcenter':
            M = min(n, max(3 * k, k))
            pool = torch.topk(proba[d], M).indices
            start = 0 if seed == 0 else int(seed * 2654435761 % M)
            return fps(P, pool, k, start)
        raise ValueError(method)
    out = {'slice': name, 'n_test_docs': mD, 'n_eval_q': Ne, 'mem': MEM, 'count': {}, 'byte': {}, 'coverage': {}}
    out['count']['full'] = {'1.0': {'ndcg': round(ndcg(lambda d: feat[d][0]), 4), 'hc': 1.0}}
    for meth in COUNT_METHODS:
        out['count'][meth] = {}
        out['coverage'][meth] = {}
        for rho in MEM:
            if rho == 1.0:
                out['count'][meth]['1.0'] = {'ndcg': out['count']['full']['1.0']['ndcg'], 'hc': 1.0}
                continue
            nd, hc = ([], [])
            for seed in range(NSEED):
                if meth == 'merge_tome':
                    val = ndcg(lambda d, rho=rho: merge_tome(feat[d][0], rho))
                    nd.append(val)
                    if seed == 0:
                        hc.append(float('nan'))
                else:

                    def vf(d, meth=meth, rho=rho, seed=seed):
                        return feat[d][0][keep_idx(meth, d, rho, seed)]
                    nd.append(ndcg(vf))
                    hcs = []
                    for d in test_docs:
                        u = U[d]
                        idx = keep_idx(meth, d, rho, seed)
                        denom = u.sum().item()
                        if denom > 0:
                            hcs.append(u[idx].sum().item() / denom)
                    hc.append(float(np.mean(hcs)) if hcs else float('nan'))
            out['count'][meth][str(rho)] = {'ndcg': round(float(np.mean(nd)), 4), 'ndcg_std': round(float(np.std(nd)), 4)}
            if meth != 'merge_tome':
                out['coverage'][meth][str(rho)] = round(float(np.nanmean(hc)), 4)
            print(f'[{name}] {meth:14s} mem={rho} ndcg={np.mean(nd):.4f}±{np.std(nd):.4f}', flush=True)
    out['byte']['int8_full'] = {'mem_bytes': 0.5, 'ndcg': round(ndcg(lambda d: quant_int8(feat[d][0].float())), 4)}

    def int8_kc(d, rho=0.2):
        P = feat[d][0]
        idx = keep_idx('kcenter', d, rho, 0)
        return quant_int8(P[idx].float())
    out['byte']['int8_kcenter0.2'] = {'mem_bytes': 0.1, 'ndcg': round(ndcg(int8_kc), 4)}
    print(f"[{name}] int8_full(50%bytes) ndcg={out['byte']['int8_full']['ndcg']}  int8+kcenter0.2(10%bytes) ndcg={out['byte']['int8_kcenter0.2']['ndcg']}", flush=True)
    json.dump(out, open(f'{CKPT}/{name}.json', 'w'), indent=2)
    print(f'[{name}] CHECKPOINTED', flush=True)

def main():
    for s in sys.argv[1:]:
        if os.path.exists(f'{CKPT}/{s}.json'):
            print(f'[{s}] ckpt exists, skip', flush=True)
            continue
        try:
            run(s)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'[{s}] ERROR {e}', flush=True)
    print('BASELINES DONE', flush=True)
if __name__ == '__main__':
    main()
