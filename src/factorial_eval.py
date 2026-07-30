import os, json, math, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import marginmerge as mm
import factorial as fac
from baselines_memory import merge_tome
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
OUT = os.environ.get('FACT_OUT', 'outputs/factorial')
RUNS = f'{OUT}/runs'
os.makedirs(RUNS, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3

def table9_manifest(s):
    m = json.load(open(f'{CACHE}/{s}/meta.json'))
    gold = m['gold']
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    td = [docs[i] for i in perm[:nte]]
    ds = set(td)
    eq = [qi for qi, g in enumerate(gold) if g in ds]
    return (td, eq, gold)

def eval_slice(name, cfg, Z, w, mlp, ns):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    qs = torch.load(f'{CACHE}/{name}/qs.pt')
    ps = torch.load(f'{CACHE}/{name}/ps.pt')
    docs, eq_ids, _ = table9_manifest(name)
    col = {d: j for j, d in enumerate(docs)}
    mD = len(docs)
    gcol = torch.tensor([col[gold[qi]] for qi in eq_ids])
    eq = [qs[qi].to(torch.bfloat16) for qi in eq_ids]
    Ne = len(eq)
    Lmax = max((q.shape[0] for q in eq))
    Dd = eq[0].shape[1]
    Qpad = torch.zeros(Ne, Lmax, Dd, dtype=torch.bfloat16, device=DEV)
    Qm = torch.zeros(Ne, Lmax, device=DEV)
    for r, q in enumerate(eq):
        Qpad[r, :q.shape[0]] = q.to(DEV)
        Qm[r, :q.shape[0]] = 1.0
    Pg = {d: ps[d].to(DEV).to(torch.bfloat16) for d in docs}

    def kf(S):
        return min(S, max(1, math.ceil(cfg['rho'] * S)))

    def smat(vecs):
        S = torch.empty(Ne, mD, device=DEV)
        for d in docs:
            X = vecs[d].to(torch.bfloat16)
            s_ = (Qpad @ X.T).max(dim=2)[0]
            S[:, col[d]] = (s_.float() * Qm).sum(dim=1)
        return S
    S_full = smat({d: Pg[d] for d in docs})
    gs_f = S_full[torch.arange(Ne), gcol.to(DEV)]
    mask = torch.ones(Ne, mD, dtype=torch.bool, device=DEV)
    mask[torch.arange(Ne), gcol.to(DEV)] = False
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    vecs = {}
    t_build = 0.0
    for d in docs:
        P = Pg[d]
        n = P.shape[0]
        V = P.float()
        t0 = time.time()
        if cfg['kind'] == 'full':
            R = P
        elif cfg['kind'] == 'merging':
            R = merge_tome(P, cfg['rho'])
        else:
            k = kf(n)
            c = fac.build_clusters(V, Z, w, k, cfg['anchor'], base_seed=cfg['seed'], doc_id=d)
            R, _ = fac.build_reps(c, cfg['representative'], mlp, ns)
        torch.cuda.synchronize()
        t_build += time.time() - t0
        vecs[d] = R
    peak_mb = torch.cuda.max_memory_allocated() / 1000000.0
    ms_per_doc = t_build * 1000.0 / mD
    S = smat(vecs)
    order = torch.argsort(S.cpu(), dim=1, descending=True)
    ranks = torch.tensor([(order[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1 for i in range(Ne)])
    ndcg_q = torch.where(ranks <= 5, 1.0 / torch.log2(ranks.float() + 1), torch.zeros(Ne))
    r1 = (ranks <= 1).float()
    r5 = (ranks <= 5).float()
    r10 = (ranks <= 10).float()
    gs_c = S[torch.arange(Ne), gcol.to(DEV)]
    mf = gs_f.unsqueeze(1) - S_full
    mc = gs_c.unsqueeze(1) - S
    flips = ((mf > 0) != (mc > 0)) & mask
    reversals = (mf > 0) & (mc <= 0) & mask
    denom = mask.float().sum()
    metrics = dict(ndcg_at_5=float(ndcg_q.mean()), recall_at_1=float(r1.mean()), recall_at_5=float(r5.mean()), recall_at_10=float(r10.mean()), flip_rate=float(flips.float().sum() / denom), reversal_rate=float(reversals.float().sum() / denom), margin_abs_err=float((mc - mf).abs()[mask].mean()), score_abs_err=float((S - S_full).abs()[mask].mean()), pos_score_abs_err=float((gs_c - gs_f).abs().mean()), ms_per_doc=round(ms_per_doc, 3), peak_mem_mb=round(peak_mb, 1), n_corpus=mD, n_queries=Ne)
    perq = dict(ndcg_at_5=ndcg_q.tolist(), recall_at_1=r1.tolist(), recall_at_5=r5.tolist(), recall_at_10=r10.tolist(), gold_rank=ranks.tolist())
    return (metrics, perq)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchor', choices=['random', 'kcenter', 'coverage'])
    ap.add_argument('--representative', choices=['anchor', 'uniform_centroid', 'response_centroid', 'learned'])
    ap.add_argument('--loss', default='none')
    ap.add_argument('--reference', choices=['merging', 'full'])
    ap.add_argument('--bank')
    ap.add_argument('--ckpt')
    ap.add_argument('--slices', nargs='+', default=['arxivqa', 'flickr', 'tabfquad'])
    ap.add_argument('--rho', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    Z = w = mlp = ns = None
    if a.reference:
        cfg = {'kind': a.reference, 'rho': a.rho, 'seed': a.seed, 'anchor': '-', 'representative': a.reference, 'loss': '-'}
        label = a.reference
    else:
        assert a.anchor and a.representative, 'need --anchor and --representative (or --reference)'
        b = torch.load(a.bank)
        Z = b['prototypes'].to(DEV).float()
        f = b['raw_frequencies'].float()
        raw = (f + 1e-08) ** 0.5
        w = (raw / raw.sum()).to(DEV)
        if a.representative == 'learned':
            assert a.ckpt, 'learned needs --ckpt'
            ck = torch.load(a.ckpt)
            mlp = mm.WeightMLP().to(DEV)
            mlp.load_state_dict(ck['state_dict'])
            mlp.eval()
            ns = {k_: v_.to(DEV) for k_, v_ in ck['norm_stats'].items()}
            a.loss = ck['loss_strategy']
        cfg = {'kind': 'factorial', 'rho': a.rho, 'seed': a.seed, 'anchor': a.anchor, 'representative': a.representative, 'loss': a.loss}
        label = f'{a.anchor}+{a.representative}+{a.loss}'
    rows = {}
    for s in a.slices:
        met, perq = eval_slice(s, cfg, Z, w, mlp, ns)
        rid = f"{cfg['anchor']}__{cfg['representative']}__{cfg['loss']}__rho{int(a.rho * 100)}__seed{a.seed}__{s}"
        rec = {'run_id': rid, 'dataset': s, 'anchor_strategy': cfg['anchor'], 'representative_strategy': cfg['representative'], 'loss_strategy': cfg['loss'], 'rho': a.rho, 'seed': a.seed, 'kind': cfg['kind'], 'metrics': met, 'per_query': perq}
        json.dump(rec, open(f'{RUNS}/{rid}.json', 'w'))
        rows[s] = met
        print(f"[{label}] {s:9s} nDCG@5={met['ndcg_at_5']:.4f} R@1={met['recall_at_1']:.3f} flip={met['flip_rate']:.4f} rev={met['reversal_rate']:.4f} mErr={met['margin_abs_err']:.3f} ms/doc={met['ms_per_doc']:.1f} peakMB={met['peak_mem_mb']:.0f}", flush=True)
    print('FACTORIAL_EVAL DONE', flush=True)
if __name__ == '__main__':
    main()
