import os, json, math, time, argparse, csv
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import gapcover as gc, marginmerge as mm
from baselines_memory import merge_tome
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
MAN = os.environ.get('CATTS_MAN', 'outputs/supportfit/smoke_manifest.json')
OUT = os.environ.get('CATTS_MM_OUT', 'outputs/marginmerge')
os.makedirs(OUT, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
RHO = float(os.environ.get('MM_RHO', '0.05'))
DOC_TEST_FRAC = 0.3

def table9_manifests(slices):
    out = {}
    for s_ in slices:
        m = json.load(open(f'{CACHE}/{s_}/meta.json'))
        gold = m['gold']
        docs = sorted(set(gold))
        perm = np.random.RandomState(0).permutation(len(docs))
        nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
        td = [docs[i] for i in perm[:nte]]
        ds = set(td)
        out[s_] = {'corpus_docs': td, 'eval_queries': [qi for qi, g in enumerate(gold) if g in ds], 'n_corpus': len(td)}
    return out

def manifests(enlarge_tabfquad=True):
    mans = json.load(open(MAN))
    if enlarge_tabfquad:
        m = json.load(open(f'{CACHE}/tabfquad/meta.json'))
        gold = m['gold']
        allc = sorted(set(gold))
        mans['tabfquad'] = {**mans['tabfquad'], 'corpus_docs': allc, 'n_corpus': len(allc)}
    return mans

def run_slice(name, man, Z, w, mlp, ns):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    qs = torch.load(f'{CACHE}/{name}/qs.pt')
    ps = torch.load(f'{CACHE}/{name}/ps.pt')
    docs = man['corpus_docs']
    eq_ids = man['eval_queries']
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
        return min(S, max(1, math.ceil(RHO * S)))

    def smat(vecs):
        S = torch.empty(Ne, mD, device=DEV)
        for d in docs:
            X = vecs[d].to(torch.bfloat16)
            s_ = (Qpad @ X.T).max(dim=2)[0]
            S[:, col[d]] = (s_.float() * Qm).sum(dim=1)
        return S

    def ndcg(S):
        o = torch.argsort(S.cpu(), dim=1, descending=True)
        v = []
        for i in range(Ne):
            p = (o[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1
            v.append(1.0 / math.log2(p + 1) if p <= 5 else 0.0)
        return float(np.mean(v))
    S_full = smat({d: Pg[d] for d in docs})
    nd_full = ndcg(S_full)
    gs_f = S_full[torch.arange(Ne), gcol.to(DEV)]
    mask = torch.ones(Ne, mD, dtype=torch.bool, device=DEV)
    mask[torch.arange(Ne), gcol.to(DEV)] = False

    def diag(S):
        gs_c = S[torch.arange(Ne), gcol.to(DEV)]
        mf = gs_f.unsqueeze(1) - S_full
        mc = gs_c.unsqueeze(1) - S
        flips = ((mf > 0) != (mc > 0)) & mask
        return {'flip_rate': float(flips.float().sum() / mask.float().sum()), 'pos_score_err': float((gs_f - gs_c).abs().mean()), 'neg_score_err': float((S - S_full).abs()[mask].mean()), 'margin_err': float((mc - mf).abs()[mask].mean()), 'pairs_reversed': float(((mf > 0) & (mc <= 0)).float().sum() / mask.float().sum()), 'mean_margin_comp': float(mc[mask].mean()), 'mean_margin_full': float(mf[mask].mean())}
    print(f'[{name}] corpus={mD} q={Ne} FULL={nd_full:.4f}', flush=True)
    rows = []
    for meth in ['merging', 'marginmerge'] if os.environ.get('MM_ONLY') else ['merging', 'gapcover_pruning', 'response_centroid', 'marginmerge']:
        vecs = {}
        t_anchor = 0.0
        t0 = time.time()
        hs = []
        for d in docs:
            P = Pg[d]
            n = P.shape[0]
            k = kf(n)
            V = P.float()
            if meth == 'merging':
                vecs[d] = merge_tome(P, RHO)
            elif meth == 'gapcover_pruning':
                vecs[d] = P[gc.gapcover_select(V, Z, w, k, 'soft_gap', 0.05, 'lazy')]
            elif meth == 'response_centroid':
                R, _ = mm.response_centroid(V, Z, w, k)
                vecs[d] = R
            elif meth == 'marginmerge':
                ta = time.time()
                _, d0 = mm.marginmerge_reps(V, Z, w, k, None)
                t_anchor += time.time() - ta
                R, dg = mm.marginmerge_reps(V, Z, w, k, mlp, ns, cache=d0['cache'])
                vecs[d] = R
                hs.append(dg['h_abs'])
        ms = (time.time() - t0) * 1000.0 / mD
        ms_excl = ms - (t_anchor * 1000.0 / mD if meth == 'marginmerge' else 0)
        S = smat(vecs)
        nd = ndcg(S)
        dg = diag(S)
        row = dict(dataset=name, method=meth, ndcg_at_5=round(nd, 4), full_ndcg=round(nd_full, 4), retention=round(nd / nd_full, 4), ms_per_doc=round(ms, 1), ms_per_doc_excl_gapcover=round(ms_excl, 1), bytes=int(sum((v.numel() * 2 for v in vecs.values()))), mlp_h_abs=round(float(np.mean(hs)), 4) if hs else 0.0, **{k_: round(v_, 5) for k_, v_ in dg.items()})
        rows.append(row)
        print(f"[{name}] {meth:20s} nDCG@5={nd:.4f} ret={row['retention']:.3f} flip={dg['flip_rate']:.4f} ms/doc={ms:.0f} (excl-gc {ms_excl:.0f})", flush=True)
    rows.append(dict(dataset=name, method='full', ndcg_at_5=round(nd_full, 4), full_ndcg=round(nd_full, 4), retention=1.0, ms_per_doc=0.0, ms_per_doc_excl_gapcover=0.0, flip_rate=0.0, bytes=int(sum((Pg[d].numel() * 2 for d in docs)))))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--slices', nargs='+', default=['tabfquad', 'flickr', 'arxivqa'])
    ap.add_argument('--out', default=f'{OUT}/marginmerge_eval.csv')
    ap.add_argument('--table9', action='store_true', help='EXACT Table-9 protocol: held-out corpus + ALL queries')
    a = ap.parse_args()
    b = torch.load(a.bank)
    Z = b['prototypes'].to(DEV).float()
    f = b['raw_frequencies'].float()
    raw = (f + 1e-08) ** 0.5
    w = (raw / raw.sum()).to(DEV)
    ck = torch.load(a.ckpt)
    mlp = mm.WeightMLP().to(DEV)
    mlp.load_state_dict(ck['state_dict'])
    mlp.eval()
    ns = {k_: v_.to(DEV) for k_, v_ in ck['norm_stats'].items()}
    print(f"[ckpt] variant={ck['variant']} seed={ck['seed']} best_ep={ck['best_epoch']} val_nDCG5={ck['val_ndcg5']:.4f} val_flip={ck['val_flip']:.4f} params={ck['n_params']}", flush=True)
    import build_query_prototypes as bqp
    mans = table9_manifests(a.slices) if a.table9 else manifests(True)
    rows = []
    trainq = {s: set(ck['splits'][s]['train_q']) | set(ck['splits'][s]['val_q']) for s in ck['splits']}
    for s in a.slices:
        man = mans[s]
        if s not in trainq:
            print(f'[{s}] LOO-TARGET: slice absent from checkpoint splits -> zero training/validation queries', flush=True)
        else:
            assert not set(man['eval_queries']) & trainq[s], f'LEAKAGE: eval query used in training ({s})'
        bqp.validate_no_leakage(a.bank, [f'{s}:{qi}' for qi in man['eval_queries']])
        print(f"[{s}] leakage PASSED: {len(man['eval_queries'])} eval queries absent from train/val and bank", flush=True)
        rows += run_slice(s, man, Z, w, mlp, ns)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    print(f'\nsaved {a.out}')
    print('MM_EVAL DONE')
if __name__ == '__main__':
    main()
