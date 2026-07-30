import os, json, math, time, argparse, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import torch.nn.functional as F
import marginmerge as mm
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
MAN = os.environ.get('CATTS_MAN', 'outputs/supportfit/smoke_manifest.json')
OUT = os.environ.get('CATTS_MM_OUT', 'outputs/marginmerge')
os.makedirs(OUT, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
RHO = 0.05
NNEG = 8
DOC_TEST_FRAC = 0.3
SLICES = ['tabfquad', 'arxivqa', 'flickr']

def maxsim(Q, X):
    return (Q @ X.T).max(dim=1).values.sum()

def build_splits(seed):
    mans = json.load(open(MAN))
    splits = {}
    for s in SLICES:
        m = json.load(open(f'{CACHE}/{s}/meta.json'))
        gold = m['gold']
        docs = sorted(set(gold))
        perm = np.random.RandomState(0).permutation(len(docs))
        nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
        eval_docs = set((docs[i] for i in perm[:nte]))
        train_docs = [docs[i] for i in perm[nte:]]
        tq = [qi for qi, g in enumerate(gold) if g not in eval_docs]
        rng = np.random.RandomState(seed)
        rng.shuffle(tq)
        nval = max(1, int(0.1 * len(tq)))
        splits[s] = {'train_q': sorted(tq[nval:]), 'val_q': sorted(tq[:nval]), 'train_docs': train_docs, 'eval_q': mans[s]['eval_queries']}
        A, B, C = (set(splits[s]['train_q']), set(splits[s]['val_q']), set(splits[s]['eval_q']))
        assert not A & B and (not A & C) and (not B & C), f'SPLIT OVERLAP in {s}'
    return splits

def mine_hard_negatives(splits, seed):
    neg = {}
    for s in SLICES:
        m = json.load(open(f'{CACHE}/{s}/meta.json'))
        gold = m['gold']
        qs = torch.load(f'{CACHE}/{s}/qs.pt')
        ps = torch.load(f'{CACHE}/{s}/ps.pt')
        tds = splits[s]['train_docs']
        Pg = {d: ps[d].to(DEV).to(torch.bfloat16) for d in tds}
        out = {}
        for qi in splits[s]['train_q'] + splits[s]['val_q']:
            Q = qs[qi].to(DEV).to(torch.bfloat16)
            sc = torch.tensor([float(maxsim(Q, Pg[d])) for d in tds], device=DEV)
            g = gold[qi]
            order = torch.argsort(sc, descending=True).tolist()
            negs = [tds[i] for i in order if tds[i] != g][:NNEG]
            out[qi] = {'pos': g, 'negs': negs, 'pos_score': float(maxsim(Q, Pg[g])), 'neg_scores': [float(sc[tds.index(d)]) for d in negs]}
        neg[s] = out
        print(f'[mine] {s}: {len(out)} queries x {NNEG} negatives (train corpus {len(tds)})', flush=True)
    return neg

def cache_clusters(splits, neg, Z, w):
    cch = {}
    for s in SLICES:
        ps = torch.load(f'{CACHE}/{s}/ps.pt')
        need = set()
        for qi, r in neg[s].items():
            need.add(r['pos'])
            need.update(r['negs'])
        t0 = time.time()
        cs = {}
        for d in sorted(need):
            V = ps[d].to(DEV).float()
            n = V.shape[0]
            k = min(n, max(1, math.ceil(RHO * n)))
            _, dg = mm.marginmerge_reps(V, Z, w, k, None)
            c = dg['cache']
            cs[d] = {'V': V, 'assign': c['assign'], 'bw': c['bw'], 'F': c['F'], 'anchors': c['anchors'], 'k': k}
        cch[s] = cs
        print(f'[cache] {s}: {len(cs)} docs in {time.time() - t0:.0f}s', flush=True)
    return cch

def reps(doc, mlp, ns):
    c = doc
    Fn_ = (c['F'] - ns['mean']) / ns['std'].clamp_min(1e-06)
    h = mlp(Fn_)
    weight = c['bw'] * torch.exp(h)
    R, alpha = mm.reps_from_weights(c['V'], c['assign'], c['k'], weight)
    return (R, alpha, h)

def losses(Q, dpos, dneg, mlp, ns, cfg):
    Rp, ap, hp = reps(dpos, mlp, ns)
    sp = maxsim(Q, Rp)
    sn, hs, als, asg = ([], [hp], [ap], [(dpos['assign'], dpos['k'], dpos['anchors'])])
    for dn in dneg:
        Rn, an, hn = reps(dn, mlp, ns)
        sn.append(maxsim(Q, Rn))
        hs.append(hn)
        als.append(an)
        asg.append((dn['assign'], dn['k'], dn['anchors']))
    sn = torch.stack(sn)
    mf = torch.tensor(cfg['full_pos'] - np.array(cfg['full_negs']), device=DEV, dtype=torch.float32)
    mc = sp - sn
    vw = torch.exp(-mf.abs() / 2.0)
    vw = vw / vw.sum().clamp_min(1e-08)
    if not cfg.get('use_vuln', True):
        vw = torch.full_like(vw, 1.0 / vw.numel())
    L_margin = (vw * F.huber_loss(mc, mf, delta=0.5, reduction='none')).sum()
    L_rank = torch.relu(0.2 - mc).mean()
    if cfg.get('use_list', True):
        pf = torch.softmax(torch.tensor([cfg['full_pos']] + list(cfg['full_negs']), device=DEV), 0)
        lpc = torch.log_softmax(torch.cat([sp.view(1), sn]), 0)
        L_list = F.kl_div(lpc, pf, reduction='sum')
    else:
        L_list = torch.zeros((), device=DEV)
    hcat = torch.cat(hs)
    L_weight = (hcat ** 2).mean()
    ents = [mm.cluster_entropy(a, g[0], g[1]) for a, g in zip(als, asg)]
    L_ent = torch.stack([torch.relu(0.2 - e).pow(2).mean() for e in ents]).mean()
    L_anc = []
    for a, g in zip(als, asg):
        asg_, k_, anc_ = g
        sizes = torch.bincount(asg_, minlength=k_).float().clamp_min(1)
        L_anc.append(torch.relu(0.25 / sizes - a[anc_]).pow(2).mean())
    L_anchor = torch.stack(L_anc).mean()
    tot = cfg['lm'] * L_margin + cfg['lr_'] * L_rank + cfg['ll'] * L_list + 0.001 * L_weight + 0.01 * L_ent + 0.01 * L_anchor
    return (tot, {'margin': float(L_margin), 'rank': float(L_rank), 'list': float(L_list), 'weight': float(L_weight), 'entropy': float(L_ent), 'anchor': float(L_anchor), 'total': float(tot)})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split-seed', dest='split_seed', type=int, default=42, help='FIXED across seeds (§3): splits+mining identical so seeds isolate training variance')
    ap.add_argument('--train-slices', dest='train_slices', nargs='+', default=None, help='§6 LOO: SOURCE slices only; target contributes no train/val/mining/norm/selection')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--out', default=f'{OUT}/mm_seed42.pt')
    ap.add_argument('--variant', default='marginmerge', choices=['marginmerge', 'marginmerge_no_listwise', 'marginmerge_no_vulnerability', 'marginmerge_margin_only'])
    a = ap.parse_args()
    if a.train_slices:
        global SLICES
        SLICES = list(a.train_slices)
        print(f'[LOO] training slices = {SLICES}', flush=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    b = torch.load(a.bank)
    Z = b['prototypes'].to(DEV).float()
    f = b['raw_frequencies'].float()
    raw = (f + 1e-08) ** 0.5
    w = (raw / raw.sum()).to(DEV)
    import build_query_prototypes as bqp
    splits = build_splits(a.split_seed)
    for s in SLICES:
        bqp.validate_no_leakage(a.bank, [f'{s}:{qi}' for qi in splits[s]['eval_q']])
    print('[leakage] splits disjoint + eval queries absent from prototype bank: PASSED', flush=True)
    neg = mine_hard_negatives(splits, a.split_seed)
    json.dump({s: {str(k_): {'pos': v['pos'], 'negs': v['negs']} for k_, v in neg[s].items()} for s in SLICES}, open(f'{OUT}/hard_negatives_split{a.split_seed}.json', 'w'))
    cch = cache_clusters(splits, neg, Z, w)
    Fall = torch.cat([cch[s][d]['F'] for s in SLICES for d in cch[s]], 0)
    ns = {'mean': Fall.mean(0), 'std': Fall.std(0).clamp_min(1e-06)}
    cfg_v = {'marginmerge': dict(lm=1.0, lr_=0.5, ll=0.5, use_list=True, use_vuln=True), 'marginmerge_no_listwise': dict(lm=1.0, lr_=0.5, ll=0.0, use_list=False, use_vuln=True), 'marginmerge_no_vulnerability': dict(lm=1.0, lr_=0.5, ll=0.5, use_list=True, use_vuln=False), 'marginmerge_margin_only': dict(lm=1.0, lr_=0.0, ll=0.0, use_list=False, use_vuln=True)}[a.variant]
    mlp = mm.WeightMLP().to(DEV)
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.0003, weight_decay=0.0001)
    pairs = [(s, qi) for s in SLICES for qi in splits[s]['train_q'] if qi in neg[s]]
    vpairs = [(s, qi) for s in SLICES for qi in splits[s]['val_q'] if qi in neg[s]]
    print(f'[train] {len(pairs)} train queries, {len(vpairs)} val queries, variant={a.variant}', flush=True)
    qcache = {s: torch.load(f'{CACHE}/{s}/qs.pt') for s in SLICES}

    def val_metrics():
        mlp.eval()
        nd = []
        flips = 0
        tot = 0
        with torch.no_grad():
            for s, qi in vpairs:
                r = neg[s][qi]
                Q = qcache[s][qi].to(DEV).float()
                Rp, _, _ = reps(cch[s][r['pos']], mlp, ns)
                sp = float(maxsim(Q, Rp))
                sns = []
                for d in r['negs']:
                    Rn, _, _ = reps(cch[s][d], mlp, ns)
                    sns.append(float(maxsim(Q, Rn)))
                rank = 1 + sum((1 for x in sns if x > sp))
                nd.append(1.0 / math.log2(rank + 1) if rank <= 5 else 0.0)
                for x, fx in zip(sns, r['neg_scores']):
                    tot += 1
                    if (r['pos_score'] > fx) != (sp > x):
                        flips += 1
        mlp.train()
        return (float(np.mean(nd)), flips / max(tot, 1))
    best = (-1, 1000000000.0, None, -1)
    hist = []
    for ep in range(a.epochs):
        t0 = time.time()
        np.random.shuffle(pairs)
        agg = {}
        opt.zero_grad()
        for i, (s, qi) in enumerate(pairs):
            r = neg[s][qi]
            Q = qcache[s][qi].to(DEV).float()
            cfg = dict(cfg_v)
            cfg['full_pos'] = r['pos_score']
            cfg['full_negs'] = r['neg_scores']
            tot, comp = losses(Q, cch[s][r['pos']], [cch[s][d] for d in r['negs']], mlp, ns, cfg)
            (tot / 4).backward()
            for k_, v_ in comp.items():
                agg[k_] = agg.get(k_, 0) + v_
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        vnd, vfl = val_metrics()
        agg = {k_: v_ / len(pairs) for k_, v_ in agg.items()}
        hist.append({'epoch': ep, **{k_: round(v_, 5) for k_, v_ in agg.items()}, 'val_ndcg5': round(vnd, 4), 'val_flip': round(vfl, 4), 'sec': round(time.time() - t0, 1)})
        print(f"[ep{ep}] total={agg['total']:.4f} margin={agg['margin']:.4f} rank={agg['rank']:.4f} list={agg['list']:.4f} | val_nDCG5={vnd:.4f} val_flip={vfl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vnd > best[0] or (abs(vnd - best[0]) < 1e-09 and vfl < best[1]):
            best = (vnd, vfl, {k_: v_.detach().cpu().clone() for k_, v_ in mlp.state_dict().items()}, ep)
        elif ep - best[3] >= 2:
            print(f'[early-stop] no val improvement for 2 epochs (best ep{best[3]})', flush=True)
            break
    torch.save({'state_dict': best[2], 'norm_stats': {k_: v_.cpu() for k_, v_ in ns.items()}, 'variant': a.variant, 'seed': a.seed, 'split_seed': a.split_seed, 'rho_train': RHO, 'n_params': sum((p.numel() for p in mlp.parameters())), 'best_epoch': best[3], 'val_ndcg5': best[0], 'val_flip': best[1], 'history': hist, 'bank': a.bank, 'splits': {s: {k_: v_ for k_, v_ in splits[s].items() if k_ != 'train_docs'} for s in SLICES}}, a.out)
    print(f'\nsaved {a.out} | best epoch {best[3]} val_nDCG5={best[0]:.4f} val_flip={best[1]:.4f}')
    print('TRAIN DONE')
if __name__ == '__main__':
    main()
