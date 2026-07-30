import os, json, math, time, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import torch.nn.functional as F
import marginmerge as mm
import factorial as fac
import train_marginmerge as tm
DEV = 'cuda'
RHO = tm.RHO

def cache_clusters_cfg(splits, neg, Z, w, anchor_strategy, base_seed):
    cch = {}
    for s in tm.SLICES:
        ps = torch.load(f'{tm.CACHE}/{s}/ps.pt')
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
            c = fac.build_clusters(V, Z, w, k, anchor_strategy, base_seed=base_seed, doc_id=d)
            cs[d] = {'V': V, 'assign': c['assign'], 'bw': c['bw'], 'F': c['F'], 'anchors': c['anchors'], 'k': k}
        cch[s] = cs
        print(f'[cache:{anchor_strategy}] {s}: {len(cs)} docs in {time.time() - t0:.0f}s', flush=True)
    return cch
PRECOMP = os.environ.get('FACT_PRECOMP', 'outputs/factorial/precompute')
os.makedirs(PRECOMP, exist_ok=True)

def build_or_load_precompute(anchor_strategy, split_seed, Z, w, bank):
    tag = f"{anchor_strategy}__ss{split_seed}__{'-'.join(tm.SLICES)}"
    fp = f'{PRECOMP}/{tag}.pt'
    import build_query_prototypes as bqp
    if os.path.exists(fp):
        bundle = torch.load(fp)
        splits, neg, ns_cpu = (bundle['splits'], bundle['neg'], bundle['ns'])
        for s in tm.SLICES:
            bqp.validate_no_leakage(bank, [f'{s}:{qi}' for qi in splits[s]['eval_q']])
        cch = {}
        for s in tm.SLICES:
            ps = torch.load(f'{tm.CACHE}/{s}/ps.pt')
            cs = {}
            for d, e in bundle['cch'][s].items():
                cs[d] = {'V': ps[d].to(DEV).float(), 'assign': e['assign'].to(DEV).long(), 'bw': e['bw'].to(DEV), 'F': e['F'].to(DEV), 'anchors': e['anchors'].to(DEV).long(), 'k': e['k']}
            cch[s] = cs
        ns = {k_: v_.to(DEV) for k_, v_ in ns_cpu.items()}
        print(f'[precompute] loaded {tag} ({sum((len(cch[s]) for s in tm.SLICES))} docs)', flush=True)
        return (splits, neg, cch, ns)
    splits = tm.build_splits(split_seed)
    for s in tm.SLICES:
        bqp.validate_no_leakage(bank, [f'{s}:{qi}' for qi in splits[s]['eval_q']])
    neg = tm.mine_hard_negatives(splits, split_seed)
    cch = cache_clusters_cfg(splits, neg, Z, w, anchor_strategy, split_seed)
    Fall = torch.cat([cch[s][d]['F'] for s in tm.SLICES for d in cch[s]], 0)
    ns = {'mean': Fall.mean(0), 'std': Fall.std(0).clamp_min(1e-06)}
    dump = {'splits': splits, 'neg': neg, 'ns': {k_: v_.cpu() for k_, v_ in ns.items()}, 'cch': {s: {d: {'assign': cch[s][d]['assign'].cpu().to(torch.int16), 'bw': cch[s][d]['bw'].cpu(), 'F': cch[s][d]['F'].cpu(), 'anchors': cch[s][d]['anchors'].cpu(), 'k': cch[s][d]['k']} for d in cch[s]} for s in tm.SLICES}}
    torch.save(dump, fp)
    print(f'[precompute] saved {tag}', flush=True)
    return (splits, neg, cch, ns)

def score_reconstruction_loss(Q, dpos, dneg, mlp, ns, full_pos, full_negs):
    Rp, _, hp = tm.reps(dpos, mlp, ns)
    sp = tm.maxsim(Q, Rp)
    sn, hs = ([], [hp])
    for dn in dneg:
        Rn, _, hn = tm.reps(dn, mlp, ns)
        sn.append(tm.maxsim(Q, Rn))
        hs.append(hn)
    sn = torch.stack(sn)
    tp = torch.tensor(full_pos, device=DEV, dtype=torch.float32)
    tn = torch.tensor(np.asarray(full_negs), device=DEV, dtype=torch.float32)
    L_pos = F.huber_loss(sp, tp, delta=0.5)
    L_neg = F.huber_loss(sn, tn, delta=0.5, reduction='mean')
    L_weight = (torch.cat(hs) ** 2).mean()
    tot = L_pos + L_neg + 0.001 * L_weight
    return (tot, {'score_pos': float(L_pos), 'score_neg': float(L_neg), 'weight': float(L_weight), 'total': float(tot)})
LOSS_CFG = {'margin': dict(lm=1.0, lr_=0.0, ll=0.0, use_list=False, use_vuln=True), 'full': dict(lm=1.0, lr_=0.5, ll=0.5, use_list=True, use_vuln=True)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--anchor', required=True, choices=['random', 'kcenter', 'coverage'])
    ap.add_argument('--loss', required=True, choices=['score_reconstruction', 'margin', 'full'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split-seed', dest='split_seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--train-slices', dest='train_slices', nargs='+', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.train_slices:
        tm.SLICES = list(a.train_slices)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    b = torch.load(a.bank)
    Z = b['prototypes'].to(DEV).float()
    f = b['raw_frequencies'].float()
    raw = (f + 1e-08) ** 0.5
    w = (raw / raw.sum()).to(DEV)
    splits, neg, cch, ns = build_or_load_precompute(a.anchor, a.split_seed, Z, w, a.bank)
    print(f'[leakage] PASSED. anchor={a.anchor} loss={a.loss} seed={a.seed} split_seed={a.split_seed}', flush=True)
    mlp = mm.WeightMLP().to(DEV)
    opt = torch.optim.AdamW(mlp.parameters(), lr=0.0003, weight_decay=0.0001)
    pairs = [(s, qi) for s in tm.SLICES for qi in splits[s]['train_q'] if qi in neg[s]]
    vpairs = [(s, qi) for s in tm.SLICES for qi in splits[s]['val_q'] if qi in neg[s]]
    qcache = {s: torch.load(f'{tm.CACHE}/{s}/qs.pt') for s in tm.SLICES}
    print(f'[train] {len(pairs)} train / {len(vpairs)} val queries, params={sum((p.numel() for p in mlp.parameters()))}', flush=True)

    def val_metrics():
        mlp.eval()
        nd = []
        flips = 0
        tot = 0
        with torch.no_grad():
            for s, qi in vpairs:
                r = neg[s][qi]
                Q = qcache[s][qi].to(DEV).float()
                Rp, _, _ = tm.reps(cch[s][r['pos']], mlp, ns)
                sp = float(tm.maxsim(Q, Rp))
                sns = [float(tm.maxsim(Q, tm.reps(cch[s][d], mlp, ns)[0])) for d in r['negs']]
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
            if a.loss == 'score_reconstruction':
                tot, comp = score_reconstruction_loss(Q, cch[s][r['pos']], [cch[s][d] for d in r['negs']], mlp, ns, r['pos_score'], r['neg_scores'])
            else:
                cfg = dict(LOSS_CFG[a.loss])
                cfg['full_pos'] = r['pos_score']
                cfg['full_negs'] = r['neg_scores']
                tot, comp = tm.losses(Q, cch[s][r['pos']], [cch[s][d] for d in r['negs']], mlp, ns, cfg)
            (tot / 4).backward()
            for k_, v_ in comp.items():
                agg[k_] = agg.get(k_, 0) + v_
            if (i + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
        vnd, vfl = val_metrics()
        agg = {k_: v_ / len(pairs) for k_, v_ in agg.items()}
        hist.append({'epoch': ep, **{k_: round(v_, 5) for k_, v_ in agg.items()}, 'val_ndcg5': round(vnd, 4), 'val_flip': round(vfl, 4)})
        print(f"[ep{ep}] total={agg.get('total', 0):.4f} val_nDCG5={vnd:.4f} val_flip={vfl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vnd > best[0] or (abs(vnd - best[0]) < 1e-09 and vfl < best[1]):
            best = (vnd, vfl, {k_: v_.detach().cpu().clone() for k_, v_ in mlp.state_dict().items()}, ep)
        elif ep - best[3] >= 2:
            print(f'[early-stop] best ep{best[3]}', flush=True)
            break
    anchors_dump = {s: {int(d): {'anchors': cch[s][d]['anchors'].cpu().tolist(), 'assign': cch[s][d]['assign'].cpu().to(torch.int16).tolist()} for d in cch[s]} for s in tm.SLICES}
    torch.save({'state_dict': best[2], 'norm_stats': {k_: v_.cpu() for k_, v_ in ns.items()}, 'anchor_strategy': a.anchor, 'loss_strategy': a.loss, 'representative_strategy': 'learned', 'seed': a.seed, 'split_seed': a.split_seed, 'rho_train': RHO, 'n_params': sum((p.numel() for p in mlp.parameters())), 'best_epoch': best[3], 'val_ndcg5': best[0], 'val_flip': best[1], 'history': hist, 'bank': a.bank, 'train_slices': list(tm.SLICES), 'splits': {s: {k_: v_ for k_, v_ in splits[s].items() if k_ != 'train_docs'} for s in tm.SLICES}, 'anchors_assign': anchors_dump}, a.out)
    print(f'saved {a.out} | best ep{best[3]} val_nDCG5={best[0]:.4f}\nFACTORIAL_TRAIN DONE')
if __name__ == '__main__':
    main()
