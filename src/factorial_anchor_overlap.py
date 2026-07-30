import os, json, math, csv, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import gapcover as gc, factorial as fac
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
OUT = 'outputs/factorial'
BANK = 'outputs/gapcover/bank_seed42.pt'
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3
PHASE = {'A': dict(seeds=[42], rhos=[0.05], datasets=['arxivqa', 'flickr', 'tabfquad']), 'B': dict(seeds=[42, 43, 44], rhos=[0.05, 0.1], datasets=['arxivqa', 'docvqa', 'infovqa', 'tatdqa', 'tabfquad', 'flickr'])}

def jacc(a, b):
    a, b = (set(a), set(b))
    return len(a & b) / max(len(a | b), 1)

def heldout(name):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    return [docs[i] for i in perm[:nte]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    P = PHASE[a.phase]
    b = torch.load(BANK)
    Z = b['prototypes'].to(DEV).float()
    f = b['raw_frequencies'].float()
    raw = (f + 1e-08) ** 0.5
    w = (raw / raw.sum()).to(DEV)
    rows = []
    for ds in P['datasets']:
        ps = torch.load(f'{CACHE}/{ds}/ps.pt')
        docs = heldout(ds)
        for rho in P['rhos']:
            for seed in P['seeds']:
                jrk, jrc, jkc, urand, ukc, ucov = ([], [], [], [], [], [])
                for d in docs:
                    V = ps[d].to(DEV).float()
                    n = V.shape[0]
                    k = min(n, max(1, math.ceil(rho * n)))
                    A = gc.response_matrix(V, Z)
                    U = set(torch.unique((Z @ V.T).argmax(1)).tolist())
                    aR = fac.select_anchors(V, Z, w, k, 'random', seed, d)['anchors'].tolist()
                    aK = fac.select_anchors(V, Z, w, k, 'kcenter', seed, d)['anchors'].tolist()
                    aC = fac.select_anchors(V, Z, w, k, 'coverage', seed, d)['anchors'].tolist()
                    jrk.append(jacc(aR, aK))
                    jrc.append(jacc(aR, aC))
                    jkc.append(jacc(aK, aC))
                    urand.append(jacc(aR, U))
                    ukc.append(jacc(aK, U))
                    ucov.append(jacc(aC, U))
                rows.append({'dataset': ds, 'rho': rho, 'seed': seed, 'n_docs': len(docs), 'jaccard_random_kcenter': round(float(np.mean(jrk)), 4), 'jaccard_random_coverage': round(float(np.mean(jrc)), 4), 'jaccard_kcenter_coverage': round(float(np.mean(jkc)), 4), 'Ucover_random': round(float(np.mean(urand)), 4), 'Ucover_kcenter': round(float(np.mean(ukc)), 4), 'Ucover_coverage': round(float(np.mean(ucov)), 4)})
                print(f'[{ds} rho={rho} seed={seed}] J(rand,cov)={np.mean(jrc):.3f} J(kc,cov)={np.mean(jkc):.3f} Ucov: rand={np.mean(urand):.3f} cov={np.mean(ucov):.3f}', flush=True)
    with open(f'{OUT}/anchor_overlap_diagnostics.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print('ANCHOR_OVERLAP DONE')
if __name__ == '__main__':
    main()
