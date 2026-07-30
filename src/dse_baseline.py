import sys, os, json, math
import numpy as np, torch
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
OUT = os.environ.get('CATTS_RES_INHARNESS', 'outputs/baselines_inharness')
os.makedirs(OUT, exist_ok=True)
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
DEV = 'cuda'
DOC_TEST_FRAC = 0.3

def meanpool(vecs):
    v = vecs.float().mean(0)
    return v / (v.norm() + 1e-08)

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
    eval_q = [qi for qi, g in enumerate(gold) if g in col]
    gcol = torch.tensor([col[gold[qi]] for qi in eval_q], device=DEV)
    Dv = torch.stack([meanpool(ps[d].to(DEV)) for d in test_docs])
    Qv = torch.stack([meanpool(qs[qi].to(DEV)) for qi in eval_q])
    S = Qv @ Dv.T
    order = torch.argsort(S, dim=1, descending=True)
    v = []
    for i in range(len(eval_q)):
        pos = (order[i] == gcol[i]).nonzero(as_tuple=True)[0].item() + 1
        v.append(1.0 / math.log2(pos + 1) if pos <= 5 else 0.0)
    ndcg = float(np.mean(v))
    mean_np = sum(m['np_tok']) / len(m['np_tok'])
    return {'slice': name, 'n_test_docs': len(test_docs), 'n_eval_q': len(eval_q), 'dse_ndcg5': round(ndcg, 4), 'mean_patches': round(mean_np, 0), 'mem_ratio': round(1.0 / mean_np, 5)}

def main():
    res = {}
    print(f"{'slice':<10}{'DSE nDCG@5':>12}{'mean_np':>10}{'mem(1/n)':>12}", flush=True)
    for s in sys.argv[1:]:
        r = run(s)
        res[s] = r
        print(f"{s:<10}{r['dse_ndcg5']:>12}{int(r['mean_patches']):>10}{r['mem_ratio'] * 100:>11.2f}%", flush=True)
    json.dump(res, open(f'{OUT}/dse_baseline.json', 'w'), indent=2)
    print('DSE DONE', flush=True)
if __name__ == '__main__':
    main()
