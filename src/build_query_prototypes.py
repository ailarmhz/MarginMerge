import os, json, hashlib, argparse
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
from sklearn.cluster import KMeans, MiniBatchKMeans
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
MODEL = os.environ.get('CATTS_MODEL', 'vidore/colqwen2.5-v0.1')
SLICES = ['arxivqa', 'docvqa', 'infovqa', 'tatdqa', 'tabfquad', 'flickr']
DOC_TEST_FRAC = 0.3
MAX_TOK_PER_Q = 32
MAX_TOTAL = 1000000
M = 128
ITERS = 50
RESTARTS = 5

def held_out_docs(name):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    return (set((docs[i] for i in perm[:nte])), gold)

def collect_training_tokens(seed, loo):
    rng = np.random.RandomState(seed)
    toks, tw, src, excl = ([], [], [], [])
    for name in SLICES:
        import time as _t
        _t0 = _t.time()
        test_docs, gold = held_out_docs(name)
        qs = torch.load(f'{CACHE}/{name}/qs.pt')
        print(f'[bank] {name}: loaded qs ({len(qs)} queries) {_t.time() - _t0:.1f}s', flush=True)
        for qi, g in enumerate(gold):
            qid = f'{name}:{qi}'
            if g in test_docs:
                excl.append(qid)
                continue
            if loo and name == loo:
                excl.append(qid)
                continue
            q = qs[qi].float()
            L = q.shape[0]
            idx = np.arange(L) if L <= MAX_TOK_PER_Q else rng.choice(L, MAX_TOK_PER_Q, replace=False)
            v = q[idx]
            v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-08)
            toks.append(v)
            tw.append(torch.full((len(idx),), 1.0 / len(idx)))
            src.append(qid)
    T = torch.cat(toks, 0)
    W = torch.cat(tw, 0)
    if T.shape[0] > MAX_TOTAL:
        keep = rng.choice(T.shape[0], MAX_TOTAL, replace=False)
        T = T[keep]
        W = W[keep]
    return (T.numpy().astype(np.float32), W.numpy().astype(np.float32), src, excl)

def build(seed, out, loo=None):
    T, W, src, excl = collect_training_tokens(seed, loo)
    d = T.shape[1]
    print(f'[bank] clustering {T.shape[0]} training-query tokens (d={d}) into M={M} ...', flush=True)
    if T.shape[0] <= 20000:
        km = KMeans(n_clusters=M, n_init=RESTARTS, max_iter=ITERS, random_state=seed).fit(T)
    else:
        km = MiniBatchKMeans(n_clusters=M, n_init=RESTARTS, max_iter=ITERS, batch_size=8192, random_state=seed).fit(T)
    Z = torch.tensor(km.cluster_centers_, dtype=torch.float32)
    Z = Z / Z.norm(dim=1, keepdim=True).clamp_min(1e-08)
    Tt = torch.tensor(T)
    assign = (Tt @ Z.T).argmax(dim=1)
    f = torch.zeros(M)
    f.index_add_(0, assign, torch.tensor(W))
    manifest = {'prototype_bank_seed': seed, 'model': MODEL, 'embedding_dim': d, 'n_prototypes': M, 'source_datasets': SLICES, 'source_splits': f'train_docs_queries(70pct) leave_one_out={loo}', 'n_source_queries': len(src), 'n_excluded_queries': len(excl), 'source_query_ids_sha256': hashlib.sha256('|'.join(sorted(src)).encode()).hexdigest(), 'excluded_query_ids_sha256': hashlib.sha256('|'.join(sorted(excl)).encode()).hexdigest(), 'source_query_ids': src, 'excluded_query_ids': excl, 'max_tokens_per_query': MAX_TOK_PER_Q, 'max_total_tokens': MAX_TOTAL, 'clustering': {'algo': 'sklearn.KMeans(spherical-fallback)', 'iters': ITERS, 'restarts': RESTARTS, 'seed': seed}, 'n_tokens_clustered': int(T.shape[0])}
    torch.save({'prototypes': Z, 'raw_frequencies': f, 'manifest': manifest}, out)
    print(f'[bank] saved {out}  M={M} d={d}  tokens={T.shape[0]}  src_q={len(src)} excl_q={len(excl)} loo={loo}')
    print(f'[bank] freq: nonzero_modes={(f > 0).sum().item()}/{M}  max={f.max():.4f} min={f.min():.4f}')
    return out

def validate_no_leakage(bank_path, eval_qids):
    man = torch.load(bank_path)['manifest']
    src = set(man['source_query_ids'])
    overlap = set(eval_qids) & src
    if overlap:
        raise RuntimeError(f'LEAKAGE: {len(overlap)} eval queries in prototype bank source, e.g. {list(overlap)[:5]}')
    return True
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--loo', type=str, default=None, help="leave-one-dataset-out: exclude this slice's queries")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    build(a.seed, a.out, a.loo)
