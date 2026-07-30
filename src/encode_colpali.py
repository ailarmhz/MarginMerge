import sys, os, json, time
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
MODEL = 'vidore/colpali-v1.3'
OUT = 'data/cache_colpali'
DEV = 'cuda'
ONE2ONE = {'arxivqa': ('vidore/arxivqa_test_subsampled', 'test', 'image', 'query'), 'docvqa': ('vidore/docvqa_test_subsampled', 'test', 'image', 'query'), 'infovqa': ('vidore/infovqa_test_subsampled', 'test', 'image', 'query'), 'flickr': ('nlphuji/flickr_1k_test_image_text_retrieval', 'test', 'image', 'caption')}
DEDUP = {'tatdqa': ('vidore/tatdqa_test', 'dense'), 'tabfquad': ('vidore/tabfquad_test_subsampled', 'dense')}

def eff_rank(M):
    s = torch.linalg.svdvals(M.float().to(DEV))
    s = s[s > 1e-06]
    return float(s.sum() ** 2 / s.square().sum()) if s.numel() else 0.0

def redundancy(M):
    Mg = M.float().to(DEV)
    G = Mg @ Mg.T
    G.fill_diagonal_(-1.0)
    return float(G.max(dim=1)[0].mean())

@torch.no_grad()
def enc_images(ds, rows, model, proc, name, t0, bs=8):
    embs, ntok = ([], [])
    for i in range(0, len(rows), bs):
        imgs = [ds[j]['image'].convert('RGB') for j in rows[i:i + bs]]
        b = proc.process_images(imgs).to(DEV)
        e = model(**b)
        m = b['attention_mask'].bool()
        for k in range(e.shape[0]):
            v = e[k][m[k]].to(torch.float16).cpu()
            embs.append(v)
            ntok.append(int(v.shape[0]))
        if i // bs % 10 == 0:
            print(f'[{name}] img {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    return (embs, ntok)

@torch.no_grad()
def enc_queries(texts, model, proc, bs=32):
    embs, ntok = ([], [])
    for i in range(0, len(texts), bs):
        b = proc.process_queries(texts[i:i + bs]).to(DEV)
        e = model(**b)
        m = b['attention_mask'].bool()
        for k in range(e.shape[0]):
            v = e[k][m[k]].to(torch.float16).cpu()
            embs.append(v)
            ntok.append(int(v.shape[0]))
    return (embs, ntok)

def encode_one2one(name, model, proc, t0):
    repo, split, icol, qcol = ONE2ONE[name]
    outdir = f'{OUT}/{name}'
    os.makedirs(outdir, exist_ok=True)
    ds = load_dataset(repo, split=split)
    n = len(ds)
    qraw = [ds[i][qcol] for i in range(n)]
    queries = [q[0] if isinstance(q, (list, tuple)) else q for q in qraw]
    print(f'[{name}] n={n}', flush=True)
    ps, np_tok = enc_images(ds, list(range(n)), model, proc, name, t0)
    qs, nq_tok = enc_queries(queries, model, proc)
    er = [eff_rank(p) for p in ps]
    rd = [redundancy(p) for p in ps]
    torch.save(qs, f'{outdir}/qs.pt')
    torch.save(ps, f'{outdir}/ps.pt')
    json.dump({'slice': name, 'repo': repo, 'n': n, 'gold': list(range(n)), 'nq_tok': nq_tok, 'np_tok': np_tok, 'eff_rank': er, 'redundancy': rd}, open(f'{outdir}/meta.json', 'w'))
    print(f'[{name}] DONE n={n} mean_np={sum(np_tok) / n:.0f} {time.time() - t0:.0f}s', flush=True)

def encode_dedup(name, model, proc, t0):
    repo, cat = DEDUP[name]
    outdir = f'{OUT}/{name}'
    os.makedirs(outdir, exist_ok=True)
    ds = load_dataset(repo, split='test')
    fns = ds['image_filename']
    corpus_idx = {}
    corpus_rows = []
    for i, fn in enumerate(fns):
        if fn not in corpus_idx:
            corpus_idx[fn] = len(corpus_rows)
            corpus_rows.append(i)
    gold = [corpus_idx[fn] for fn in fns]
    queries = [ds[i]['query'] for i in range(len(ds))]
    keep = [i for i, q in enumerate(queries) if isinstance(q, str) and q.strip()]
    gold = [gold[i] for i in keep]
    queries = [queries[i] for i in keep]
    print(f'[{name}] queries={len(queries)} corpus={len(corpus_rows)} cat={cat}', flush=True)
    ps, np_tok = enc_images(ds, corpus_rows, model, proc, name, t0, bs=8)
    qs, nq_tok = enc_queries(queries, model, proc)
    er = [eff_rank(p) for p in ps]
    rd = [redundancy(p) for p in ps]
    torch.save(qs, f'{outdir}/qs.pt')
    torch.save(ps, f'{outdir}/ps.pt')
    json.dump({'slice': name, 'repo': repo, 'category': cat, 'n_queries': len(ds), 'n_corpus': len(ps), 'gold': gold, 'nq_tok': nq_tok, 'np_tok': np_tok, 'eff_rank': er, 'redundancy': rd}, open(f'{outdir}/meta.json', 'w'))
    print(f'[{name}] DONE corpus={len(ps)} mean_np={sum(np_tok) / len(ps):.0f} {time.time() - t0:.0f}s', flush=True)

def main():
    t0 = time.time()
    proc = ColPaliProcessor.from_pretrained(MODEL)
    model = ColPali.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
    print(f'[all] ColPali loaded {time.time() - t0:.0f}s', flush=True)
    for name in sys.argv[1:]:
        if os.path.exists(f'{OUT}/{name}/meta.json'):
            print(f'[{name}] cached, skip', flush=True)
            continue
        (encode_one2one if name in ONE2ONE else encode_dedup)(name, model, proc, t0)
    print('[all] ENCODE DONE', flush=True)
if __name__ == '__main__':
    main()
