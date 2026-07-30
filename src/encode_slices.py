import sys, os, json, time
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '8')
MAX_CORPUS = int(os.environ.get('MAX_CORPUS', '0'))
import torch
torch.backends.cuda.matmul.allow_tf32 = True
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
print('[cuda] init', flush=True)
from datasets import load_dataset
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5.processing_colqwen2_5 import ColQwen2_5_Processor
MODEL = 'vidore/colqwen2.5-v0.1'
OUT = 'data/cache_colqwen'
SLICES = {'tatdqa': ('vidore/tatdqa_test', 'dense'), 'tabfquad': ('vidore/tabfquad_test_subsampled', 'dense'), 'shiftproject': ('vidore/shiftproject_test', 'dense'), 'synth_energy': ('vidore/syntheticDocQA_energy_test', 'semidense'), 'synth_gov': ('vidore/syntheticDocQA_government_reports_test', 'semidense')}

def eff_rank(M):
    s = torch.linalg.svdvals(M.float().to('cuda'))
    s = s[s > 1e-06]
    return float(s.sum() ** 2 / s.square().sum()) if s.numel() else 0.0

def redundancy(M):
    Mg = M.float().to('cuda')
    G = Mg @ Mg.T
    G.fill_diagonal_(-1.0)
    return float(G.max(dim=1)[0].mean())

def encode_slice(name, model, proc, t0):
    repo, cat = SLICES[name]
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
    if MAX_CORPUS and len(corpus_rows) > MAX_CORPUS:
        step = len(corpus_rows) / MAX_CORPUS
        keep_pages = sorted({int(i * step) for i in range(MAX_CORPUS)})
        keepset = set(keep_pages)
        remap = {old: new for new, old in enumerate(keep_pages)}
        corpus_rows = [corpus_rows[p] for p in keep_pages]
        q_keep = [i for i, fn in enumerate(fns) if corpus_idx[fn] in keepset]
        gold = [remap[corpus_idx[fns[i]]] for i in q_keep]
        queries = [ds[i]['query'] for i in q_keep]
    else:
        gold = [corpus_idx[fn] for fn in fns]
        queries = [ds[i]['query'] for i in range(len(ds))]
    keep = [i for i, q in enumerate(queries) if isinstance(q, str) and q.strip()]
    if len(keep) < len(queries):
        print(f'[{name}] dropping {len(queries) - len(keep)} null queries', flush=True)
        gold = [gold[i] for i in keep]
        queries = [queries[i] for i in keep]
    print(f'[{name}] queries={len(queries)} corpus={len(corpus_rows)} cat={cat}', flush=True)

    @torch.no_grad()
    def enc_imgs(rows, bs=4):
        embs, ntok = ([], [])
        for i in range(0, len(rows), bs):
            imgs = [ds[j]['image'].convert('RGB') for j in rows[i:i + bs]]
            b = proc.process_images(imgs).to('cuda')
            e = model(**b)
            m = b['attention_mask'].bool()
            for k in range(e.shape[0]):
                v = e[k][m[k]].to(torch.float16).cpu()
                embs.append(v)
                ntok.append(int(v.shape[0]))
            if i // bs % 20 == 0:
                print(f'[{name}] img {i}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
        return (embs, ntok)

    @torch.no_grad()
    def enc_qs(texts, bs=32):
        embs, ntok = ([], [])
        for i in range(0, len(texts), bs):
            b = proc.process_queries(texts[i:i + bs]).to('cuda')
            e = model(**b)
            m = b['attention_mask'].bool()
            for k in range(e.shape[0]):
                v = e[k][m[k]].to(torch.float16).cpu()
                embs.append(v)
                ntok.append(int(v.shape[0]))
        return (embs, ntok)
    ps, np_tok = enc_imgs(corpus_rows)
    qs, nq_tok = enc_qs(queries)
    print(f'[{name}] computing proxies {time.time() - t0:.0f}s', flush=True)
    erank = [eff_rank(p) for p in ps]
    redun = [redundancy(p) for p in ps]
    torch.save(qs, f'{outdir}/qs.pt')
    torch.save(ps, f'{outdir}/ps.pt')
    json.dump({'slice': name, 'repo': repo, 'category': cat, 'n_queries': len(ds), 'n_corpus': len(ps), 'gold': gold, 'nq_tok': nq_tok, 'np_tok': np_tok, 'eff_rank': erank, 'redundancy': redun}, open(f'{outdir}/meta.json', 'w'))
    print(f'[{name}] DONE corpus={len(ps)} mean_np={sum(np_tok) / len(ps):.0f} mean_erank={sum(erank) / len(ps):.1f} {time.time() - t0:.0f}s', flush=True)

def main():
    t0 = time.time()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL)
    model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda').eval()
    print(f'[all] model loaded {time.time() - t0:.0f}s', flush=True)
    for name in sys.argv[1:]:
        if os.path.exists(f'{OUT}/{name}/meta.json'):
            print(f'[{name}] cached, skip', flush=True)
            continue
        encode_slice(name, model, proc, t0)
    print(f'[all] DONE {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
