import sys, os, json, time
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('RAYON_NUM_THREADS', '1')
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
print('[cuda] context initialized', flush=True)
from datasets import load_dataset
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5.processing_colqwen2_5 import ColQwen2_5_Processor
MODEL = 'vidore/colqwen2.5-v0.1'
OUT = 'data/cache_colqwen'
SLICES = {'arxivqa': ('vidore/arxivqa_test_subsampled', 'test', 'image', 'query'), 'docvqa': ('vidore/docvqa_test_subsampled', 'test', 'image', 'query'), 'infovqa': ('vidore/infovqa_test_subsampled', 'test', 'image', 'query'), 'flickr': ('nlphuji/flickr_1k_test_image_text_retrieval', 'test', 'image', 'caption')}

def eff_rank(M):
    s = torch.linalg.svdvals(M.float())
    s = s[s > 1e-06]
    return float(s.sum() ** 2 / s.square().sum()) if s.numel() else 0.0

def redundancy(M):
    G = M.float() @ M.float().T
    G.fill_diagonal_(-1.0)
    return float(G.max(dim=1)[0].mean())

def encode_slice(name, model, proc, t0):
    repo, split, icol, qcol = SLICES[name]
    outdir = f'{OUT}/{name}'
    os.makedirs(outdir, exist_ok=True)
    ds = load_dataset(repo, split=split)
    n = len(ds)
    qraw = [ds[i][qcol] for i in range(n)]
    queries = [q[0] if isinstance(q, (list, tuple)) else q for q in qraw]
    print(f'[{name}] n={n}', flush=True)
    img_bs = 16 if name == 'flickr' else 4

    @torch.no_grad()
    def encode_images(idxs, bs=img_bs):
        embs, ntok = ([], [])
        for i in range(0, len(idxs), bs):
            imgs = [ds[j][icol].convert('RGB') for j in idxs[i:i + bs]]
            b = proc.process_images(imgs).to('cuda')
            e = model(**b)
            m = b['attention_mask'].bool()
            for k in range(e.shape[0]):
                v = e[k][m[k]].to(torch.float16).cpu()
                embs.append(v)
                ntok.append(int(v.shape[0]))
            if i // bs % 3 == 0:
                print(f'[{name}] img {i + len(imgs)}/{len(idxs)} {time.time() - t0:.0f}s', flush=True)
        return (embs, ntok)

    @torch.no_grad()
    def encode_queries(texts, bs=32):
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
    ps, np_tok = encode_images(list(range(n)))
    qs, nq_tok = encode_queries(queries)
    print(f'[{name}] encoded, computing density proxies {time.time() - t0:.0f}s', flush=True)
    erank = [eff_rank(p) for p in ps]
    redun = [redundancy(p) for p in ps]
    torch.save(qs, f'{outdir}/qs.pt')
    torch.save(ps, f'{outdir}/ps.pt')
    meta = {'slice': name, 'repo': repo, 'n': n, 'gold': list(range(n)), 'nq_tok': nq_tok, 'np_tok': np_tok, 'eff_rank': erank, 'redundancy': redun}
    json.dump(meta, open(f'{outdir}/meta.json', 'w'))
    print(f'[{name}] DONE n={n} mean np_tok={sum(np_tok) / n:.0f} mean eff_rank={sum(erank) / n:.1f} {time.time() - t0:.0f}s', flush=True)

def main():
    names = sys.argv[1:]
    t0 = time.time()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL)
    model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to('cuda').eval()
    print(f'[all] model loaded {time.time() - t0:.0f}s for slices={names}', flush=True)
    for name in names:
        if os.path.exists(f'{OUT}/{name}/meta.json'):
            print(f'[{name}] already cached, skipping', flush=True)
            continue
        encode_slice(name, model, proc, t0)
    print(f'[all] ALL DONE {time.time() - t0:.0f}s', flush=True)
if __name__ == '__main__':
    main()
