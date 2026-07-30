import sys, os, time, json
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = True
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
from datasets import load_dataset
from colpali_engine.models.qwen2_5.colqwen2_5.modeling_colqwen2_5 import ColQwen2_5
from colpali_engine.models.qwen2_5.colqwen2_5.processing_colqwen2_5 import ColQwen2_5_Processor
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colqwen')
OUT = 'outputs/baselines_inharness/sap_centrality'
os.makedirs(OUT, exist_ok=True)
MODEL = 'vidore/colqwen2.5-v0.1'
DEV = 'cuda'
DEPTH_FRAC = 0.55
DOC_TEST_FRAC = 0.3
REPO = {'arxivqa': ('vidore/arxivqa_test_subsampled', False), 'docvqa': ('vidore/docvqa_test_subsampled', False), 'infovqa': ('vidore/infovqa_test_subsampled', False), 'flickr': ('nlphuji/flickr_1k_test_image_text_retrieval', False), 'tatdqa': ('vidore/tatdqa_test', True), 'tabfquad': ('vidore/tabfquad_test_subsampled', True)}

def held_out_docs(name):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    return ([docs[i] for i in perm[:nte]], m)

def corpus_rows_for(name, ds, m):
    if not REPO[name][1]:
        return {d: d for d in range(len(ds))}
    fns = ds['image_filename']
    seen = {}
    rows = []
    for i, fn in enumerate(fns):
        if fn not in seen:
            seen[fn] = len(rows)
            rows.append(i)
    return {pos: row for pos, row in enumerate(rows)}

def main():
    slices = sys.argv[1:]
    t0 = time.time()
    proc = ColQwen2_5_Processor.from_pretrained(MODEL)
    model = ColQwen2_5.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='eager').to(DEV).eval()
    nL = len(model.model.language_model.layers)
    Lmid = int(nL * DEPTH_FRAC)
    print(f'[load] {time.time() - t0:.0f}s  layers={nL} using layer {Lmid}', flush=True)
    cap = {}

    def hook(mod, inp, out):
        if isinstance(out, tuple):
            for o in out:
                if torch.is_tensor(o) and o.dim() == 4:
                    cap['a'] = o
                    break
    model.model.language_model.layers[Lmid].self_attn.register_forward_hook(hook)
    for name in slices:
        fpath = f'{OUT}/{name}.pt'
        done = torch.load(fpath) if os.path.exists(fpath) else {}
        test_docs, m = held_out_docs(name)
        repo, _ = REPO[name]
        ds = load_dataset(repo, split='test')
        rowmap = corpus_rows_for(name, ds, m)
        todo = [d for d in test_docs if d not in done]
        print(f'[{name}] held-out={len(test_docs)} done={len(done)} todo={len(todo)}', flush=True)
        t1 = time.time()
        for n, d in enumerate(todo):
            img = ds[rowmap[d]]['image'].convert('RGB')
            batch = proc.process_images([img]).to(DEV)
            cap.clear()
            with torch.no_grad():
                _ = model(**batch)
            a = cap.get('a')
            if a is None:
                print(f'[{name}] doc {d}: NO ATTN', flush=True)
                continue
            indeg = a[0].float().mean(0).sum(0).cpu()
            done[d] = indeg
            if (n + 1) % 20 == 0:
                torch.save(done, fpath)
                print(f'[{name}] {n + 1}/{len(todo)}  {(time.time() - t1) / (n + 1):.1f}s/img  seq={a.shape[-1]}', flush=True)
        torch.save(done, fpath)
        print(f'[{name}] SAVED {len(done)} docs  {time.time() - t1:.0f}s', flush=True)
    print('SAP_EXTRACT DONE', flush=True)
if __name__ == '__main__':
    main()
