import sys, os, time, json
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = True
_ = torch.zeros(1, device='cuda')
torch.cuda.synchronize()
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
CACHE = os.environ.get('CATTS_CACHE', 'data/cache_colpali')
OUT = os.environ.get('SAP_OUT', 'outputs/baselines_inharness_colpali/sap_centrality_colpali')
os.makedirs(OUT, exist_ok=True)
MODEL = 'vidore/colpali-v1.3'
DEV = 'cuda'
DEPTH_FRAC = float(os.environ.get('SAP_DEPTH', '0.55'))
DOC_TEST_FRAC = 0.3
REPO = {'arxivqa': ('vidore/arxivqa_test_subsampled', False), 'docvqa': ('vidore/docvqa_test_subsampled', False), 'infovqa': ('vidore/infovqa_test_subsampled', False), 'flickr': ('nlphuji/flickr_1k_test_image_text_retrieval', False), 'tatdqa': ('vidore/tatdqa_test', True), 'tabfquad': ('vidore/tabfquad_test_subsampled', True)}

def held_out_docs(name, train=False):
    m = json.load(open(f'{CACHE}/{name}/meta.json'))
    gold = m['gold']
    docs = sorted(set(gold))
    perm = np.random.RandomState(0).permutation(len(docs))
    nte = max(1, int(round(DOC_TEST_FRAC * len(docs))))
    if train:
        return ([docs[i] for i in perm[nte:]], m)
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

def find_lm_layers(model):
    cands = ['model.language_model.model.layers', 'model.language_model.layers', 'model.model.language_model.layers', 'language_model.model.layers']
    for path in cands:
        obj = model
        try:
            for p in path.split('.'):
                obj = getattr(obj, p)
            if hasattr(obj, '__len__') and len(obj) > 0:
                return (obj, path)
        except AttributeError:
            continue
    best = (None, None, 0)
    for n, mod in model.named_modules():
        if mod.__class__.__name__.endswith('ModuleList') and len(mod) > best[2] and ('layers' in n) and ('self_attn' in dir(mod[0])):
            best = (mod, n, len(mod))
    return (best[0], best[1])

def image_token_span(batch, model):
    return batch['attention_mask'][0].bool()

def make_hook(cap):

    def hook(mod, inp, out):
        if isinstance(out, (tuple, list)):
            for o in out:
                if torch.is_tensor(o) and o.dim() == 4:
                    cap['a'] = o
                    return
        elif torch.is_tensor(out) and out.dim() == 4:
            cap['a'] = out
    return hook

def load_model():
    proc = ColPaliProcessor.from_pretrained(MODEL)
    model = ColPali.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation='eager').to(DEV).eval()
    model.config.output_attentions = True
    if hasattr(model.config, 'text_config'):
        model.config.text_config.output_attentions = True
    layers, path = find_lm_layers(model)
    return (proc, model, layers, path)

@torch.no_grad()
def run_one(model, proc, img, layer, cap):
    batch = proc.process_images([img]).to(DEV)
    cap.clear()
    _ = model(**batch, output_attentions=True)
    a = cap.get('a')
    span = image_token_span(batch, model)
    return (a, span, batch)

def introspect():
    proc, model, layers, path = load_model()
    print(f'[introspect] LM layers found at: {path}  (n={len(layers)})', flush=True)
    nL = len(layers)
    Lmid = int(nL * DEPTH_FRAC)
    cap = {}
    h = layers[Lmid].self_attn.register_forward_hook(make_hook(cap))
    ds = load_dataset(REPO['arxivqa'][0], split='test')
    a, span, batch = run_one(model, proc, ds[0]['image'].convert('RGB'), Lmid, cap)
    h.remove()
    if a is None:
        print('[introspect] *** NO 4-D ATTENTION CAPTURED *** at layer', Lmid)
        sys.exit(2)
    print(f"[introspect] layer {Lmid}/{nL} attn shape={tuple(a.shape)}  seq={a.shape[-1]}  img_tokens={int(span.sum())}  total_valid={int(batch['attention_mask'].sum())}")
    print('>>> SAP introspection OK')

def extract(slices, sweep=False, dev=False):
    proc, model, layers, path = load_model()
    nL = len(layers)
    print(f'[load] layers={nL} at {path}', flush=True)
    fracs = [0.35, 0.45, 0.55, 0.65, 0.75] if sweep else [DEPTH_FRAC]
    for name in slices:
        base_docs, m = held_out_docs(name, train=dev)
        repo, _ = REPO[name]
        ds = load_dataset(repo, split='test')
        rowmap = corpus_rows_for(name, ds, m)
        docs = base_docs[:60] if sweep else base_docs
        for frac in fracs:
            Lmid = int(nL * frac)
            pre = 'dev' if dev else ''
            tag = f'{name}' if not sweep else f'{name}_{pre}f{int(frac * 100)}'
            fpath = f'{OUT}/{tag}.pt'
            done = torch.load(fpath) if os.path.exists(fpath) and (not sweep) else {}
            todo = [d for d in docs if d not in done]
            cap = {}
            h = layers[Lmid].self_attn.register_forward_hook(make_hook(cap))
            print(f'[{tag}] layer={Lmid}/{nL} held-out={len(docs)} todo={len(todo)}', flush=True)
            t1 = time.time()
            for n, d in enumerate(todo):
                img = ds[rowmap[d]]['image'].convert('RGB')
                a, span, _ = run_one(model, proc, img, Lmid, cap)
                if a is None:
                    print(f'[{tag}] doc {d}: NO ATTN', flush=True)
                    continue
                A = a[0].float().mean(0)
                sidx = span.nonzero(as_tuple=True)[0]
                sub = A[sidx][:, sidx]
                indeg = sub.sum(0).cpu()
                done[d] = indeg
                if (n + 1) % 20 == 0 and (not sweep):
                    torch.save(done, fpath)
                    print(f'[{tag}] {n + 1}/{len(todo)} {(time.time() - t1) / (n + 1):.2f}s/img n_img={indeg.numel()}', flush=True)
            h.remove()
            torch.save(done, fpath)
            print(f'[{tag}] SAVED {len(done)} docs {time.time() - t1:.0f}s', flush=True)
    print('SAP_EXTRACT_COLPALI DONE', flush=True)
if __name__ == '__main__':
    args = sys.argv[1:]
    if '--introspect' in args:
        introspect()
    elif '--devsweep' in args:
        sl = [a for a in args if a != '--devsweep']
        extract(sl, sweep=True, dev=True)
    elif '--sweep' in args:
        sl = [a for a in args if a != '--sweep']
        extract(sl, sweep=True)
    else:
        extract(args, sweep=False)
