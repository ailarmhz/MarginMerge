import sys, os, io, warnings, logging, contextlib
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '8')
import torch
from datasets import load_dataset
from colpali_engine.models import ColPali, ColPaliProcessor
MODEL = 'vidore/colpali-v1.3'
DEV = 'cuda'
buf = io.StringIO()
hf_logger = logging.getLogger('transformers.modeling_utils')
h = logging.StreamHandler(buf)
hf_logger.addHandler(h)
hf_logger.setLevel(logging.WARNING)
with warnings.catch_warnings(record=True) as wcap:
    warnings.simplefilter('always')
    proc = ColPaliProcessor.from_pretrained(MODEL)
    model = ColPali.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV).eval()
hf_logger.removeHandler(h)
log_text = buf.getvalue() + '\n'.join((str(w.message) for w in wcap))
bad = [ln for ln in log_text.splitlines() if 'newly initialized' in ln or 'should probably TRAIN' in ln]
print('=== (A) load-warning capture ===')
print(log_text.strip()[:2000] if log_text.strip() else '(no warnings emitted)')
if bad:
    print('\n*** GATE A FAILED: PaliGemma LM was random-initialized ***')
    for b in bad:
        print('   ', b)
    sys.exit(2)
print('>>> GATE A PASSED: no random-init warning\n')
ds = load_dataset('vidore/arxivqa_test_subsampled', split='test')
N = 10
imgs = [ds[i]['image'].convert('RGB') for i in range(N)]
qraw = [ds[i]['query'] for i in range(N)]
queries = [q[0] if isinstance(q, (list, tuple)) else q for q in qraw]

@torch.no_grad()
def emb_imgs(imgs):
    b = proc.process_images(imgs).to(DEV)
    e = model(**b)
    m = b['attention_mask'].bool()
    return [e[k][m[k]].float() for k in range(e.shape[0])]

@torch.no_grad()
def emb_qs(qs):
    b = proc.process_queries(qs).to(DEV)
    e = model(**b)
    m = b['attention_mask'].bool()
    return [e[k][m[k]].float() for k in range(e.shape[0])]
P = emb_imgs(imgs)
Q = emb_qs(queries)
correct = 0
for qi in range(N):
    scores = [float((Q[qi] @ P[di].T).max(dim=1)[0].sum()) for di in range(N)]
    top = int(torch.tensor(scores).argmax())
    ok = top == qi
    correct += ok
    print(f"  q{qi:2d} -> top1=doc{top:2d} {('OK' if ok else 'MISS')} (self={scores[qi]:.2f} best={max(scores):.2f})")
print(f'\n=== (B) retrieval sanity: {correct}/{N} top-1 correct ===')
if correct < 8:
    print('*** GATE B FAILED: retrieval sanity < 8/10 -> embeddings are not trustworthy ***')
    sys.exit(2)
print('>>> GATE B PASSED\n>>> COLPALI VERIFICATION PASSED')
