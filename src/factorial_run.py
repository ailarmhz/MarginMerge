import os, json, argparse, subprocess, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np, torch
torch.set_num_threads(1)
import marginmerge as mm
import factorial_eval as fe
SRC = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('FACT_OUT', 'outputs/factorial')
RUNS = f'{OUT}/runs'
CKPTS = f'{OUT}/ckpts'
os.makedirs(RUNS, exist_ok=True)
os.makedirs(CKPTS, exist_ok=True)
BANK = os.environ.get('BANK', 'outputs/bank_seed42.pt')
DEV = 'cuda'
ANCHORS = ['random', 'kcenter', 'coverage']
NONLEARNED_REPS = ['anchor', 'uniform_centroid', 'response_centroid']
LEARNED_LOSSES = ['score_reconstruction', 'margin', 'full']
PHASE = {'A': dict(datasets=['arxivqa', 'flickr', 'tabfquad'], rhos=[0.05], seeds=[42]), 'B': dict(datasets=['arxivqa', 'docvqa', 'infovqa', 'tatdqa', 'tabfquad', 'flickr'], rhos=[0.05, 0.1], seeds=[42, 43, 44])}

def ckpt_path(anchor, loss, seed):
    return f'{CKPTS}/{anchor}__{loss}__seed{seed}.pt'

def ensure_ckpts(seeds):
    todo = [(a, l, s) for s in seeds for a in ANCHORS for l in LEARNED_LOSSES if not os.path.exists(ckpt_path(a, l, s))]
    print(f'[train] {len(todo)} MLPs to train (of {len(ANCHORS) * len(LEARNED_LOSSES) * len(seeds)})', flush=True)
    for a, l, s in todo:
        out = ckpt_path(a, l, s)
        cmd = [sys.executable, f'{SRC}/factorial_train.py', '--bank', BANK, '--anchor', a, '--loss', l, '--seed', str(s), '--out', out]
        t0 = time.time()
        print(f'[train] anchor={a} loss={l} seed={s} ...', flush=True)
        r = subprocess.run(cmd, cwd=SRC, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            print(r.stdout[-2000:])
            print(r.stderr[-2000:])
            raise RuntimeError(f'train failed {a}/{l}/{s}')
        print(f'[train] done {a}/{l}/{s} ({time.time() - t0:.0f}s)', flush=True)

def load_bank():
    b = torch.load(BANK)
    Z = b['prototypes'].to(DEV).float()
    f = b['raw_frequencies'].float()
    raw = (f + 1e-08) ** 0.5
    w = (raw / raw.sum()).to(DEV)
    return (Z, w)

def run_variant(cfg, Z, w, mlp, ns, datasets):
    for s in datasets:
        rid = f"{cfg['anchor']}__{cfg['representative']}__{cfg['loss']}__rho{int(cfg['rho'] * 100)}__seed{cfg['seed']}__{s}"
        fp = f'{RUNS}/{rid}.json'
        if os.path.exists(fp):
            continue
        met, perq = fe.eval_slice(s, cfg, Z, w, mlp, ns)
        json.dump({'run_id': rid, 'dataset': s, 'anchor_strategy': cfg['anchor'], 'representative_strategy': cfg['representative'], 'loss_strategy': cfg['loss'], 'rho': cfg['rho'], 'seed': cfg['seed'], 'kind': cfg['kind'], 'metrics': met, 'per_query': perq}, open(fp, 'w'))
        print(f"[eval] {rid} nDCG@5={met['ndcg_at_5']:.4f} flip={met['flip_rate']:.4f}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    P = PHASE[a.phase]
    ensure_ckpts(P['seeds'])
    Z, w = load_bank()
    for seed in P['seeds']:
        for rho in P['rhos']:
            run_variant({'kind': 'full', 'rho': rho, 'seed': seed, 'anchor': '-', 'representative': 'full', 'loss': '-'}, Z, w, None, None, P['datasets'])
            run_variant({'kind': 'merging', 'rho': rho, 'seed': seed, 'anchor': '-', 'representative': 'merging', 'loss': '-'}, Z, w, None, None, P['datasets'])
            for anc in ANCHORS:
                for rep in NONLEARNED_REPS:
                    run_variant({'kind': 'factorial', 'rho': rho, 'seed': seed, 'anchor': anc, 'representative': rep, 'loss': 'none'}, Z, w, None, None, P['datasets'])
            for anc in ANCHORS:
                for loss in LEARNED_LOSSES:
                    ck = torch.load(ckpt_path(anc, loss, seed))
                    mlp = mm.WeightMLP().to(DEV)
                    mlp.load_state_dict(ck['state_dict'])
                    mlp.eval()
                    ns = {k_: v_.to(DEV) for k_, v_ in ck['norm_stats'].items()}
                    run_variant({'kind': 'factorial', 'rho': rho, 'seed': seed, 'anchor': anc, 'representative': 'learned', 'loss': loss}, Z, w, mlp, ns, P['datasets'])
    print(f'FACTORIAL_RUN_{a.phase}_DONE', flush=True)
if __name__ == '__main__':
    main()
