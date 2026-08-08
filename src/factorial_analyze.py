import os, json, glob, csv, argparse
import numpy as np
SRC = '.'
OUT = f'{SRC}/outputs/factorial'
RUNS = f'{OUT}/runs'
DATASETS = ['arxivqa', 'docvqa', 'infovqa', 'tatdqa', 'tabfquad', 'flickr']
DCOL = {'arxivqa': 'ArxivQA', 'docvqa': 'DocVQA', 'infovqa': 'InfoVQA', 'tatdqa': 'TAT-DQA', 'tabfquad': 'TabFQuad', 'flickr': 'Flickr'}
METRICS = ['ndcg_at_5', 'recall_at_1', 'recall_at_5', 'recall_at_10', 'flip_rate', 'reversal_rate', 'margin_abs_err', 'score_abs_err', 'pos_score_abs_err', 'ms_per_doc', 'peak_mem_mb']
SIG = 0.005
B = 2000
RNG = np.random.RandomState(0)
PHASE = {'A': dict(seeds=[42], rhos=[0.05], datasets=['arxivqa', 'flickr', 'tabfquad']), 'B': dict(seeds=[42, 43, 44], rhos=[0.05, 0.1], datasets=DATASETS)}

def load_runs(P):
    recs = []
    for f in glob.glob(f'{RUNS}/*.json'):
        r = json.load(open(f))
        if r['seed'] in P['seeds'] and round(r['rho'], 2) in P['rhos'] and (r['dataset'] in P['datasets']):
            r['anchor'] = r['anchor_strategy']
            r['representative'] = r['representative_strategy']
            r['loss'] = r['loss_strategy']
            recs.append(r)
    return recs

def vkey(r):
    return (r['anchor'], r['representative'], r['loss'])

def write_all_runs(recs):
    rows = []
    for r in recs:
        row = {'run_id': r['run_id'], 'dataset': r['dataset'], 'anchor': r['anchor'], 'representative': r['representative'], 'loss': r['loss'], 'rho': r['rho'], 'seed': r['seed'], 'kind': r['kind']}
        row.update({m: r['metrics'].get(m) for m in METRICS})
        rows.append(row)
    rows.sort(key=lambda x: (x['rho'], x['dataset'], x['anchor'], x['representative'], x['loss'], x['seed']))
    with open(f'{OUT}/all_runs.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows

def summary_mean_std(recs):
    grp = {}
    for r in recs:
        grp.setdefault((*vkey(r), r['rho'], r['dataset']), []).append(r['metrics'])
    rows = []
    for key, ms in sorted(grp.items()):
        a, rep, loss, rho, ds = key
        row = {'anchor': a, 'representative': rep, 'loss': loss, 'rho': rho, 'dataset': ds, 'n_seeds': len(ms)}
        for m in METRICS:
            vals = [x[m] for x in ms if x.get(m) is not None]
            row[f'{m}_mean'] = round(float(np.mean(vals)), 4)
            row[f'{m}_std'] = round(float(np.std(vals)), 4)
        rows.append(row)
    with open(f'{OUT}/summary_mean_std.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows

def _pool(recs, key, rho, ds, metric='ndcg_at_5'):
    arrs = [np.asarray(r['per_query'][metric]) for r in recs if vkey(r) == key and r['rho'] == rho and (r['dataset'] == ds)]
    return np.concatenate(arrs) if arrs else np.array([])

def bootstrap_ci(recs):
    keys = sorted(set((vkey(r) for r in recs)))
    rows = []
    for rho in sorted(set((r['rho'] for r in recs))):
        for ds in sorted(set((r['dataset'] for r in recs))):
            for key in keys:
                x = _pool(recs, key, rho, ds)
                if x.size == 0:
                    continue
                bs = x[RNG.randint(0, x.size, size=(B, x.size))].mean(1)
                rows.append({'anchor': key[0], 'representative': key[1], 'loss': key[2], 'rho': rho, 'dataset': ds, 'ndcg_mean': round(float(x.mean()), 4), 'ci_lo': round(float(np.percentile(bs, 2.5)), 4), 'ci_hi': round(float(np.percentile(bs, 97.5)), 4), 'n_queries': int(x.size)})
    with open(f'{OUT}/bootstrap_ci.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows

def paired_delta(recs, ka, kb, rho, ds):
    A = [r for r in recs if vkey(r) == ka and r['rho'] == rho and (r['dataset'] == ds)]
    Bv = [r for r in recs if vkey(r) == kb and r['rho'] == rho and (r['dataset'] == ds)]
    if not A or not Bv:
        return None
    seeds = sorted(set((r['seed'] for r in A)) & set((r['seed'] for r in Bv)))
    da = np.concatenate([np.asarray(next((r for r in A if r['seed'] == sd))['per_query']['ndcg_at_5']) for sd in seeds])
    db = np.concatenate([np.asarray(next((r for r in Bv if r['seed'] == sd))['per_query']['ndcg_at_5']) for sd in seeds])
    if da.shape != db.shape:
        return None
    d = da - db
    bs = d[RNG.randint(0, d.size, size=(B, d.size))].mean(1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

def pairwise_deltas(recs):
    refs = {'merging': ('-', 'merging', '-'), 'coverage_response': ('coverage', 'response_centroid', 'none'), 'full_marginmerge': ('coverage', 'learned', 'full')}
    keys = sorted(set((vkey(r) for r in recs)))
    rows = []
    for rho in sorted(set((r['rho'] for r in recs))):
        for ds in sorted(set((r['dataset'] for r in recs))):
            for key in keys:
                for rname, rkey in refs.items():
                    if key == rkey:
                        continue
                    res = paired_delta(recs, key, rkey, rho, ds)
                    if res is None:
                        continue
                    d, lo, hi = res
                    sig = (lo > 0 or hi < 0) and abs(d) >= SIG
                    rows.append({'anchor': key[0], 'representative': key[1], 'loss': key[2], 'rho': rho, 'dataset': ds, 'vs': rname, 'delta_ndcg': round(d, 4), 'ci_lo': round(lo, 4), 'ci_hi': round(hi, 4), 'significant': sig})
    with open(f'{OUT}/pairwise_deltas.csv', 'w', newline='') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    return rows

def mean_ndcg(recs, filt, rho=None):
    vals = [r['metrics']['ndcg_at_5'] for r in recs if all((r[k] == v for k, v in filt.items())) and (rho is None or r['rho'] == rho)]
    return float(np.mean(vals)) if vals else None

def latex_table(summ, rho):
    order = [('-', 'merging', '-', 'Geometric merging'), ('-', 'full', '-', 'Full index'), ('random', 'anchor', 'none', 'Retained anchor'), ('random', 'uniform_centroid', 'none', 'Uniform centroid'), ('random', 'response_centroid', 'none', 'Response centroid'), ('random', 'learned', 'score_reconstruction', 'Learned (score-recon)'), ('random', 'learned', 'margin', 'Learned (margin)'), ('random', 'learned', 'full', 'Learned (full)'), ('kcenter', 'anchor', 'none', 'Retained anchor'), ('kcenter', 'uniform_centroid', 'none', 'Uniform centroid'), ('kcenter', 'response_centroid', 'none', 'Response centroid'), ('kcenter', 'learned', 'score_reconstruction', 'Learned (score-recon)'), ('kcenter', 'learned', 'margin', 'Learned (margin)'), ('kcenter', 'learned', 'full', 'Learned (full)'), ('coverage', 'anchor', 'none', 'Retained anchor'), ('coverage', 'uniform_centroid', 'none', 'Uniform centroid'), ('coverage', 'response_centroid', 'none', 'Response centroid'), ('coverage', 'learned', 'score_reconstruction', 'Learned (score-recon)'), ('coverage', 'learned', 'margin', 'Learned (margin)'), ('coverage', 'learned', 'full', 'Learned (full)')]
    S = {(r['anchor'], r['representative'], r['loss'], r['dataset']): r['ndcg_at_5_mean'] for r in summ if r['rho'] == rho}
    ds_present = [d for d in DATASETS if any((r['dataset'] == d and r['rho'] == rho for r in summ))]
    L = ['\\begin{tabular}{lll' + 'c' * (len(ds_present) + 1) + '}', '\\toprule', 'Anchor & Representative & Loss & ' + ' & '.join((DCOL[d] for d in ds_present)) + ' & Avg. \\\\', '\\midrule']
    last_anchor = None
    for anc, rep, loss, lab in order:
        cells = [S.get((anc, rep, loss, d)) for d in ds_present]
        if all((c is None for c in cells)):
            continue
        avg = np.mean([c for c in cells if c is not None])
        astr = {'random': 'Random', 'kcenter': '$k$-center', 'coverage': 'Coverage', '-': '--'}[anc]
        if anc != last_anchor and last_anchor is not None:
            L.append('\\midrule')
        anc_disp = astr if anc != last_anchor else ''
        last_anchor = anc
        lstr = {'none': '--', '-': '--', 'score_reconstruction': 'score-recon', 'margin': 'margin', 'full': 'full'}[loss]
        row = f'{anc_disp} & {lab} & {lstr} & ' + ' & '.join((f'{c:.3f}' if c is not None else '--' for c in cells)) + f' & {avg:.3f} ' + '\\\\'
        L.append(row)
    L += ['\\bottomrule', '\\end{tabular}']
    return '\n'.join(L)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['A', 'B'])
    a = ap.parse_args()
    P = PHASE[a.phase]
    recs = load_runs(P)
    print(f'[analyze] {len(recs)} runs for phase {a.phase}')
    write_all_runs(recs)
    summ = summary_mean_std(recs)
    bci = bootstrap_ci(recs)
    pdl = pairwise_deltas(recs)
    print(f'[analyze] all_runs/summary/bootstrap_ci({len(bci)})/pairwise_deltas({len(pdl)}) written')
    rho0 = P['rhos'][0]
    open(f'{OUT}/table_factorial_rho{int(rho0 * 100)}.tex', 'w').write(latex_table(summ, rho0))
    if len(P['rhos']) > 1:
        open(f"{OUT}/table_factorial_rho{int(P['rhos'][1] * 100)}.tex", 'w').write(latex_table(summ, P['rhos'][1]))

    def me(rep, loss, anc, rho=rho0):
        return mean_ndcg(recs, {'anchor': anc, 'representative': rep, 'loss': loss}, rho)
    lines = [f'# Factorial ablation report (phase {a.phase})\n', f'Backbone: frozen ColQwen2.5. Metric: nDCG@5, held-out Table-9 protocol. Bootstrap CIs over queries (B={B}). Effects at ρ={rho0}. Non-significant if |Δ|<{SIG} unless a paired CI excludes 0.\n', '## 1. Main effects\n', '### Anchor strategy (holding synthesis fixed), mean nDCG@5\n', '| Synthesis | random | kcenter | coverage |', '|---|---|---|---|']
    for rep, loss, tag in [('response_centroid', 'none', 'fixed response centroid'), ('learned', 'margin', 'learned (margin)')]:
        cells = [me(rep, loss, anc) for anc in ['random', 'kcenter', 'coverage']]
        lines.append(f'| {tag} | ' + ' | '.join((f'{c:.4f}' if c is not None else '-' for c in cells)) + ' |')
    lines += ['\n### Learned synthesis vs fixed response centroid (holding anchors fixed)\n', '| Anchor | response_centroid | learned(margin) | Δ (learned−fixed) |', '|---|---|---|---|']
    anchor_gain = {}
    for anc in ['random', 'kcenter', 'coverage']:
        base = me('response_centroid', 'none', anc)
        learn = me('learned', 'margin', anc)
        d = learn - base if base is not None and learn is not None else None
        anchor_gain[anc] = d
        lines.append(f'| {anc} | {base:.4f} | {learn:.4f} | {d:+.4f} |')

    def cov_minus_rand(rep, loss):
        cov = me(rep, loss, 'coverage')
        ran = me(rep, loss, 'random')
        return cov - ran if cov is not None and ran is not None else None
    g_fixed = cov_minus_rand('response_centroid', 'none')
    g_learn = cov_minus_rand('learned', 'margin')
    inter = g_learn - g_fixed if g_fixed is not None and g_learn is not None else None
    lines += ['\n## 2. Interaction (is the coverage-random gain amplified by learning?)\n', f'- coverage-random under fixed response centroid: {g_fixed:+.4f}', f'- coverage-random under learned synthesis (margin): {g_learn:+.4f}', f'- interaction (learned minus fixed): {inter:+.4f}']
    lines += ['\n## 3. Loss ladder (learned synthesis), mean nDCG@5 over anchors & datasets\n', '| Loss | mean nDCG@5 |', '|---|---|']
    loss_means = {}
    for loss in ['score_reconstruction', 'margin', 'full']:
        v = mean_ndcg(recs, {'representative': 'learned', 'loss': loss}, rho0)
        loss_means[loss] = v
        lines.append(f'| {loss} | {v:.4f} |' if v is not None else f'| {loss} | - |')

    def verdict(d):
        if d is None:
            return 'n/a'
        return 'no (below %.3f)' % SIG if abs(d) < SIG else 'yes' if d > 0 else 'no (negative)'
    learn_ind = float(np.mean([v for v in anchor_gain.values() if v is not None])) if anchor_gain else None
    q4 = loss_means.get('margin') - loss_means.get('score_reconstruction') if loss_means.get('margin') is not None and loss_means.get('score_reconstruction') is not None else None
    interp = 'primarily interaction' if inter is not None and abs(inter) >= SIG and (g_fixed is None or abs(g_fixed) < SIG) else 'both independent and interaction' if inter is not None and abs(inter) >= SIG else 'mostly independent effects'
    lines += ['\n## 4. Summary\n', f'- coverage anchors help on their own: {verdict(g_fixed)} (coverage-random under fixed synthesis = {g_fixed:+.4f})', f'- learned synthesis helps on its own: {verdict(learn_ind)} (mean learned-fixed over anchors = {learn_ind:+.4f})', f'- interaction = {inter:+.4f}, i.e. {interp}', f'- margin beats score reconstruction: {verdict(q4)} (margin-score_reconstruction = {q4:+.4f})']
    open(f'{OUT}/report.md', 'w').write('\n'.join(lines) + '\n')
    print('[analyze] report.md + LaTeX table written\n')
    print('\n'.join(lines))
if __name__ == '__main__':
    main()
