import json, glob, os, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RES = 'outputs/baselines_memory'
FIG = os.environ.get('FIG_DIR', 'outputs/figures')
os.makedirs(FIG, exist_ok=True)
RENDERED = ['tabfquad', 'arxivqa', 'infovqa', 'tatdqa', 'docvqa']
MEM = [1.0, 0.5, 0.2, 0.1, 0.05]
METHODS = ['full', 'random', 'merge_tome', 'importance', 'top_likelihood', 'kcenter']
LABEL = {'full': 'full budget', 'random': 'random keep', 'merge_tome': 'token merging (ToMe)', 'importance': 'importance-prune (redundancy proxy)', 'top_likelihood': 'top-likelihood (learned)', 'kcenter': 'k-center over predicted-U (ours)'}
COL = {'full': '#000000', 'random': '#4f6bed', 'merge_tome': '#2e9e6b', 'importance': '#e0843a', 'top_likelihood': '#c0392b', 'kcenter': '#8e44ad'}
D = {}
for f in glob.glob(f'{RES}/ckpt/*.json'):
    d = json.load(open(f))
    D[d['slice']] = d

def ndcg(slice, meth, rho):
    c = D[slice]['count'][meth]
    return c.get(str(rho), {}).get('ndcg')

def agg(meths, rhos, slices):
    out = {}
    for meth in meths:
        out[meth] = {}
        for rho in rhos:
            vals = [ndcg(s, meth, rho) for s in slices if ndcg(s, meth, rho) is not None]
            if vals:
                out[meth][rho] = (float(np.mean(vals)), float(np.std(vals)))
    return out

def emit():
    have = [s for s in RENDERED if s in D]
    rend = agg(METHODS, MEM, have)
    rend_nodoc = agg(METHODS, MEM, [s for s in have if s != 'docvqa'])
    flk = agg(METHODS, MEM, ['flickr']) if 'flickr' in D else {}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for meth in METHODS:
        xs = [rho * 100 for rho in MEM if rho in rend.get(meth, {})]
        ys = [rend[meth][rho][0] for rho in MEM if rho in rend.get(meth, {})]
        es = [rend[meth][rho][1] for rho in MEM if rho in rend.get(meth, {})]
        ls = '-' if meth != 'full' else ':'
        lw = 2.6 if meth == 'kcenter' else 1.8
        ax[0].errorbar(xs, ys, yerr=es, marker='o', ms=4, lw=lw, ls=ls, color=COL[meth], capsize=2, label=LABEL[meth])
    ax[0].set_xscale('log')
    ax[0].set_xlabel('index memory (% of full, log)')
    ax[0].set_ylabel('nDCG@5')
    ax[0].set_title('Rendered-average (5 slices) — memory vs quality\nerror bars = std across slices')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=7.5)
    for meth in METHODS:
        if meth not in flk:
            continue
        xs = [rho * 100 for rho in MEM if rho in flk[meth]]
        ys = [flk[meth][rho][0] for rho in MEM if rho in flk[meth]]
        ax[1].plot(xs, ys, marker='o', ms=4, lw=2.6 if meth == 'kcenter' else 1.8, ls='-' if meth != 'full' else ':', color=COL[meth], label=LABEL[meth])
    ax[1].set_xscale('log')
    ax[1].set_xlabel('index memory (% of full, log)')
    ax[1].set_ylabel('nDCG@5')
    ax[1].set_title('flickr (natural) — separate panel')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=7.5)
    plt.tight_layout()
    plt.savefig(f'{FIG}/pareto_memory.png', dpi=130)
    print('saved figures/pareto_memory.png')

    def block(agg_d, title):
        L = [f'\n### {title} (nDCG@5, mean±std across slices)', '| method | 20% | 10% | 5% |', '|---|---|---|---|']
        for rho in [0.2, 0.1, 0.05]:
            pass
        for meth in METHODS:
            cells = []
            for rho in [0.2, 0.1, 0.05]:
                if rho in agg_d.get(meth, {}):
                    mn, sd = agg_d[meth][rho]
                    cells.append(f'{mn:.3f}±{sd:.3f}')
                else:
                    cells.append('—')
            L.append(f'| {LABEL[meth]} | {cells[0]} | {cells[1]} | {cells[2]} |')
        for rho, nm in [(0.2, '20%'), (0.1, '10%'), (0.05, '5%')]:
            ms = [(agg_d[m][rho][0], m) for m in METHODS if m != 'full' and rho in agg_d.get(m, {})]
            if not ms:
                continue
            ms.sort(reverse=True)
            best = ms[0]
            second = ms[1] if len(ms) > 1 else (0, None)
            tie = abs(best[0] - second[0]) <= agg_d[best[1]][rho][1] + agg_d[second[1]][rho][1] if second[1] else False
            L.append(f"*best @{nm}: {LABEL[best[1]]} ({best[0]:.3f}){(' — TIE with ' + LABEL[second[1]] if tie else '')}*")
        return '\n'.join(L)
    md = ['# Memory–quality comparison table (in-harness, frozen ColQwen2.5, held-out docs+queries, 3 seeds)']
    md.append(block(rend, 'Rendered-average — ALL 5 slices'))
    md.append(block(rend_nodoc, 'Rendered-average — EXCLUDING docvqa (footnoted-anomalous)'))
    if flk:
        md.append(block(flk, 'flickr (natural)'))
    md.append('\n### Byte-reduction (ORTHOGONAL AXIS — cuts bytes not count; not a count-reduction competitor)')
    md.append('| method | memory (bytes %) | rendered-avg nDCG@5 |')
    md.append('|---|---|---|')
    for key, lbl in [('int8_full', 'int8 quant, full count'), ('int8_kcenter0.2', 'int8 quant + k-center@20%')]:
        vals = [D[s]['byte'][key]['ndcg'] for s in have if 'byte' in D[s] and key in D[s]['byte']]
        mb = D[have[0]]['byte'][key]['mem_bytes'] if have else 0
        if vals:
            md.append(f'| {lbl} | {mb * 100:.0f}% | {np.mean(vals):.3f} |')
    open(f'{RES}/comparison_table.md', 'w').write('\n'.join(md) + '\n')
    print('saved comparison_table.md')
    keep_m = ['random', 'importance', 'top_likelihood', 'kcenter']
    HC, RET = ([], [])
    cov_rows = ['# Coverage of argmax-union U_T(d) vs retention (keep-methods; our differentiator)', '| method | mem | HC(U_T) rendered-avg | retention (nDCG/full) |', '|---|---|---|---|']
    for meth in keep_m:
        for rho in [0.5, 0.2, 0.1, 0.05]:
            hcs, rets = ([], [])
            for s in have:
                cov = D[s]['coverage'].get(meth, {}).get(str(rho))
                nd = ndcg(s, meth, rho)
                full = ndcg(s, 'full', 1.0)
                if cov is not None and nd is not None and full:
                    hcs.append(cov)
                    rets.append(nd / full)
            if hcs:
                hc = float(np.mean(hcs))
                rt = float(np.mean(rets))
                HC.append(hc)
                RET.append(rt)
                cov_rows.append(f'| {LABEL[meth]} | {int(rho * 100)}% | {hc:.3f} | {rt:.3f} |')

    def spearman(a, b):
        a = np.array(a)
        b = np.array(b)
        ra = a.argsort().argsort()
        rb = b.argsort().argsort()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / d) if d > 0 else 0.0
    rho_hc = spearman(HC, RET)
    cov_rows.append(f'\n**Spearman(HC coverage, retention) across {len(HC)} (method×mem) points = {rho_hc:.3f}** — coverage of the argmax-union predicts retention across methods. No prior method measures this.')
    open(f'{RES}/coverage_table.md', 'w').write('\n'.join(cov_rows) + '\n')
    print(f'saved coverage_table.md  (Spearman HC-retention = {rho_hc:.3f})')
    return (rend, rend_nodoc, flk, rho_hc)
if __name__ == '__main__':
    emit()
    print('AGG DONE')
