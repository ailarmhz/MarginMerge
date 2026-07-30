# MarginMerge: Coverage-Aware Compression of Multi-Vector Visual Document Retrievers

Official code for **“Coverage Matters: MarginMerge for Compressing Multi-Vector Visual Document Retrievers.”**

Multi-vector visual retrievers (ColPali, ColQwen) store hundreds–thousands of patch vectors per page. **MarginMerge** compresses each document **once at indexing time** into `k = ⌈ρn⌉` synthetic representatives, while keeping the **standard MaxSim retrieval interface unchanged**. It (1) selects **coverage-aware anchors** that span complementary query-relevant regions, (2) assigns patches to anchors, (3) synthesizes one representative per cluster with a **tiny shared network**, and (4) trains that network by **ranking-margin distillation** against the frozen full index. Across six datasets on **ColQwen2.5** and **ColPali**, it attains the best matched query-agnostic average at 5% and 10% retention while preserving 97–99% of full-index nDCG@5.

---

## Repository layout

```
.
├── src/                         # all method + pipeline modules (add to PYTHONPATH)
│   ├── gapcover.py              # coverage objective + coverage-aware anchor selection   (Eq. 4–5)
│   ├── clustering.py            # patch→anchor assignment (anchors own their cluster)     (Eq. 6)
│   ├── marginmerge.py           # learned representative synthesis: features, WeightMLP, reps  (Eq. 7–9)
│   ├── train_marginmerge.py     # hard-negative mining + ranking-margin distillation      (Eq. 10–12)
│   ├── eval_marginmerge.py      # MarginMerge vs merging / response-centroid / full (Tables 1, 3)
│   ├── factorial.py             # config-driven factorial: anchor × representative × loss (Table 4)
│   ├── factorial_{train,eval,run,analyze,anchor_overlap}.py
│   ├── build_query_prototypes.py# training-query prototype bank (leakage-checked)
│   ├── baselines_memory.py, baselines_aggregate.py   # random / k-center / importance / top-likelihood / ToMe-merge / int8 (Tables 1–2)
│   ├── dse_baseline.py, hpc_eval.py, sap_extract*.py, sap_eval.py   # DSE / HPC / SAP references (Table 1)
│   └── encode.py, encode_slices.py, encode_colpali.py, verify_colpali.py   # produce the frozen embedding caches
├── tests/                       # unit tests for the core (synthetic tensors, no data needed)
├── configs/factorial.json       # the factorial grid + shared-config contract
├── results/                     # the paper's aggregated numbers (CSV / LaTeX / report)
├── requirements.txt             # core + ColQwen2.5 encoding
└── requirements-colpali.txt     # ColPali-v1.3 encoding (separate env, see below)
```

`src/` uses flat imports, so run everything with `PYTHONPATH=src` from the repo root.
Large/regenerable artifacts (`*.pt`, `data/`, `outputs/`) are git-ignored — see [Data](#1-data-frozen-embedding-caches).

## Method → code map

| Paper | Equations | Code |
|---|---|---|
| Coverage `C_{ti}` + submodular anchor objective `F_d` | (4), (5) | `gapcover.py`: `response_matrix`, `soft_gap_coverage`, `gapcover_select` |
| Nearest-anchor assignment | (6) | `clustering.py`: `assign_clusters` |
| Log-weight `ℓ_i`, within-cluster `α_i`, representative `r_c` | (7), (8), (9) | `marginmerge.py`: `base_weights`, `patch_features`, `WeightMLP`, `reps_from_weights`, `marginmerge_reps` |
| Full/compressed margins + margin distillation loss | (10), (11), (12) | `train_marginmerge.py`: `mine_hard_negatives`, `losses` |
| Compressed MaxSim (unchanged retrieval) | (2) | used throughout eval scripts |

---

## Installation

Two environments are required because the two backbones need different `transformers`:

```bash
# core + ColQwen2.5 encoding
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

# ColPali-v1.3 encoding / SAP attention (separate env)
python -m venv .venv-colpali && . .venv-colpali/bin/activate
pip install -r requirements-colpali.txt
```

**Version pins matter.**
- ColPali-v1.3 is a PaliGemma checkpoint; `transformers ≥ 4.49` silently **re-initialises** its language model (garbage embeddings). `colpali-engine==0.3.5` resolves to the correct pre-refactor `transformers==4.46.3`. Always run `python src/verify_colpali.py` first — it asserts *no re-init warning* and retrieval-sanity ≥ 8/10 before any ColPali embedding is trusted.
- Encoding **ColQwen2.5** needs `transformers==4.47.1`.
- All scripts pin `OMP_NUM_THREADS=1` / `torch.set_num_threads(1)` (they will be >1000× slower otherwise on many-core hosts). Keep this.

Quick sanity check (no data needed):
```bash
PYTHONPATH=src python tests/test_marginmerge.py    # synthesis invariants
PYTHONPATH=src python tests/test_factorial.py      # factorial-core invariants
PYTHONPATH=src python tests/test_gapcover.py        # coverage/anchor invariants
```

---

## Reproducing the paper

Set once:
```bash
export PYTHONPATH=$PWD/src
```

### 1. Data (frozen embedding caches)
```bash
# ColQwen2.5 → data/cache_colqwen/<slice>/{qs,ps,meta}
python src/encode.py         arxivqa docvqa infovqa flickr
python src/encode_slices.py  tatdqa tabfquad
# ColPali (in the colpali env) → data/cache_colpali/...
python src/verify_colpali.py
python src/encode_colpali.py arxivqa docvqa infovqa flickr tatdqa tabfquad
```
Datasets are pulled from the ViDoRe / HF hubs (`vidore/arxivqa_test_subsampled`, `…/docvqa_test_subsampled`,
`…/infovqa_test_subsampled`, `vidore/tatdqa_test`, `vidore/tabfquad_test_subsampled`,
`nlphuji/flickr_1k_test_image_text_retrieval`).

### 2. Prototype bank (coverage anchors need it)
```bash
python src/build_query_prototypes.py --seed 42 --out outputs/bank_seed42.pt
```
Built **only from training-query embeddings**; held-out (eval) queries are excluded and `validate_no_leakage`
fails loudly on any overlap. Held-out protocol = 30% of documents (`RandomState(0)`) + all their queries,
document-disjoint from training.

### 3. Train MarginMerge (reported config = coverage anchors + learned synthesis + full objective)
```bash
python src/train_marginmerge.py --bank outputs/bank_seed42.pt --seed 42 \
       --out outputs/marginmerge_seed42.pt
```

### 4. Tables 1 & 3 — MarginMerge vs merging / response-centroid / full index
```bash
# 5% (repeat with MM_RHO=0.10 / 0.20 for the 10%/20% columns and Table 3 transfer)
MM_RHO=0.05 python src/eval_marginmerge.py --bank outputs/bank_seed42.pt \
       --ckpt outputs/marginmerge_seed42.pt --table9 \
       --slices arxivqa docvqa infovqa tatdqa tabfquad flickr
```
Baseline rows of Table 1 come from `dse_baseline.py`, `hpc_eval.py`, `sap_extract*.py`+`sap_eval.py`.

### 5. Table 2 — selection baselines
```bash
python src/baselines_memory.py arxivqa docvqa infovqa tatdqa tabfquad flickr
python src/baselines_aggregate.py         # Pareto figure + comparison table
```
(random / k-center / importance-prune / top-likelihood / ToMe-merge / int8; MarginMerge from step 4.)

### 6. Table 4 — controlled anchor × synthesis × loss factorial
```bash
python src/factorial_run.py --phase A       # fast diagnostic (arxivqa/flickr/tabfquad, 5%, seed 42)
python src/factorial_run.py --phase B       # confirmatory: 6 datasets × {5%,10%} × seeds {42,43,44}
python src/factorial_analyze.py --phase B    # → summary_mean_std / bootstrap_ci / pairwise_deltas / report.md / LaTeX
python src/factorial_anchor_overlap.py --phase B
```
`factorial_run.py` is **idempotent** (skips existing checkpoints and per-run JSONs), so interrupted runs resume.

Switch backbone by pointing the cache/bank at the ColPali artifacts:
```bash
CATTS_CACHE=data/cache_colpali BANK=outputs/bank_colpali_seed42.pt python src/factorial_run.py --phase B
```

---

## Included results (`results/`)
Aggregated numbers from our runs (the raw per-run dumps and `.pt` checkpoints are not committed):
- `factorial_summary_mean_std.csv`, `factorial_bootstrap_ci.csv`, `factorial_pairwise_deltas.csv`,
  `factorial_anchor_overlap_diagnostics.csv`, `report.md`, `table_factorial_rho{5,10}.tex` — Table 4 + analysis.
- `TABLE1_colpali_final.{md,csv}` — the ColPali Table-1 compression landscape.

**Headline of the factorial (matches the paper):** learned representative synthesis is the dominant, robust
source of gain (+0.056 avg over anchors, significant on all six datasets); coverage-aware anchors help most
when the representative is fixed but become largely redundant once synthesis is learned (interaction ≈ −0.04);
and margin matching ≈ absolute score reconstruction (Δ < 0.005) — so the ablation does **not** claim the margin
objective as a key ingredient.

## Configuration knobs
Scripts read these environment variables (defaults in parentheses):
`CATTS_CACHE` (`data/cache_colqwen`), `BANK` (`outputs/bank_seed42.pt`), `FACT_OUT` (`outputs/factorial`),
`CATTS_MM_OUT` (`outputs/marginmerge`), `MM_RHO` (`0.05`), `HF_HOME` (`~/.cache/huggingface`).

## Citation
```bibtex
@inproceedings{marginmerge2027,
  title     = {Coverage Matters: MarginMerge for Compressing Multi-Vector Visual Document Retrievers},
  author    = {Anonymous submission},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2027}
}
```

## License
MIT — see [LICENSE](LICENSE).
