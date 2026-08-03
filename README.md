# MarginMerge: Coverage-Aware Compression of Multi-Vector Visual Document Retrievers

Official code for **“Coverage Matters: MarginMerge for Compressing Multi-Vector Visual Document Retrievers.”**

Multi-vector visual retrievers (ColPali, ColQwen) store hundreds–thousands of patch vectors per page. MarginMerge compresses each document **once at indexing time** into `k = ⌈ρn⌉` synthetic representatives, leaving the standard MaxSim retrieval interface unchanged. Across six datasets on ColQwen2.5 and ColPali it preserves 97–99% of full-index nDCG@5 at 5–10% retention.

![MarginMerge method overview](assets/method.png)

Offline, per document: (1) select **coverage-aware anchors** that span complementary query-relevant directions, (2) assign each patch to its nearest anchor, (3) synthesize one representative per cluster with a tiny shared MLP, (4) train that MLP by **ranking-margin distillation** against the frozen full index. Online retrieval is untouched — standard MaxSim over `k` vectors.

## Install

Two environments, because the two backbones need different `transformers`:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt              # core + ColQwen2.5 encoding (transformers 4.47.1)

python -m venv .venv-colpali && . .venv-colpali/bin/activate
pip install -r requirements-colpali.txt      # ColPali-v1.3 (colpali-engine 0.3.5 → transformers 4.46.3)
```

Pins matter: `transformers ≥ 4.49` silently re-initialises ColPali-v1.3's PaliGemma weights. Run `python src/verify_colpali.py` before trusting any ColPali embedding — it asserts no re-init warning and retrieval sanity ≥ 8/10.

`src/` uses flat imports, so set `export PYTHONPATH=$PWD/src` once. Quick check, no data needed:

```bash
PYTHONPATH=src python tests/test_marginmerge.py   # also test_gapcover.py, test_factorial.py
```

## Reproduce

```bash
# 1. frozen embedding caches
python src/encode.py arxivqa docvqa infovqa flickr
python src/encode_slices.py tatdqa tabfquad
python src/encode_colpali.py arxivqa docvqa infovqa flickr tatdqa tabfquad   # colpali env

# 2. prototype bank (training queries only, leakage-checked)
python src/build_query_prototypes.py --seed 42 --out outputs/bank_seed42.pt

# 3. train
python src/train_marginmerge.py --bank outputs/bank_seed42.pt --seed 42 \
  --out outputs/marginmerge_seed42.pt

# 4. main tables (repeat with MM_RHO=0.10 / 0.20)
MM_RHO=0.05 python src/eval_marginmerge.py --bank outputs/bank_seed42.pt \
  --ckpt outputs/marginmerge_seed42.pt --table9 \
  --slices arxivqa docvqa infovqa tatdqa tabfquad flickr

# 5. selection baselines (random / k-center / importance / top-likelihood / ToMe / int8)
python src/baselines_memory.py arxivqa docvqa infovqa tatdqa tabfquad flickr
python src/baselines_aggregate.py

# 6. anchor × synthesis × loss factorial
python src/factorial_run.py --phase B
python src/factorial_analyze.py --phase B
```

Datasets are pulled from the ViDoRe / HF hubs. `factorial_run.py` is idempotent, so interrupted runs resume.

Environment knobs: `CATTS_CACHE` (`data/cache_colqwen`), `BANK` (`outputs/bank_seed42.pt`), `FACT_OUT`, `CATTS_MM_OUT`, `MM_RHO` (`0.05`), `HF_HOME`. Switch backbone with `CATTS_CACHE=data/cache_colpali BANK=outputs/bank_colpali_seed42.pt`.

## Code map

| Paper | Eq. | Code |
|---|---|---|
| Coverage + submodular anchor selection | 4–5 | `gapcover.py` |
| Nearest-anchor assignment | 6 | `clustering.py` |
| Representative synthesis (weights, MLP) | 7–9 | `marginmerge.py` |
| Margin-distillation loss + hard negatives | 10–12 | `train_marginmerge.py` |

## Results

`results/` holds the aggregated numbers from our runs: `TABLE1_colpali_final.{md,csv}` for the ColPali compression landscape, and `factorial_summary_mean_std.csv`, `factorial_bootstrap_ci.csv`, `factorial_pairwise_deltas.csv`, `factorial_anchor_overlap_diagnostics.csv`, `report.md`, `table_factorial_rho{5,10}.tex` for the factorial. Raw per-run dumps and `.pt` checkpoints are not committed.

## License

MIT — see [LICENSE](LICENSE).
