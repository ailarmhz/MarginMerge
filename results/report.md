# Factorial ablation report (phase B)

Backbone: frozen ColQwen2.5. Metric: nDCG@5, held-out Table-9 protocol. Bootstrap CIs over queries (B=2000). Effects at ρ=0.05. Non-significant if |Δ|<0.005 unless a paired CI excludes 0.

## 1. Main effects

### Anchor strategy (holding synthesis fixed), mean nDCG@5

| Synthesis | random | kcenter | coverage |
|---|---|---|---|
| fixed response centroid | 0.7914 | 0.8136 | 0.8303 |
| learned (margin) | 0.8665 | 0.8706 | 0.8667 |

### Learned synthesis vs fixed response centroid (holding anchors fixed)

| Anchor | response_centroid | learned(margin) | Δ (learned−fixed) |
|---|---|---|---|
| random | 0.7914 | 0.8665 | +0.0751 |
| kcenter | 0.8136 | 0.8706 | +0.0569 |
| coverage | 0.8303 | 0.8667 | +0.0365 |

## 2. Interaction (coverage−random gain: amplified by learning?)

- coverage−random under **fixed** response centroid: +0.0388
- coverage−random under **learned** synthesis (margin): +0.0002
- **interaction** (learned − fixed): -0.0387

## 3. Loss ladder (learned synthesis), mean nDCG@5 over anchors & datasets

| Loss | mean nDCG@5 |
|---|---|
| score_reconstruction | 0.8688 |
| margin | 0.8679 |
| full | 0.8682 |

## 4. Conclusions

- **Coverage-aware anchors independently useful?** yes (coverage−random under fixed synthesis = +0.0388).
- **Learned synthesis independently useful?** yes (mean learned−fixed over anchors = +0.0562).
- **Gain primarily from interaction?** interaction=-0.0387 → both independent and interaction.
- **Margin > absolute score reconstruction?** no (below 0.005) (margin−score_reconstruction = -0.0009).
