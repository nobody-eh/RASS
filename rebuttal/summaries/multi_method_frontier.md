# Multi-method frontier (Task: `multi_method_frontier`)

Reliability frontier on the 3,473-scene common cross-method intersection.
Formal joint event: Zip-NeRF full image (PSNR, SSIM, LPIPS), Feature-Splatting
full image (PSNR, SSIM, LPIPS), and Instant-NGP full image (PSNR, SSIM) must
each pass every per-metric mean tolerance (0.5 dB / 0.01 / 0.01) and
two-sample KS ≤ 0.14 against their own reference population on the
intersection. Instant-NGP object-centric is reported as a diagnostic only and
is excluded from the formal event. Balanced generator on the k=6 regimes
restricted to the intersection, M = 400, seed 0, Wilson LCB at 95%.

## Headline numbers

| Scenes | Joint pass | Joint LCB | Zip-NeRF | Feat-Splat | INGP-FI | INGP-OC (diag) |
|---|---|---|---|---|---|---|
| 24 | 1/400 | 0.0004 | 26 | 19 | 46 | 30 |
| 36 | 4/400 | 0.0039 | 49 | 56 | 136 | 70 |
| 48 | 6/400 | 0.0069 | 73 | 61 | 160 | 82 |
| 60 | 16/400 | 0.0248 | 92 | 90 | 204 | 100 |
| 72 | 21/400 | 0.0346 | 101 | 88 | 209 | 110 |
| 84 | 19/400 | 0.0306 | 105 | 107 | 227 | 130 |
| 96 | 32/400 | 0.0572 | 119 | 99 | 256 | 161 |
| 120 | 26/400 | 0.0447 | 116 | 91 | 270 | 155 |

- **The multi-method budget does jump, and the p_min = 0.08 target is not
  reached by any budget up to 120 scenes.** The best joint LCB is 0.0572 at
  96 scenes. Reported as-is per the honesty convention; the rebuttal text
  should use the "target not reached up to 120 scenes" framing.
- Per-method pass events are positively correlated but far from perfectly
  aligned (at 96 scenes the independence product 0.2975 × 0.2475 × 0.64 ≈
  0.047 vs observed joint 0.080): a candidate that fits one method's
  distribution tends to fit the others, yet the joint event remains rare.
- Per-method pass rates at 48 scenes: Zip-NeRF 73/400, Feature-Splatting
  61/400, Instant-NGP-FI 160/400 (only two metrics constrained).

## Caveats

- Reference populations are per-method empirical CDFs on the same 3,473-scene
  intersection, so single-method rates here are not directly comparable to the
  3,521-scene Zip-NeRF frontier (Task `full_budget_sweep`): at 48 scenes,
  Zip-NeRF alone passes 73/400 on the intersection vs 88/400 on the full
  population, and the candidate sets themselves differ because the sampling
  pool differs.
- Instant-NGP PSNR uses the `psnr avgmse` column, matching the canonical
  column-priority rule in the existing audit code.
- Validation: the replicated generator + evaluator reproduces 88/400 at b=8
  on the 3,521-scene Zip-NeRF population before being applied to the
  intersection. Seeds: global seed 0; per-budget effective seed = 0 + b.

Artifacts: `rebuttal/multi_method_frontier.csv`,
`rebuttal/multi_method_frontier_trials.csv` (3,200 trial rows with per-method
per-metric gaps and KS distances).
