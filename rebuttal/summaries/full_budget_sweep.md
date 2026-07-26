# Full budget sweep (Task: `full_budget_sweep`)

Reliability frontier over the complete budget sweep b ∈ {4, 6, 8, 10, 12, 14, 16, 20}
(subset sizes 24–120), balanced generator on the k=6 K-Means regimes (seed 0),
M = 400 candidates per budget, Zip-NeRF audit population of 3,521 scenes.
Joint pass event: |dPSNR| ≤ 0.5 dB, |dSSIM| ≤ 0.01, |dLPIPS| ≤ 0.01, and
two-sample KS ≤ 0.14 per metric vs the full-population empirical CDFs.
Wilson LCB at 95% (z = 1.959964).

## Headline numbers

| Scenes | b | Pass | Rate | Wilson LCB |
|---|---|---|---|---|
| 24 | 4 | 26/400 | 0.0650 | 0.0447 |
| 36 | 6 | 58/400 | 0.1450 | 0.1139 |
| 48 | 8 | 88/400 | 0.2200 | 0.1822 |
| 60 | 10 | 92/400 | 0.2300 | 0.1914 |
| 72 | 12 | 95/400 | 0.2375 | 0.1984 |
| 84 | 14 | 101/400 | 0.2525 | 0.2124 |
| 96 | 16 | 113/400 | 0.2825 | 0.2406 |
| 120 | 20 | 120/400 | 0.3000 | 0.2572 |

- The frontier is monotone non-decreasing in both empirical pass rate and LCB
  across the full sweep, including the newly evaluated budgets 24, 36, and 84.
- The smallest budget whose LCB meets the p_min = 0.08 reliability target is
  **36 scenes (LCB 0.1139)**; 24 scenes falls below it (LCB 0.0447).
- Validation: the shared budgets reproduce the paper values exactly
  (48 scenes → 88/400, LCB 0.1822; 96 scenes → 113/400, LCB 0.2406), because
  per-budget RNG streams are `default_rng(seed + b)` and independent of which
  budgets are in the sweep list.

## Caveats

- The effective audit population is 3,521 scenes, not 3,522: one descriptor
  scene (`dish_1563900172`) has no Zip-NeRF log entry. This matches the
  population used for the paper's exported reference values.
- Budgets 24/36/84 were previously only evaluated under the older
  `budget_sweep_k6_auto_v2` protocol; these rows supersede those numbers under
  the current `strong_accept` protocol.
- Seeds: global seed 0; per-budget effective seed = 0 + b (balanced mode),
  identical to `generate_strong_accept_results.py`.

Artifacts: `rebuttal/full_budget_sweep_frontier.csv`,
`rebuttal/full_budget_sweep_trials.csv` (3,200 trial rows),
best-candidate manifests under `rebuttal/manifests/`.
