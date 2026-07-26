# Additional per-scene method logs

- `nutrition5k_nerfacto_metrics.csv`, `nutrition5k_bionerf_metrics.csv`:
  additional per-scene method logs for the Nutrition5k-derived scenes
  (PSNR/SSIM/LPIPS; bionerf uses the fine-network metrics). Coverage as of
  2026-07-26; the E9 audits used the recorded 2026-07-24 snapshot sizes.
  Scenes still training are absent. One nerfacto scene evaluated from a
  93%-trained checkpoint is flagged in `note`.
- `dl3dv_nerfacto/`: 140 per-scene DL3DV-Benchmark metric JSONs generated
  by us (see provenance field in each file); `dl3dv_nerfacto_metrics.csv`
  is the consolidated table. Only our derived metrics are redistributed -
  no DL3DV images or inputs (per the DL3DV-10K license, CC BY-NC 4.0 +
  terms of use; metrics released for non-commercial research with
  attribution to DL3DV).

Source paths have been rewritten for anonymity; provenance beyond
"additional per-scene method logs" (machines, accounts) is intentionally
omitted.
