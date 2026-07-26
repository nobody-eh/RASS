# Method logs (rebuttal release)

Per-scene metric outputs only; no dataset images or inputs are redistributed.

- `nutrition5k_nerfacto_metrics.csv` (per-scene PSNR/SSIM/LPIPS) and
  `nutrition5k_bionerf_metrics.csv` (fine-network metrics): additional
  per-scene method logs for the Nutrition5k-derived population, from the
  project's recorded training outputs (nerfstudio `ns-eval`; sources
  anonymized). One nerfacto scene was evaluated from a 28k/30k-step
  checkpoint and is flagged in its `note` column.
- `dl3dv_nerfacto/`: 140 per-scene DL3DV-Benchmark metric JSONs generated
  by the authors (ns-train nerfacto defaults, seed 0, downscale 4/960P);
  provenance embedded per file. DL3DV itself is licensed CC BY-NC 4.0 with
  its own terms of use; only our derived metrics are released here.
- `dl3dv_descriptors.csv`: 140-scene descriptor table used for the k=4
  regime clustering (E13).
