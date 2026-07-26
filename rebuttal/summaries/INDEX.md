# Rebuttal results index (NeurIPS 2026 submission 675)

Machine-readable results: `rebuttal/rebuttal_results.json` (one key per
task, all seeds recorded). Each summary below is self-contained.

| Task | Key | Summary | Status | One-line result |
|---|---|---|---|---|
| P0 validation | `P0` | P0.md | COMPLETE, 20/20 checks | Pipeline reproduces every paper number exactly; INGP PSNR column convention documented |
| P1 joint event | `E1` | E1.md | COMPLETE | RASS-96 passes all 16 constraints; RASS-48 passes 14/16 (only FS-LPIPS, already disclosed in paper); joint LCB target unreached ≤120sc |
| P2 ranking | `E2` | E2.md | COMPLETE | Ordering preservation is FREE (variant A ≡ E1 at every budget; 100% conditional) |
| P3 sensitivity | `E3` | E3.md | COMPLETE | Tolerances 4.5–8% of std, below every method gap; budget stable 36–48 near defaults; subset non-unique (Jaccard 0.005) |
| P4 regime fidelity | `E4` | E4.md | COMPLETE (negative) | Per-regime means uncertifiable ≤360sc (0/400 at c=1) — statistical inevitability, keep guarantees global |
| P5 FL-36 | `E5` | E5.md | RESOLVED via labeled reimplementation | Original FL artifacts unrecoverable (search documented); FL-36′ certified+packaged; FL unstable (overlap 0.038), breaks PSNR ordering |
| P6 descriptors | `E6` | E6.md | COMPLETE | DINOv2 gives the SAME 36-scene budget; full 57-D beats every group ablation |
| P7 DL3DV transfer | `E7` | E7.md | GATE FAILED → superseded by E13 | Per-scene DL3DV-140 logs not published (benchmark-meta.csv = labels only); logs since generated locally, see E13 |
| — extended audit | `E8` | E8.md | STRESS TEST ONLY | 0/400 under equal allocation on skewed coverage = audit correctly detects population shift; fair rerun pending full training |
| — available-data | `E9` | E9.md | COMPLETE (deadline product) | 4-method certification at 72sc (LCB .105), 5-method at 120sc (LCB .148), proportional allocation, RASS-96 passes 4-method event |
| P10 proportional E1 | `E10` | E10.md | COMPLETE | Proportional allocation reaches the 0.08 LCB target at 72sc under default tolerances (equal never does ≤120sc); E1 frontier reproduced as baseline |
| P11 dispersion-matched | `E11` | E11.md | COMPLETE (post-hoc label mandatory) | Dispersion-matched taus: 0.08 at 60sc equal / 48sc proportional; RASS-96 passes all 16; RASS-48 still fails only FS-LPIPS |
| P12 why-48 | `E12` | E12.md | COMPLETE, verdict (a) | Selection-time sweep evaluated 36 and it failed its recorded rule (LCB 0.041<0.08); 48 smallest passing; paper frontier starts at 48 by design |
| P13 DL3DV logs+audit | `E13` | E13.md | COMPLETE (in window) | 140/140 nerfacto logs generated locally; audit transfers: proportional certifies 32sc at p_min=0.08 (LCB .123), equal-allocation bias replicated on second dataset |
| P14 single-method prop. | `E14` | E14.md | COMPLETE | Paper's Zip-NeRF frontier under proportional: 0.08 budget unchanged (36sc), comparable at 48 (LCB .168 vs .182), dominates beyond (0.20: 60 vs 96sc); RASS-48/96 unaffected |
| P15 contract evidence | `E15` | E15.md | COMPLETE, verdict (a) | "Declared before results" holds for the core contract (timestamped trail packaged); proportional/uniform companions + budget grid labeled post hoc; 9 event configs exported |
| P16 repo sync | `E16` | E16.md | COMPLETE, commit d50bf31 | 223 files pushed to anonymous repo; 33/33 release claims resolve to live paths; DL3DV license gate PROCEED (metrics only); anonymity grep 0 hits |
| P17 DL3DV 3DGS | `E17` | E17.md | RUNNING (camera-ready) | splatfacto DL3DV-140 queued on local GPU; two-method audit on completion; nothing in the rebuttal depends on it |

## For the drafting session — DO and DO NOT

- DO quote: E1 (formal multi-method), E9 (4/5-method extension), E2, E3, E6
  as the positive results; E4 and E8 with their recorded framings (honest
  negatives turned into design justifications); E13 as completed-in-window
  DL3DV transfer (single method, caveats in E13.md); E11 only with its
  "post hoc" label; E12 verdict-(a) reply for the why-48 question.
- DO NOT quote: the `multi_method_frontier` and `full_budget_sweep` keys'
  multi-method numbers where they overlap E1 (superseded: different INGP
  PSNR column); E8 pass rates as if they were the multi-method result;
  FL-36′ as "the paper's FL-36" (it is a labeled reimplementation).
- Population facts to state once: effective Zip-NeRF audit set is 3,521
  (documented off-by-one; paper's own exports use it); cross-method
  intersection 3,473; E9 subpopulations I4=2,915 / I5=2,228 as of
  2026-07-24 (nerfacto includes 188 validated local GPU evals; one scene
  at 28k/30k steps, flagged).
- Reviewer mapping: p5AG W1/Q1→E1, W2/Q2→E7+E6, W3→E4, W4→E3+E6, Q3→E5;
  rTZt W1→E7, W2→E1, W3→E3; 6aTr Q2→E3+E6, Q3→E2, W3→E5; MU2f→E1;
  6aTr Q2 (methods)→E9.
