# Drafting brief — NeurIPS 2026 submission 675 rebuttal

Everything needed to write the rebuttal. Attach
`rebuttal/rebuttal_handoff_675.zip` (240 KB) alongside this file; it contains
`rebuttal_results.json` (every key, every seed), all task summaries, the nine
versioned event configs, the DL3DV audit card and contract declaration, the
selection-time artifacts, and the 140 DL3DV per-scene metric logs.

Anonymous repository reviewers can inspect: `github.com/nobody-eh/RASS`
(current head `26f95c2`). Every "released" claim below resolves to a live
path there.

---

## 1. Hard rules — violating any of these is a defect

1. **Never claim E17 (3DGS/splatfacto) results.** The run is incomplete and
   its audit is null. The only permitted statement is the existing
   commitment: 3DGS as a second DL3DV method **in the camera-ready**.
2. **Never present these as pre-declared** (they are labeled post hoc in
   `dl3dv_contract_declaration.json`): the proportional and uniform
   companion frontiers in E13, and E13's concrete budget grid. The core
   DL3DV contract (k=4, dispersion-rule tolerances, size-dependent KS,
   balanced generator, M=400, seed 0) *was* declared first — verdict (a),
   evidence trail packaged.
3. **E11 requires its label on every use**: "dispersion-matched operating
   point, declared post hoc during the discussion period." Report the
   default contract's numbers first, always.
4. **Never call FL-36′ the paper's FL-36.** It is a labeled reimplementation;
   the original selection code and seeds were unrecoverable.
5. **State the population honestly**: effective Zip-NeRF audit set is
   **3,521** scenes (one descriptor scene has no log entry — documented
   off-by-one vs the stated 3,522; the paper's own exports already use
   3,521). Cross-method intersection is **3,473**. E9 subpopulations:
   I4 = 2,915, I5 = 2,228.
6. **Re-check every "released / ships / we release / audited configuration"
   phrase you write** against E16's 33-row claims table. A claim without a
   live repository path is a STOP.
7. Do not quote the deprecated `multi_method_frontier` or
   `full_budget_sweep` multi-method numbers where they overlap E1 — they
   used a different Instant-NGP PSNR column and are superseded.

---

## 2. Reviewer → evidence mapping

| Reviewer | Concern | Answer with |
|---|---|---|
| p5AG W1/Q1 | formal multi-method guarantee | E1 (+ E10 proportional) |
| p5AG W2/Q2 | generalization beyond Nutrition5k | **E13** (DL3DV, 140 scenes) + E6 |
| p5AG W3 | regime-level fidelity | E4 (honest negative) |
| p5AG W4 | tolerance justification | E3 + E6 |
| p5AG Q3 | facility location / FL-36 | E5 (FL-36′, labeled) |
| rTZt W1 | transfer to another dataset | **E13** |
| rTZt W2 | multi-method | E1 |
| rTZt W3 | sensitivity of tolerances | E3 |
| 6aTr Q2 | more methods | E9 (4- and 5-method) + E3/E6 |
| 6aTr Q3 | ranking preservation | E2 |
| 6aTr W3 | FL | E5 |
| MU2f | joint event | E1 |
| (anticipated) | "why 48 and not 36?" | **E12** |
| (anticipated) | "does proportional change your headline?" | **E14** |

---

## 3. Results to use, with the honest framing

**E1 — formal three-method joint event** (3,473 scenes, default tolerances,
KS ≤ 0.14, equal allocation, M=400, seed 0). RASS-96 passes all 16
constraints. RASS-48 passes 14/16, failing only Feature-Splatting LPIPS —
already disclosed in the paper. The joint LCB target is **not** reached at
any budget ≤ 120 under equal allocation (96 scenes: 30/400, LCB 0.053).
Report that plainly; E10 is the constructive answer.

**E10 — proportional allocation fixes it.** Same event, proportional-to-
regime-size allocation with largest-remainder rounding: **0.08 target reached
at 72 scenes, 0.20 at 120**, versus "not reached up to 120" for equal
allocation. This is the paper's own diagnosis of why equal allocation is
biased on regime-skewed populations.

**E2 — ranking preservation is free.** Variant A (joint event + sign
preservation of all six pairwise method gaps) is identical to E1 at every
budget; 100% conditional preservation. Ordering costs nothing.

**E3 — tolerances are calibrated, not arbitrary.** They sit at 4.5–8% of
per-scene standard deviations and below every cross-method gap. The
recommended budget is stable across 36–48 scenes near the defaults. Subsets
are non-unique (Jaccard 0.005), which is a feature: the audit certifies a
distribution of subsets, not one magic list.

**E4 — honest negative.** Per-regime means are uncertifiable at any budget
≤ 360 scenes (0/400 at c=1). This is a statistical inevitability at these
regime sizes, and it is *why* the guarantees are scoped globally. Present it
as a scoping justification, not a failure.

**E5 — FL-36′.** Original FL artifacts unrecoverable (search documented).
The labeled reimplementation is certified and packaged. Finding: FL is
unstable (overlap 0.038 across seeds) and breaks PSNR ordering — supports
the paper's choice to export the balanced subset.

**E6 — descriptors.** DINOv2 embeddings give the **same** 36-scene budget as
the 57-D hand-crafted descriptors, and the full 57-D set beats every group
ablation. The descriptor design is not load-bearing for the result.

**E9 — multi-method on available data.** 4-method certification at 72 scenes
(LCB 0.105) on I4 = 2,915; 5-method at 120 scenes (LCB 0.148) on I5 = 2,228;
proportional allocation. RASS-96 passes the 4-method event.

**E13 — DL3DV transfer, the strongest new result.** We generated all 140
per-scene nerfacto logs ourselves (DL3DV publishes none — verified three
times; `benchmark-meta.csv` is labels only). The protocol transferred
unmodified — descriptors, clustering, E3c tolerance calibration, KS
guardrail, audit machinery — and certifies a **32-scene subset (4.4×
reduction) at p_min = 0.08** under proportional allocation (62/400, LCB
0.123). Equal allocation never reaches 0.08 and *declines* past 48 scenes,
independently replicating the E10 bias diagnosis on a second dataset whose
largest regime holds 42% of scenes.
Mandatory caveats: single method; logs self-generated during the discussion
period; DL3DV chosen after submission; at N=140 the KS critical distances
are loose (0.19–0.49), so the binding constraints are the dispersion-matched
means; the proportional/uniform frontiers are post hoc (rule 2).

**E15 — provenance.** The DL3DV contract was declared 2026-07-24 22:53 UTC,
recorded in the repo 23 minutes later with `"audit": null`, ~2 h before the
first scene log and ~36 h before the first audit statistic. Evidence is
local (session transcript, file mtimes, the placeholder) — there was no
pushed-commit timestamp at declaration time. Say so if pressed. The
validation gate rejected a mis-wired population on its first run, which
demonstrates the gate was live.

**E16 — artifacts.** 223 files on the anonymous remote; 33/33 release claims
resolve; DL3DV license gate PROCEED (CC BY-NC + ToU permit releasing our
derived metrics, not dataset inputs — and no inputs are released);
anonymity scrub returned zero hits.

---

## 4. Verbatim hold-ready replies (use as-is or tighten)

**"Why 48 scenes when the audit passes at 36?" (E12, 346 chars)**

> The rule is recorded: the selection-time sweep
> (budget_recommendation.json) evaluated {12,24,36,48}; 36 scored LCB 0.041
> (<0.08), 48 was the smallest passing (LCB 0.081). The paper frontier
> evaluates budgets from 48 upward, hence 'lowest-cost evaluated'. The
> revision adds the current audit's 36-scene row (LCB 0.114) and reconciles
> the frontier.

**"Does proportional allocation undermine RASS-48/96?" (E14, 345 chars)**

> No. RASS-48/96 are export-audited scene lists, allocation-independent;
> equal allocation is the documented generator of record. Proportional keeps
> the 0.08 budget at 36 scenes, is comparable at 48 (LCB .168 vs .182), and
> dominates beyond (0.20 target: 60 vs 96 scenes). The revision reports both
> allocations and adopts proportional going forward.

**Dispersion-matched operating point (E11, 154 chars — needs the post-hoc label)**

> under it the three-method joint event gives 27/400 at 48 scenes and
> reaches the 0.08 LCB target at 60 scenes with equal allocation (48 with
> proportional).

---

## 5. Revision commitments that are safe to make

- Report both allocations throughout and adopt proportional as the
  recommended design going forward (E10, E13, E14).
- Add the 36-scene row to the frontier and reconcile the selection-time vs
  paper-frontier grids (E12).
- State the 3,521-scene erratum (already in the README and audit cards).
- Release the versioned event configs so every audited configuration is
  reproducible (E15/E16, already live).
- Add 3DGS as a second DL3DV method in the camera-ready (E17 — commitment
  only, no numbers).

---

## 6. What is deliberately absent

- **E7** is superseded by E13; the gate-failure framing ("logs not
  published") is still true and is *why* E13 exists.
- **E8** is stress-test material only, superseded by E9; its 0/400 result
  demonstrates the audit correctly detects population shift under equal
  allocation on skewed coverage. Its fair rerun is incomplete and is not
  rebuttal-critical.
- **E17** — see rule 1.
