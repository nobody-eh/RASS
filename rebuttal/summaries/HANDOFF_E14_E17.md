# Handoff note: E14–E17 for the drafting session

Read this before writing any rebuttal text that touches E14–E17.
Machine-readable results: `rebuttal_results.json` keys `E14`–`E17`.

## Quotable now

**E14 — the paper's own single-method frontier under proportional
allocation.** Answers the obvious follow-up to the "equal allocation is
biased" argument: does the headline Zip-NeRF frontier that produced RASS-48
and RASS-96 change too? Verified against both paper references before the
run (88/400 at 48 scenes, 113/400 at 96). Result: the 0.08 budget does
**not** move (36 scenes either way), 48 scenes is comparable (LCB 0.168
proportional vs 0.182 equal), and proportional dominates from 60 scenes up,
pulling the 0.20 operating point from 96 to 60 scenes. A 345-character
hold-ready reply is in `E14.md` and in the JSON.

**E15 — evidence audit of the "contract declared before results" claim.**
Verdict **(a)**: supported for the core contract, with a timestamped trail
packaged in `rebuttal/dl3dv_contract_declaration.json`.

**E16 — repository sync.** Commit `d50bf31` (+ `80247a3`) on the anonymous
remote; 223 files; 33/33 release claims resolve to live paths; DL3DV license
gate PROCEED (derived metrics only, no dataset inputs); anonymity grep zero
hits.

## Three constraints — the first two are blocking

**1. BLOCKING (E15). Do not claim the two post-hoc items as pre-declared.**
The declaration covers k=4 regimes, the E3c dispersion-rule tolerances, the
size-dependent KS guardrail, the balanced generator, M=400 and seed 0. It
does **not** cover (i) the proportional and uniform companion frontiers,
added after the balanced frontier was computed, or (ii) the concrete budget
grid. Both are labeled post hoc in the declaration package and must stay
labeled wherever used. Note also that the supporting evidence is local
(session transcript, file mtimes, an `"audit": null` placeholder) — there
was no externally anchored timestamp such as a pushed commit at declaration
time. Say so if a reviewer presses.

**2. BLOCKING (E17). Nothing may be claimed from E17.** The splatfacto run
is in progress (4/140 at last check) and `E17.audit` is null. It is
camera-ready material. The rebuttal's existing commitment — 3DGS as a second
DL3DV method in the camera-ready — is already the right and only claim.
Early two-scene results and the measured rate live in `E17.md` for internal
planning, not for posting.

**3. Re-run E16's claims checklist against the FINAL posted text.** The
33-row table maps claims as staged in the summaries and hold-ready replies.
Any "released / ships / we release / audited configuration" phrasing added
during drafting needs its own live path, and a claim without one is a STOP.

## Carried forward from the earlier handoff

Still binding: quote E11 only with its "dispersion-matched operating point,
declared post hoc during the discussion period" label; never call FL-36′ the
paper's FL-36 (it is a labeled reimplementation); state the effective
audit population as 3,521 with the documented off-by-one; and E13's DL3DV
result carries its own caveat list (single method, self-generated logs,
loose KS at N=140 so the binding constraints are the dispersion-matched
means). `INDEX.md` holds the full DO / DO-NOT-quote rules and the reviewer
mapping.

## Also released since the last bundle

`rebuttal/event_configs/` (nine versioned event configurations, schema
`rass-event-config/1.0`), `rebuttal/dl3dv_audit_card.json`,
`rebuttal/dl3dv_contract_declaration.json`,
`rebuttal/selection_artifacts/` (the recorded artifacts backing E12's
why-48 answer), and `rebuttal/method_logs/` (DL3DV per-scene metrics plus
metrics-only Nutrition5k nerfacto/bionerf CSVs).
