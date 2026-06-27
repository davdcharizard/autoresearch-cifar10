# EXP-064: ACNet asymmetric convolution blocks — probe-gated NO LAUNCH

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md
- **Plan**: plans/plan-064.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-064
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented plan-064 M1 in full on train.py: a ~25-line `ACB` module (3x3+BN ∥ 1x3+BN ∥ 3x1+BN summed; paddings (1,1)/(0,1)/(1,0); shared stride; bias=False), swapped into all 19 conv+BN sites (BasicBlock acb1/acb2 and the stem) with everything else byte-identical — a +32/−11 line diff. CPU sanity 6/6: forward shape, 19 ACB sites, strided shape equality, **fold-equivalence to 2.15e-06** (eval-mode ACB == single folded 3x3 conv — the eval'd function IS the plain-net function class), loss decreasing, lr_at unchanged. NUM_PARAMS_ACB = 7,149,002 train-time params pinned. M2 probe with internal baseline control ran at the cleanest conditions (apps=0, load 14.9) and returned NO LAUNCH; the experiment terminated at its pre-registered primary falsification point with zero charged seconds.

### Surprises & Discoveries

- **The toll is ~2×, not the few-ms estimate**: control B = 22.36ms (mid-family-band — probe verifiably clean), ACB net P = 43.15ms, toll ratio 1.930, P_norm = 43.23 vs ≤ 26.0 required. The +20.8ms toll for +76 small-kernel launches (2 convs + 2 BNs × 19 sites) is FAR above linear launch pricing — the 1D convs (1x3/3x1) appear to land on slow kernel implementations, consistent with the off-lattice/odd-shape slow-tier law (EXP-044/045: unusual shapes get one slow implementation per regime, FLOPs second-order).
- **Minimum-variant diagnostic (uncharged)**: DBB-lite (3x3 ∥ 1x1 only — the fewest extra launches ANY reparameterization can have: +1 conv +1 BN per site) timed 28.61ms at load 8.9 → P_norm ≈ 28.7 → ~107 epochs → dilution ≈ −0.45 → required realized gain ≈ 0.69, while single-branch ablations in the ACNet/DBB papers deliver LESS than the full-block gains (+0.35–1.11 full). Even the cheapest family member fails its inequality → the structural-reparameterization FAMILY is closed on launch-pricing grounds, not just ACB.
- Two-net probe with internal control worked exactly as designed: B sat at 22.36 inside [21.5, 25.0], making the P reading load-attributable without any band gymnastics.

### Decisions

- Ran the DBB-lite pricing diagnostic after the NO-LAUNCH verdict (uncharged, ~3 min) to upgrade the closure from variant-level to family-level — mirrors the EXP-063 eager-attribution pattern. No charged time used; the pre-registered criterion was applied verbatim.
- train.py changes remain uncommitted on autoresearch/exp-064 and will be discarded at analyze-phase housekeeping.

## Experimental Adjustments

- **Sanity script NameError fix** (`nn.conv =` typo → plain `ref =`): one-line fix on first run; checks then 6/6. (ref: /tmp/exp064_sanity.py first invocation)

## Run Log

### Run 1 (M2 probe — THE LAUNCH GATE; no charged run ever started)

Metadata:
- **Job ID**: N/A (foreground probe, GPU 0)
- **Log file(s)**: stdout only (no run.log was ever created — train.py was never launched)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 11:05
- **Ended**: 2026-06-11 11:12

Description:
- /tmp/exp064_gpu_probe.py: gate-checked GPU 0 (apps==0, load 14.9), built + compiled + warmed BOTH the plain baseline net (control) and the ACB net, timed 40 full steps each with pinned H2D included. Pre-registered criterion: LAUNCH iff B ∈ [21.5, 25.0] AND P_norm = 22.4×P/B ≤ 26.0. Expected per hypothesis: P_norm ≤ ~26 (toll a few ms).

Observations:
- Gate: apps=0, load1=14.9; control B = 22.36ms ∈ [21.5, 25.0] and mid-family-band → probe clean and internally validated (source: probe stdout)
- ACB P = 43.15ms; toll_ratio = 1.930; P_norm = 43.23 — criterion fails by 66%, not marginal (source: probe stdout)
- Warmup: B compile 11.5s, P compile 36.4s (3× — 57 convs vs 19 to compile; informational)
- VRAM peak 3,195MB across both builds (source: probe stdout)
- **NO_LAUNCH** printed per pre-registered criterion (source: probe stdout)
- Diagnostic /tmp/exp064_dbblite_diag.py (uncharged): minimum reparam variant (3x3 ∥ 1x1) = 28.61ms at load 8.9 → required gain ≈ 0.69 > single-branch published ablation gains → family-level cost-closure (source: diagnostic stdout)

Key Metrics:
- B_ms (control): 22.36 (source: probe stdout)
- P_ms (ACB): 43.15; toll_ratio: 1.930; P_norm: 43.23 (source: probe stdout)
- DBBlite_ms (min variant): 28.61 (source: diagnostic stdout)
- best_test_acc: NaN — no charged run (pre-registered branch (ii))

## Verification Results

### Conditions Checked

- **M2 pre-registered launch criterion (gates everything)**: FAILED — P_norm = 43.23 vs ≤ 26.0 required (B control valid at 22.36). Per plan-064 branch (ii), verdict `invalid` with metric NaN; Conditions 1–3 **skipped** (no charged run, no run.log).
- Integrity of the gate decision: the internal control (B mid-family-band at the same load in the same session) rules out load-inflation by construction; the criterion was applied verbatim as pre-registered; arithmetic re-checked (at 43.23ms → 71 epochs → dilution −0.95 → required gain 1.19 > published max 1.11).

### Informational Metrics

- Toll ratio 1.93 (full ACB), 1.28 (minimum DBB-lite) — the family pricing curve.
- ACB compile warmup 36.4s (3× baseline's ~11.5s) — informational for any future multi-conv-variant probes.

## Errors & Dead Ends

### 2026-06-11 — Structural reparameterization family closed on launch pricing
- Error: `NO_LAUNCH — P_norm 43.23 vs ≤ 26.0 (full ACB); minimum variant DBB-lite 28.61ms still requires gain ≈ 0.69 > its published ablation band`
- Root cause: on this launch-bound box, each extra branch prices at kernel launches with FLOPs second-order, and the 1D convs (1x3/3x1) additionally land on slow odd-shape kernel implementations (EXP-044/045 law). Per-step toll: +6.25ms for the cheapest possible branch (1x1+BN), +20.8ms for the full ACB.
- Source: /tmp/exp064_gpu_probe.py stdout; /tmp/exp064_dbblite_diag.py stdout
- Do NOT retry: any train-time multi-branch conv reparameterization on this stack (ACB, DBB, RepVGG-style, any branch subset) — the minimum-launch variant already fails its gain-vs-dilution inequality. Partial-site variants (e.g., stage-3 only) inherit proportionally smaller gains with proportionally smaller tolls — same inequality, no escape (gain and toll both scale with site count).

## Human Notes

> (none — autopilot)
