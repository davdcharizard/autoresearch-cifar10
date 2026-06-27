# EXP-045: 64/192/256 gate-first — the last kernel-lattice capacity point

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-045.md
- **Plan**: plans/plan-045.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-045
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed (pre-registered GATE_KILL branch (iii) — dt screen resolved decisively; no relaunch per plan, no lattice points remain)

## Implementation Notes

### Summary
Identical mechanics to EXP-044, per plan Milestone 1: `STAGE_WIDTHS = (64, 192, 256)` constant (comment notes the baseline and the lattice rationale), `ResNet.__init__` widths-tuple signature, construction + print line updated. No other line of train.py touched. CPU sanity suite (`/tmp/exp045_sanity.py`) passed all four checks: params exactly **5,392,714** (analytic delta +1,105,920 conv / +768 BN over baseline 4,286,026); forward (4,3,32,32)→(4,10) finite; pad shortcuts 128 at stage2 transition (192−64) and 64 at stage3 (256−192), identity elsewhere; 2-step train smoke 3.19→1.22. Composite launcher `/tmp/exp045_composite.sh` derived from the validated exp044 script with exactly three threshold edits (gate 28→31ms, contention floor 28→31), confirmed by diff.

### Surprises & Discoveries
- None — second consecutive use of this implementation pattern; everything matched analytic predictions.

### Decisions
- Gate threshold 31ms per plan: off-rung, between the dense-law pass population (~28) and the EXP-044 mispricing population (≥35); also the epoch-arithmetic boundary (~100 epochs projected at 31ms).

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: background task bzzehsl3a (/tmp/exp045_composite.sh); gate watcher task bwmck9svy
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: GATE_KILLed (pre-registered dt screen, exit 47)
- **Started**: 2026-06-10 21:59:45 (gates clear poll 1: apps=0, load=17)
- **Ended**: 2026-06-10 ~22:01:05 (killed at watchdog tick 5, ~80s into run)

Description:
- Single gated run of the 64/192/256 net under the unchanged certified recipe — the last point on the H20 fast-kernel lattice ({64,128,192?,256} permutations, every other option closed by the cliff/alignment/starvation laws). Dense-law projection 27.7ms → ~112 epochs. Pre-registered branches: (i) best ≥ 96.81 → improvement (replicate pair first if 96.70–96.80); (ii) plateau ≈ mean − deficit (~96.1–96.3), family test_loss → capacity level-saturated, class closed statistically; (iii) GATE_KILL D0 > 31ms → 192 misprices, fast lattice = {64,128,256} exactly, verdict invalid, class closed on hardware grounds.

Observations:
- Gates clear on poll 1 (apps=0, load=17). Params printed **5,392,714** as predicted; 97 batches/epoch (source: run.log header).
- **GATE_KILL at tick 5**: windows **33.6 / 33.4 / 33.4 ms** → D0 = 33.4ms > 31ms; projected ~93 epochs (source: task bzzehsl3a output). Printed dt agrees (34ms uniform, img/s ~15,100); host clean post-kill (zero GPU-0 apps, load 12.2) — true kernel cost, not contention.
- **The result REFINES the EXP-044 law**: 192 = 3×64 IS a 64-multiple, yet it misprices identically — so "64-alignment" is not the mechanism. The fast set among all measured widths is exactly the powers of two: {64, 128, 256} fast (22.4ms); {80, 160, 192, 288, 320} slow. EXP-034's 48/96/192 fallback (27.4ms, all non-powers-of-2) is consistent.
- **Second flat tier observed**: 160 at +18% FLOPs → 32.4ms; 192 at +40% FLOPs → 33.4ms — off-lattice stage-2 widths land on a near-flat ~33ms tier, mirroring the flat 54ms tier above 256 (EXP-040: 288 and 320 identical).
- Training healthy pre-kill (loss falling, ep1–10 best 77.47 — incidentally byte-similar to EXP-044's 77.48 at ep10; the kill is purely the throughput screen).

Key Metrics:
- D0 windowed dt: 33.4ms (33.6/33.4/33.4 over steps ~300–1000) → ~93 projected epochs = starvation regime (source: bzzehsl3a output)
- num_params: 5,392,714 (printed, matches analytic) (source: run.log L2)
- best_test_acc: N/A — killed at ~11% per pre-registered screen; metric NaN per plan branch (iii)

## Verification Results

### Conditions Checked

- **Integrity pre-condition / dt gate**: FAILED BY DESIGN — D0 = 33.4ms > 31ms on a provably clean host (dual gates clear at launch, load 12.2 / zero GPU-0 apps at decision, three consistent windows, printed dt agreeing). Pre-registered branch (iii) from brainstorm-045/plan-045: 192 misprices → the H20 fast lattice is exactly {64,128,256}; NO relaunch (no lattice points remain).
- **Condition 1 (best_test_acc ≥ 96.81)**: skipped — not evaluable, run killed at ~11% progress.
- **Conditions 2–3**: skipped — aborted after pre-condition branch.
- **Verdict basis (for analyze)**: `invalid`, metric NaN, per plan-045 Abort Criteria.

### Informational Metrics

- Not collected (conditions did not pass). Hardware data recorded instead: 64/192/256 prices at 33.4ms; combined with EXP-044, off-lattice widths form a flat ~33ms tier; fast set = powers of two {64,128,256}.

## Errors & Dead Ends

### 2026-06-10 — GATE_KILL: 192 misprices too — the law is powers-of-two, not 64-multiples
- Error: `GATE_KILL: D0=33.4ms > 31ms (projected 93 epochs < ~111)` — exit 47 from /tmp/exp045_composite.sh
- Root cause: the fast-kernel path on H20+compile+channels_last+bf16 exists only at power-of-two channel counts. 192 (= 3×64) was the test case separating "64-multiple" from "power-of-2" explanations of EXP-044 — it misprices (+11.0ms vs +5.3 dense-law), settling the law: fast = {64,128,256}; everything else lands on flat slow tiers (~33ms below 256, ~54ms above).
- Source: bzzehsl3a output (windows 33.6/33.4/33.4); run.log step prints (34ms uniform, clean host)
- Do NOT retry: ANY stage width that is not a power of two on this stack. On-lattice capacity-increasing permutations (64/256/256 ≈ +88% FLOPs → ~34ms dense-law; 128/128/256 ≈ +100% → ~36ms) are starvation-priced BEFORE gating — the asymmetric/within-cliff capacity class is now closed in full: 64/128/256 is the unique fast increasing triple with viable epoch arithmetic.

## Human Notes

> (none — autopilot)
