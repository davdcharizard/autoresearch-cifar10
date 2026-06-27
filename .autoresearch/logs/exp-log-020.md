# EXP-020: Projection shortcuts at stage transitions (ResNet option B, WRN-faithful)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-020.md
- **Plan**: plans/plan-020.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-020
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (run completed cleanly; necessary condition 1 not met — 96.58 < 96.81)

## Implementation Notes

### Summary

Exactly the plan's Milestone 1 diff, nothing else: in `BasicBlock.__init__`, the three option-A bookkeeping lines (`self.stride` / `self.need_pad` / `self.pad_channels`) replaced by a `self.shortcut` module — `nn.Sequential(nn.Conv2d(in, out, 1, stride=s, bias=False), nn.BatchNorm2d(out))` when `stride != 1 or in_channels != out_channels`, else `nn.Identity()`; in `forward`, the slice-and-pad branch replaced by `out += self.shortcut(x)`. Diff is +12/−8 lines confined to `BasicBlock`. Pre-validation (CPU): syntax OK; params exactly 4,327,754 (baseline 4,286,026 + 41,728); forward (2,3,32,32)→(2,10); exactly 2 projection shortcuts with shapes (128,64,1,1) and (256,128,1,1) and 7 identity blocks; decay-group split picks up the 1×1 weights (decay-group params 4,321,472) and excludes the projection BNs, as the plan required — no optimizer changes needed.

### Surprises & Discoveries

- None — the existing `_weights_init` Kaiming pass and the `ndim`-based decay split both absorb the new modules with zero extra code, exactly as the plan predicted.

### Decisions

- None beyond the plan. Projection BN left at default init (γ=1) per the plan's explicit EXP-018 guard.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: bnfojwnrw (composite background script: pre-check + train + inline watchdog)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-10 11:16:29 (GPU 0 confirmed free at launch by the script's pre-check)
- **Ended**: 2026-06-10 11:24:59 (TRAIN_EXIT rc=0)

Description:
- Single 300s-budget run of the baseline recipe with learned 1×1+BN projection shortcuts at the two stage transitions (layer2[0]: 64→128 s2; layer3[0]: 128→256 s2), identity shortcuts elsewhere — the last structural 2016-era component modernized to match the WRN reference at our width regime. Launched on GPU 0 via the standard composite launcher with inline contention watchdog. Expected: signatures on-family (params 4,327,754, dt ≈ 22.4–23ms, 137–139 epochs, VRAM ≈ 1615–1650MB, startup ~13s); early trail at-or-above baseline family (ep1 ≈ 38–39, ep10 ≥ 75); best_test_acc ≥ 96.81 per hypothesis.

Observations:
- Launch 11:16:29, GPU 0 confirmed free by pre-check (source: task bnfojwnrw output L1). run.log live within seconds: `ResNet-20 (4x wide) | params: 4,327,754` — attribution pin matches the CPU pre-validation exactly; step 50 at dt 23ms / 21,970 img/s — on-family throughput (baseline 22.4ms), the two projection kernels cost ≲0.5ms as planned (source: run.log header + first step line).
- Pristine execution: zero watchdog SLOW events; post-hoc profile 0 of 259 windows >30ms, mean 23.0ms — the projection costs a real but small +0.6ms/step, yielding 135 epochs / 13,044 steps vs baseline's 139 (within the plan's 133–141 window). startup_seconds 22.6 — a fresh inductor compile for the new graph topology (matches EXP-006's 22.8s cold-compile, vs ~13s warm-cache reruns); total 495.2s, well under the 600s cap. VRAM 1661.2MB (+48MB), params 4,327,754 — attribution exact (source: task bnfojwnrw output; run.log; profile command).
- **Early trail is SLOWER than the baseline family, not faster**: ep1 34.55 vs 38.2–38.95, ep5 59.84 vs 63.76, ep10 best 74.20 vs ~75 — the two projection blocks spend early heat learning their shortcuts (a mild echo of the EXP-018 mechanism, though far gentler since 7 of 9 shortcuts stay identity). Mid-run it closes onto the family (87.67@60, 93.00@100) (source: run.log eval trail).
- Proper converged plateau: final eight evals 96.44–96.58, final 96.50 ≈ best 96.58 (Δ0.08) — converged flat at −0.13pp. final_test_loss 0.1882 on par with the best on record; the max-statistic pays in accuracy (source: run.log eval trail).

Key Metrics:
- best_test_acc: 96.58% @ ep 134/135 (source: run.log summary + eval trail) — baseline 96.71, bar 96.81: −0.13pp
- total_seconds: 495.2 | training_seconds: 300.0 | startup_seconds: 22.6 (fresh compile for the new topology) (source: run.log summary)
- num_epochs: 135 | num_steps: 13,044 | windowed dt mean 23.0ms, 0/259 >30ms (source: run.log + profile)
- peak_vram_mb: 1661.2 (+48 vs baseline) | num_params: 4,327,754 (= pre-validation pin) (source: run.log summary)
- final_test_acc: 96.50% | final_test_loss: 0.1882 (source: run.log summary)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition (contention sanity)**: 0 of 259 windows >30ms, mean 23.0ms; num_epochs 135 within the plan's 133–141 window (the +0.6ms/step projection cost accounts for 139→135) — CLEAN, analyzable (source: profile command output).
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: `grep "^best_test_acc:" run.log` → 96.58%. **FAILED** (−0.13pp vs baseline).
- **Condition 2 — total ≤ 600s, rc=0**: skipped — aborted after prior failure (informally: 495.2s, rc=0 — would have passed).
- **Condition 3 — eval at most once per epoch**: skipped — aborted after prior failure (informally: 135 eval lines = 135 epochs — would have passed).

### Informational Metrics

Not collected per protocol (necessary condition failed). Informal values from run.log for the report: peak_vram_mb 1661.2; num_epochs 135; num_params 4,327,754 — params match the pre-validation pin exactly, attribution is clean.

## Errors & Dead Ends

<!-- Append only. Never delete. Agent reads this before proposing any next action.
     Include source pointers for traceability. -->

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
