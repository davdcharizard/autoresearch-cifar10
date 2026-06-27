# EXP-015: Halve weight decay (WEIGHT_DECAY 5e-4 → 2.5e-4)

## Execution

Overall Status & Info:
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-015.md
- **Plan**: plans/plan-015.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-015
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: failed (necessary condition 1 failed: best_test_acc 96.41 < 96.81 bar; clean valid run — research no-improvement, not infra)

## Implementation Notes

### Summary

Milestone 1 executed exactly as planned: branched `autoresearch/exp-015` from `autoresearch/dev` (clean @ 1990397), changed line 26 of train.py `WEIGHT_DECAY = 5e-4` → `2.5e-4` (git diff --stat: 1 file, +1/−1), py_compile exits 0. The optimizer's param-group split (decay on ndim>1 only) makes this a true single-knob change — BN/bias groups were already at 0.0. Launch used the composite-script protocol from EXP-014's lesson: GPU-0 pre-check, training launch, and the inline contention watchdog all in one background command chain (task b4a4n6lud), so no turn-scheduling gap can blind the detector.

### Surprises & Discoveries

- None at implementation time — GPU 0 was free at the pre-check (unlike EXP-014, no wait was needed).

### Decisions

- Kept the plan's one-octave halving (2.5e-4) rather than a smaller step: the dose-response framing wants a clearly-resolvable increment, and the diagnostic trail (over-fit tail vs variance shave vs gain) disambiguates follow-ups regardless of sign.

## Experimental Adjustments

## Run Log

### Run 1

Metadata:
- **Job ID**: task b4a4n6lud (local background; composite launcher + inline watchdog, kills run on 4 consecutive >30ms windows)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-fable-5/run.log
- **WandB**: N/A
- **Status**: completed (clean)
- **Started**: 2026-06-10 09:11
- **Ended**: 2026-06-10 09:21

Description:
- Single training run of the baseline recipe with conv/linear weight decay halved 5e-4 → 2.5e-4 — the last never-probed recipe constant. Tests pressure-reduction (saturated augmentation dose-response) and, via the WD-with-BN effective-LR mechanism, a mild cold-side probe of the heat curve. Expected: byte-identical throughput signatures (~139 epochs, dt ~22.3ms, 1613MB, 4,286,026 params); hypothesis predicts best_test_acc ≥ 96.81 with train loss running below baseline mid-schedule; an over-fitting tail (peak-then-decay test acc) would instead close the pressure axis.

Observations:
- LAUNCH 09:11:44; early run.log shows CIFAR-10 DOWNLOAD progress lines — the post-EXP-014 `git clean -fd` removed the untracked `data/` dir, so this run pays a one-time re-download in startup. The download is outside the timed budget (epochs unaffected) but inflates total_seconds; if total > 600 solely from download-inflated startup, that is an INFRASTRUCTURE artifact (baseline never paid it) → rerun with cached data rather than failing the condition. Future cleanups must use `git clean -fd -e .autoresearch/ -e data/`.
- Clean execution: watchdog zero SLOW events; post-hoc windowed profile 0 of 266 windows > 30ms (mean 22.4ms); 139 epochs / 13393 steps on projection; total 509.8s (≈27s above EXP-014's 482.9 — absorbed download/cache effects, well under cap); startup 12.9s; VRAM 1613.0, params 4,286,026 (source: task b4a4n6lud output; run.log windowed profile)
- HYPOTHESIS REFUTED — and the diagnostic disambiguated cleanly: trajectory ran slightly BEHIND baseline all schedule (ep 20: 81.5; ep 60: 89.9; ep 100: 94.2) and the tail CONVERGED FLAT (96.38 @ ep 130 → best 96.41 first at ep 133, final = best 96.41) with final_test_loss 0.1901 ≈ baseline — NO over-fitting tail (no peak-then-decay). Less pressure did not free capacity; it just trained marginally worse (source: run.log eval lines ep 1/20/60/100/120/130/139)
- Reading per brainstorm-015's pre-registered diagnostics: this is neither "over-fit tail" nor a pure "variance shave" — it is "pressure was AT optimum": the regularization dose-response now has a measured point BELOW the current dose (WD-half −0.30) to go with the points above it (reflect −0.14, mixup −0.46); the recipe sits at a measured local optimum in BOTH pressure directions. The effective-LR reading concurs: the mild cooling also produced no gain, consistent with the heat curve being at optimum too (source: run.log trajectory; goal-learnings § Patterns High)

Key Metrics:
- best_test_acc: 96.41% @ ep 133 (source: run.log summary; bar was 96.81)
- total_seconds: 509.8; training_seconds: 300.0; num_epochs: 139; num_steps: 13393; peak_vram_mb: 1613.0; num_params: 4,286,026; final_test_loss: 0.1901 (source: run.log summary block)

## Verification Results

<!-- Filled after the experiment completes successfully.
     If ANY necessary condition fails, remaining conditions are not evaluated. -->

### Conditions Checked

- **Pre-condition — contention sanity (Protocol Findings EXP-011/014)**: num_epochs 139 = projection; watchdog zero SLOW events; post-hoc profile 0/266 windows > 30ms (mean 22.4ms). **CLEAN — conditions evaluable.** (source: run.log windowed profile; task b4a4n6lud output)
- **Condition 1 — best_test_acc ≥ 96.81 (baseline 96.71 + 0.1)**: parsed 96.41 from `grep "^best_test_acc:" run.log`. **FAILED** (96.41 < 96.81; −0.30pp vs baseline). (source: run.log summary block)
- **Condition 2 — total ≤ 600s**: skipped — aborted after prior failure (observed informally: 509.8s would have passed; the feared download cap-bust did not materialize)
- **Condition 3 — validation ≤ once/epoch**: skipped — aborted after prior failure (observed informally: 139 eval lines = 139 epochs would have passed)

### Informational Metrics

- Not collected per protocol (necessary condition failed). Informal: peak_vram_mb 1613.0 (= baseline), num_epochs 139 (= baseline), num_params 4,286,026 (= baseline), final_test_loss 0.1901 (≈ baseline — no over/under-fit signature).

## Errors & Dead Ends

## Human Notes

> {Researcher can add comments, corrections, or context here}

<!-- NOTE: Human notes are high trust and privileged relative to other info in this document. -->
