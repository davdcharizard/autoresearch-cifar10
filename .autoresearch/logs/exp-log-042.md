# Experiment Log: EXP-042 — Grouped two-member deep ensemble (2 × 3x ResNet-20, sum-CE, logit-mean inference)

## Execution

- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-042.md
- **Plan**: plans/plan-042.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-042
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed (run GATE_KILLed per pre-registered protocol — screen verdict `invalid`, grouped-conv pricing fact recorded)

## Implementation Notes

### Summary
Milestone 1 as planned (66 insertions / 25 deletions, train.py only): constants `MEMBER_WIDTH_MULT = 3`, `NUM_MEMBERS = 2` replace `WIDTH_MULT = 4`; `BasicBlock` gains a `groups` arg on both convs and a per-member pad shortcut (reshape to (B, groups, C/groups, H, W) → F.pad the per-member channel dim → reshape back; `reshape` not `view` for channels_last strides); `ResNet` is now a grouped ensemble — total widths 96/192/384 with `groups=2` threaded through the stem (`in_channels=6`, fed by `x.repeat(1,2,1,1)`) and all blocks, two per-member fc heads (192→10 each), `_forward_members()` returning (l1, l2), and `forward()` branching on `self.training`: tuple in train mode, `(l1+l2)/2` in eval mode (satisfies `Eval.evaluate()`'s single-tensor contract architecturally). Compile warmup and timed loop compute `loss = CE(l1) + CE(l2)` (SUM — each member's per-step gradient/LR/noise byte-identical to the certified baseline recipe). All five CPU sanities passed: (A) member isolation — perturbing every member-2 slice (second-half conv out-channels, second-half BN affines, fc2) left member-1 logits BIT-IDENTICAL while l2 changed; (B) gradient isolation — backward on CE(l1) alone gave fc2 zero/None grad and zero second-half rows on a stage-2 conv, sum loss touched both halves; (C) eval contract — eval-mode forward returns a single (4,10) tensor equal to mean member logits; (D) constructed num_params = 4,825,460 (hand estimate exact this time); (E) 2-step CPU train smoke, losses 7.89 → 7.11 (sum-of-two-CE scale, ~2× family — cosmetic only). Extra: channels_last forward (input and model) matches contiguous to 1e-5.

### Surprises & Discoveries
- First sanity script run flagged a channels_last "mismatch" that was a test bug (reference output captured BEFORE the 2-step train smoke mutated weights); a fresh paired comparison passed exactly. No model-side surprises — grouped-conv semantics, per-member Kaiming fans (weight shape (out, in/groups, k, k)), and the per-member pad all behaved as the plan predicted.

### Decisions
- Kept the loss as an explicit two-term sum (not a loop over heads) to keep the compiled graph shape obvious and the diff minimal.
- `_forward_members` exposed as a method (not inlined) so the eval-contract sanity can compare `model(x)` in eval mode against the explicit member mean — also useful for any future per-member diagnostics.

## Run Log

### Run 1
- **Description**: Full budget-matched run of the grouped 2×3x ensemble on GPU 0 via `/tmp/exp042_composite.sh` (EXP-040 D0-median dt-gate variant: median of first 3 watchdog windows, GATE_KILL if D0 > 28.0ms pre-registered as a screen `invalid`, contention threshold D0×1.25 thereafter, STARTUP_KILL tick 12 for the cold grouped-graph compile, NaN/divergence/wall guards). Projected dt ≈ 24.1ms (FLOPs 1.125× via ∂dt/∂FLOPs ≈ 13.3ms/unit) → ~125–129 epochs IF grouped kernels price like dense ones — the gate resolves that unknown in ~90s. Hypothesis: function-space (multi-mode) averaging of two independently-initialized members raises the converged plateau MEAN above the single-model family (~96.5); falsified by gate-kill (grouped pricing) or a plateau in the baseline band (init-only diversity insufficient).
- **Job ID / PID**: background task blgkv9hx4 (`/tmp/exp042_composite.sh`)
- **Log file**: run.log (project root); watchdog via composite stdout (task output file)
- **WandB**: N/A
- **Status**: killed by dt gate (composite exit 47, GATE_KILL)
- **Started**: 2026-06-10 21:10:14 (gates clear at poll 1: apps=0, load=6)
- **Ended**: 2026-06-10 ~21:11:45 (GATE_KILL at tick 6, ~90s into the timed loop)
- **Observations**: Three gate windows (200 steps each, steps 300/500/700) read EXACTLY 63.0ms; printed dt 63ms on every step line; img/s ~8,150. Host clean (load 6, zero GPU-0 apps) — the reading is the grouped-kernel cost, not contention. Startup/compile completed normally; training itself was healthy: sum-CE loss 4.58 (step 50) -> 3.68 (step 250), ep1 eval 39.22% vs family ~38 — ensemble-mean inference showed NO deferral toll. Source: composite stdout (task blgkv9hx4); run.log header + step lines.
- **Key Metrics**: D0 = 63.0ms (vs dense-conv baseline 22.4ms and the dense >256-channel cliff at 54ms) at FLOPs 1.125x -> projected 49 epochs (gate threshold required >=111). num_params 4,825,460 (matches constructed value). 7 evals before kill; no NaN, no divergence. GATE_DECISION: D0=63.0ms projected_epochs=49 contention_thresh=78.8ms.

## Experimental Adjustments

(none yet)

## Errors & Dead Ends

### 2026-06-10 — GATE_KILL: grouped convolutions price at 2.8x dense dt on H20 + torch.compile(default) + channels_last + bf16
- Error: `GATE_KILL: D0=63.0ms > 28ms (projected 49 epochs < ~111)` (composite exit 47)
- Root cause: groups=2 convs at total widths 96/192/384 (192/group max, all per-group channels <= 256) run at 63.0ms vs 22.4ms dense baseline for only 1.125x FLOPs — grouped-conv kernels on this stack do not inherit dense pricing (likely cuDNN/inductor fallback path under channels_last+bf16); the dense >256 cliff (54ms, EXP-040) is not the binding number, grouped is WORSE.
- Source: task blgkv9hx4 stdout; run.log step lines (dt 63ms uniform)
- Do NOT retry: ANY grouped/depthwise-separable conv design on this stack without a fresh dt gate; assume grouped convs cost ~2.5-3x dense dt regardless of per-group width. The in-one-kernel multi-member trick is closed on this hardware — multi-member designs must use full dense kernels (e.g., alternating-step members, brainstorm-042 Idea B).

## Verification Results

### Conditions Checked

First-failure-stop per plan-042 § Verification Protocol; baseline at verification time 96.71 (bar 96.81).

- **Pre-condition — gate**: **FAILED (pre-registered branch)**: GATE_DECISION D0 = 63.0 > 28.0ms. Per plan § Abort Criteria, the pre-registered screen verdict applies: `invalid` (NaN metric) — the experiment is a ~90s throughput screen; the ensemble mechanism was never tested at a fair epoch count, so no metric judgment is made. Conditions 1–3 not evaluated (no completed run exists).
- Integrity of the screen itself: host clean at launch and through the kill (load 6, zero foreign GPU-0 apps); three independent 200-step windows identical at 63.0ms; params matched the constructed value — the pricing fact is trustworthy.

### Informational Metrics

- num_params: 4,825,460 (constructed and printed values match)
- peak_vram_mb: n/a (run killed before summary)
- num_epochs: n/a (projected 49 at D0=63.0ms)

## Human Notes

(autopilot — none)
