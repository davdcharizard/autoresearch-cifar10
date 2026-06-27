# Plan EXP-042: Grouped two-member deep ensemble (2 × 3x ResNet-20 via groups=2, sum-CE training, logit-mean inference)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-042.md

## Milestones

### Milestone 1: Code change implemented and sanity-checked (CPU) — member-isolation test is MANDATORY before launch
- [x] On branch `autoresearch/exp-042` (cut from `autoresearch/dev`), edit `train.py`:
  (a) constants: replace `WIDTH_MULT = 4` with `MEMBER_WIDTH_MULT = 3` and `NUM_MEMBERS = 2` (per-member widths 48/96/192; totals 96/192/384);
  (b) `BasicBlock`: add `groups` arg → both convs get `groups=groups`; pad-shortcut becomes per-member: `B,C,H,W = shortcut.shape; shortcut = shortcut.reshape(B, groups, C//groups, H, W); shortcut = F.pad(shortcut, (0,0, 0,0, 0, self.pad_channels//groups)); shortcut = shortcut.reshape(B, -1, H, W)` (use `reshape`, not `view` — channels_last strides);
  (c) `ResNet` → grouped ensemble: widths `w1,w2,w3 = 16*mult*M, 32*mult*M, 64*mult*M` with `groups=M` threaded through stem conv (in_channels `3*M`) and all blocks; per-member fc heads `self.fc1, self.fc2 = nn.Linear(w3//M, 10), nn.Linear(w3//M, 10)`; `_forward_members(x)`: `x = x.repeat(1, M, 1, 1)` → trunk → avgpool/flatten → `l1 = self.fc1(out[:, :w3//M])`, `l2 = self.fc2(out[:, w3//M:])`; `forward(x)`: `l1, l2 = self._forward_members(x)`; `return (l1, l2) if self.training else (l1 + l2) / 2`;
  (d) compile warmup: `l1, l2 = model(warm_x)`; `warm_loss = F.cross_entropy(l1, warm_y, label_smoothing=LS) + F.cross_entropy(l2, warm_y, label_smoothing=LS)`;
  (e) training loop: `l1, l2 = model(inputs)`; `loss = CE(l1, targets, ls) + CE(l2, targets, ls)` — **SUM, not mean** (each member's per-step gradient/LR/noise byte-identical to baseline recipe); `outputs` no longer needed in train path;
  (f) keep everything else byte-identical: schedule, optimizer (selective WD picks up grouped conv weights via ndim>1), batch 512, transforms, eval call `evaluator.evaluate(base_model, device)` untouched
- [x] CPU sanity A — MEMBER ISOLATION: perturb all member-2 parameter slices (second half of every conv weight's out-channel dim, second half of every BN weight/bias, fc2 entirely) → member-1 logits `l1` BIT-IDENTICAL pre/post on the same input (train-mode forward, fixed input); `l2` changed
- [x] CPU sanity B — GRADIENT ISOLATION: backward on `CE(l1)` alone → fc2 grad None/zero AND second-half out-channel rows of a stage-2 conv weight grad all zero; backward on the sum loss → both halves nonzero
- [x] CPU sanity C — EVAL CONTRACT: `model.eval()`; `model(x)` returns single tensor shape (B, 10); `torch.allclose(model(x), (l1+l2)/2)` where `l1,l2 = model._forward_members(x)` in eval mode
- [x] CPU sanity D — params/shape: record constructed `num_params` (hand estimate ≈ 4,825,460 — TRUST THE CONSTRUCTED VALUE per EXP-040's estimate-error lesson) and forward shape on (4,3,32,32); `git diff --stat` shows train.py only
- [x] CPU sanity E — 2-step train smoke on CPU (tiny batch, no autocast): forward/backward/optimizer.step run without error, loss finite and decreasing-or-flat

### Milestone 2: Gated launch, gate decision recorded
- [x] Copy `/tmp/exp040_composite.sh` → `/tmp/exp042_composite.sh` (D0-median dt-gate variant) with: GATE_KILL if D0 > 28.0ms (off-rung; projected dt 24.1ms + margin); contention THRESH = D0 × 1.25 thereafter (4 consecutive windows); STARTUP_KILL tick 12 (~180s — cold inductor compile of new grouped graph); NaN guard; DIVERGENCE_KILL eval < 15% after epoch 5; WALL_CAP tick 44. Dual launch gates unchanged (GPU-0 zero compute apps AND 1-min load < 60, poll 30s)
- [x] GATE_DECISION line observed (D0=63.0ms → GATE_KILL, pre-registered screen verdict) in composite stdout; if GATE_KILL → record D0, skip to verification with the pre-registered screen verdict (see Abort Criteria)
- [x] Run terminated by gate (composite exit 47); no summary by design

### Milestone 3: Verification and exp-log complete
- [x] First-failure-stop verification executed (pre-condition gate failed → screen verdict, conditions 1–3 not evaluated) (protocol below), recorded in `logs/exp-log-042.md § Verification Results`
- [x] Diagnostics recorded (63.0ms ×3 windows, ep1 eval 39.22 family-equal, loss falling — kill purely on throughput): ep5/10/20 evals vs family (~64/~75/~79), last-15 plateau mean/spread vs ~96.5/±0.15, final_test_loss vs ~0.185 (ensemble averaging may read LOWER loss — diagnostic for mechanism even on a miss)
- [ ] run.log deleted after extraction (analyze housekeeping)

## Code Changes
- **train.py** (only editable file, ~60-line diff): restructure the single 4x ResNet-20 into a 2-member grouped ensemble. Mechanism under test: function-space (multi-mode) averaging — two independently-initialized 3x members co-trained on identical batches, inference = mean logits. Member independence is STRUCTURAL (groups=2 convs never mix across the channel halves; BN is per-channel; separate fc heads; sum-of-CE gives each member exactly the baseline per-step dynamics). Edge cases: (i) stem — input has 3 channels, so `x.repeat(1,2,1,1)` feeds each group its own RGB copy (grouped conv requires in_channels divisible by groups); (ii) pad shortcut — baseline `F.pad` appends zeros at the channel END, which under grouped layout would give them all to member 2; the reshape-pad-reshape fix pads each member's half (sanity A catches any error); (iii) Kaiming init sees grouped weight shapes (out, in/groups, k, k) → correct per-member fan automatically; (iv) train-mode forward returns a tuple (compile traces training=True graph), eval-mode returns mean logits — `Eval.evaluate()` (prepare.py L32–47) calls `model.eval()` then `model(inputs)` expecting a single (B,10) logits tensor: contract satisfied by the architecture itself, no wrapper, eval untouched; (v) loss PRINT scale ≈ 2× family (sum of two CEs) — cosmetic only; do not compare printed loss to family curves; NaN guard string-match unaffected.

## Configuration Changes
- WIDTH_MULT 4 → MEMBER_WIDTH_MULT 3 × NUM_MEMBERS 2: capacity→multiplicity reallocation at FLOPs 1.125× (2 × (3/4)²). Params ≈ 4.83M (vs 4.286M). All training constants (PEAK_LR 0.4, WARMUP 0.15, momentum 0.9 nesterov, WD 5e-4 selective, LS 0.1, batch 512, TA+RE transforms, time-keyed cosine) byte-identical — per the certified-local-optimum and gradient-noise laws, nothing else may move.

## Execution Environment
- Method: local, via `/tmp/exp042_composite.sh` with `run_in_background: true`
- Resources: GPU 0 ONLY (never GPU 1; wait if busy), ~1.8–2.0GB VRAM est., host load < 60 at launch
- Estimated runtime: ~520s total (300s charged + startup ~20–40s cold compile + ~125–129 × ~1.5s evals); cap 600s
- Log output: `uv run train.py > run.log 2>&1` (no tee); run.log deleted after extraction
- Tool skill: none (local)

## Abort Criteria
- **GATE_KILL (pre-registered)**: D0 (median of first 3 watchdog windows) > 28.0ms ⇒ grouped kernels do not price like dense ones (cliff or grouped-kernel inefficiency) ⇒ projected epochs < ~111 make the member-starvation arithmetic implausible. Verdict pre-registered as `invalid` (NaN metric, screen) with key learning = the grouped-conv pricing fact; Idea B (alternating-step ensemble, brainstorm-042) becomes the natural next-loop candidate. Do NOT relaunch a slower variant this loop.
- STARTUP_KILL: no step line by tick 12 (~180s)
- CONTENTION_KILL: 4 consecutive 15s windows > D0 × 1.25 after the gate passes → confirm contamination (nvidia-smi apps, /proc/loadavg), relaunch byte-identically when both launch gates clear
- NaN guard: any `loss: nan` → kill
- DIVERGENCE_KILL: any eval < 15% after epoch 5
- WALL_CAP_KILL: still running at tick 44 (~660s)
- Experiment-specific monitor (not a kill): if early evals sit far below family (ep5 < ~55), suspect a member-mixing bug despite sanity A — let the run finish (divergence guard protects), flag for analysis

## Verification Protocol

### Verification Procedure
First-failure-stop; baseline via `exp-index.sh baseline` at verification time (currently 96.71; bar = 96.81).

**Pre-condition (run integrity):**
- Gate: GATE_DECISION shows D0 ≤ 28.0 (else the pre-registered screen verdict applies, no metric judgment).
- Profile: 200-step quantization-safe windows (every 4th step-line pair) — require mean within [D0 − 1.5, D0 + 1.5] ms and 0 windows > D0 × 1.25; num_epochs ≥ 111 and within ±5 of 300/(mean_dt × 0.0977) (97.66 → 97 steps/epoch with drop_last). If contaminated → rerun byte-identically, do not judge.
- Integrity: `num_params` equals the Milestone-1 constructed value; `training_seconds: 300.0`; eval-line count == num_epochs.
- Timeout: greps on finished run.log; missing summary ⇒ crash (`tail -n 50 run.log`).

**Condition 1 — best_test_acc ≥ 96.81**: `grep "^best_test_acc:" run.log`. Fail → STOP, verdict `no-improvement` (rest incidental).

**Condition 2 — within budget**: composite rc == 0 AND `total_seconds` ≤ 600.0.

**Condition 3 — validation ≤ once/epoch**: `grep -c "eval ep" run.log` ≤ num_epochs.

**Diagnostics (always):** ep5/10/20 vs family (~64/~75/~79) — members are 3x (less capacity) but ensembled from epoch 1, so early evals should track family within a few pp; last-15 plateau mean/spread vs ~96.5/±0.15 — the hypothesis predicts a plateau MEAN shift upward with REDUCED-or-equal scatter (averaging lowers prediction variance; a scatter INCREASE would contradict the mechanism); final_test_loss vs ~0.185 — ensemble averaging should read lower loss even if accuracy misses (calibration gain), separating "members too correlated" (loss ≈ family, acc ≈ family) from "diversity real but boundary gain insufficient" (loss < family, acc ≈ family).

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect ~1.8–2.0GB
- num_epochs: `grep "^num_epochs:" run.log` — expect ~125–129 at dt 24.1ms
- num_params: `grep "^num_params:" run.log` — expect the Milestone-1 constructed value (≈4.83M)
