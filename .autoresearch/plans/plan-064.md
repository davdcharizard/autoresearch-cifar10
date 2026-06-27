# Plan EXP-064: ACNet asymmetric convolution blocks (structural reparameterization), probe-gated
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-064.md

Baseline 96.71 @ 1990397 (bar ≥ 96.81; family mean 96.57, σ 0.16; bar-over-mean = +0.24). Hypothesis per brainstorm-064: replacing all 19 3x3 convs with ACBs (3x3 ∥ 1x3 ∥ 3x1, per-branch BN, summed pre-ReLU) raises the plateau level by ≥ the dilution + 0.24. The launch decision is made UNCHARGED by a GPU probe with a pre-registered inequality (EXP-063 pattern); the probe times the BASELINE net as an internal control so the toll is a load-robust ratio (EXP-062 inversion lesson).

## Milestones

### Milestone 1: Code changes + CPU sanity
- [x] train.py: `ACB(nn.Module)` — `conv3x3+BN ∥ conv1x3+BN ∥ conv3x1+BN`, summed (paddings (1,1)/(0,1)/(1,0); shared stride; all convs bias=False; BNs default init). Replace every `conv+bn` pair in BasicBlock (conv1/bn1, conv2/bn2) and the stem (conv1/bn1) with one ACB each — 19 sites; forward becomes `relu(acb(x))` etc. Everything else byte-identical to baseline (constants, lr_at, transforms, optimizer policy ndim>1 decay — ACB conv weights are 4D → decayed; BNs 1D → not).
- [x] CPU sanity — 6/6 pass; NUM_PARAMS_ACB = 7,149,002; fold-equivalence max|d| = 2.15e-06 (`CUDA_VISIBLE_DEVICES="" PYTHONPATH=. uv run python /tmp/exp064_sanity.py`): (a) model constructs, forward shape (8,10), record EXACT param count (pin as `NUM_PARAMS_ACB` for integrity); (b) strided ACB (stage transitions) output shape matches baseline path; (c) fold-equivalence: for one ACB in eval mode, folded single conv (BN-fused, center-aligned kernel add) matches branched output to ≤1e-5 — proves the eval'd function IS the plain-net function; (d) 5 CPU steps at lr 0.01, loss decreasing; (e) lr_at unchanged at p ∈ {0.05, 0.15, 0.5, 1.0}.

### Milestone 2: GPU probe with internal control (uncharged, ~4 min) — THE LAUNCH GATE
- [x] Gate: passed at apps=0, load1=14.9.
- [x] /tmp/exp064_gpu_probe.py: ran clean — **B = 22.36ms** (control mid-family-band), **P = 43.15ms**, **toll ratio 1.930, P_norm = 43.23**; warmups 11.5s/36.4s; VRAM 3,195MB.
- [x] **Pre-registered launch criterion**: FAILED decisively (43.23 vs ≤ 26.0; at 43.23 → 71 epochs → dilution −0.95 → required gain 1.19 > published max 1.11). **NO LAUNCH → verdict `invalid` (NaN), reparameterization cost-closure.** Family-level diagnostic (/tmp/exp064_dbblite_diag.py, uncharged): minimum variant DBB-lite (3x3 ∥ 1x1, fewest possible extra launches) = 28.61ms → required gain ≈ 0.69 > single-branch ablation gains → the WHOLE multi-branch reparam family fails its inequality. Arithmetic: epochs(P_norm) = 300000/P_norm/97.65; dilution ≈ 0.014/ep × (139 − epochs); at P_norm = 26.0 → 118 ep → −0.29, required realized gain = 0.24 + 0.29 = 0.53 = published mid-band (+0.35–1.11). Above 26.0 the required gain exceeds the band's defensible middle → NO LAUNCH → verdict `invalid` (NaN), reparameterization cost-closure (launch-pricing law EXP-034/040).
- [ ] If LAUNCH: record probe-revised bands — steps ∈ [300000/(P_norm+1.5), 300000/(P_norm+0.1)]; epochs = steps/97.65; evals = epochs ± 2 (every-epoch cadence); D0 ∈ [P_norm−1.3, P_norm+1.3]; wall projection = 300 + startup(probe-measured) + epochs×~0.95 + stalls ≤ 600 — at ~118 ep ≈ 300+35+112 ≈ 447s, ample margin (if projection > 580, thin evals to every 2nd epoch BEFORE launch).

### Milestone 3: Gated composite run
- [ ] /tmp/exp064_composite.sh (exp061-standard single-phase, retargeted): dual gates (apps==0 AND load<60, poll 30s×240) → rm -f run.log → train background → watchdog 44×15s; GATE_KILL D0 > max(29, P_norm+2.5); CONTENTION_KILL 4 consecutive > max(29, D0×1.25); STARTUP_KILL no step lines by tick 12; NaN guard; divergence guard test acc < 20 at/after ep 10; WALL_CAP backstop.
- [ ] Launch via Bash run_in_background (stdout → /tmp/exp064_composite_run1.log) + until-grep watcher + TaskOutput(block=true).
- [ ] Clean completion: RC=0, no kill markers, D0/windows/steps in bands.

### Milestone 4: Verification (first-failure-stop; integrity gates Condition 1)
- [ ] Integrity: RC=0; step LEDGER in probe band (BINDING contamination gate); epochs = steps/97.65 ±2; eval-line count = epochs (+ final partial); `num_params` printed == `NUM_PARAMS_ACB` from sanity; `training_seconds: 300.0`; first eval ≥ 30 (relaxed; ledger binding); zero NaN; VRAM < 4,200MB.
- [ ] Condition 1: best_test_acc ≥ 96.81. **Near-bar protocol (EXP-052)**: any Run-1 read ≥ 96.81 triggers one byte-identical replicate; the PAIR MEAN decides (max never decides).
- [ ] Condition 2: total_seconds ≤ 600.
- [ ] Condition 3: validation ≤ once/epoch (structural: unchanged every-epoch cadence).

## Code Changes

- **train.py** (only editable file):
  1. **ACB module** (~20 lines): three parallel convs with per-branch BN, summed. Signature `ACB(in_ch, out_ch, stride=1)`. Paddings: square (1,1); horizontal 1x3 (0,1); vertical 3x1 (1,0). All bias=False (BN absorbs). With stride s, all branches share s — output dims match (floor arithmetic identical for H=32/16/8).
  2. **BasicBlock**: `self.acb1 = ACB(in, out, stride)`, `self.acb2 = ACB(out, out)`; forward `out = F.relu(self.acb1(x)); out = self.acb2(out); out += shortcut; out = F.relu(out)` — same topology as baseline with conv+bn → ACB swapped. Shortcut path untouched (pad shortcut as baseline).
  3. **Stem**: `self.acb1 = ACB(3, 64)`; forward `F.relu(self.acb1(x))`.
  4. **Eval path**: UNCHANGED — `evaluator.evaluate(base_model, device)` on the branched module; in eval mode the ACB computes the exact same function as its folded plain conv (BN-eval affine; fold exact — sanity check (c) proves it). No folding machinery, no wrapper, Eval untouched.
  5. Everything else byte-identical: constants, lr_at, transforms, loader, optimizer (2-group selective WD — ACB convs 4D → decay, BNs → none), compile + 3-iter warmup, timed loop, every-epoch eval, summary prints.
- **Why this tests the hypothesis**: the ONLY change is the train-time parameterization of each conv; eval function class is identical to a plain ResNet-20 4x. Any metric movement is attributable to the reparameterized optimization geometry (+ its dt toll, priced by the probe).
- **Risks/edge cases**: inductor fusion quality of parallel 1D convs unknown (probe measures truth); BN-stat triplication is VRAM-trivial; stride-2 ACB shape equality asserted in sanity (b).

## Configuration Changes
- None. All hyperparameters byte-identical to baseline (the intervention is purely structural at train time). num_params changes (recorded by sanity, pinned for integrity); per-step dt changes (priced by probe).

## Execution Environment
- Method: local, GPU 0 only; probe first (M2 gate), then single gated run via /tmp/exp064_composite.sh; `uv run train.py > run.log 2>&1`
- Resources: GPU 0; VRAM expect ~2.0–2.6GB (≤1.6× baseline activations); host load gates (probe <40 with internal control normalization, launch <60)
- Estimated runtime: probe ~4 min; run ≈ 440–470s total (300 charged + ~35s startup + ~118 evals × ~0.95s); charged exactly 300.0s
- Log output: run.log (truth); /tmp/exp064_composite_run1.log (gate/watchdog); delete run.log after the experiment

## Abort Criteria
- M2 gate: NO LAUNCH if P_norm > 26.0 (or control B out of [21.5, 25.0] after one re-probe) — primary falsification point, zero charged cost
- GATE_KILL: D0 > max(29, P_norm+2.5) at GATE_DECISION
- CONTENTION_KILL: 4 consecutive windows > max(29, D0×1.25)
- STARTUP_KILL: no step lines by tick 12
- NaN in loss prints → kill (crash verdict); test acc < 20% at/after epoch 10 → kill
- WALL_CAP backstop at watchdog window end

## Verification Protocol

### Verification Procedure
First-failure-stop; integrity gates Condition 1.

0. **Integrity** (2 min, run.log + composite log): RC=0, no kill markers; `num_steps` ∈ [300000/(P_norm+1.5), 300000/(P_norm+0.1)] (BINDING); epochs consistent; eval lines = epochs (+ final); `num_params` == NUM_PARAMS_ACB (sanity-pinned); `training_seconds: 300.0`; zero NaN; VRAM < 4,200MB.
1. **Condition 1** — `grep "^best_test_acc:" run.log` ≥ 96.81 (baseline 96.71 + 0.1 via `exp-index.sh baseline`). Near-bar protocol (EXP-052): Run-1 ≥ 96.81 → one byte-identical replicate; PAIR MEAN decides.
2. **Condition 2** — `grep "^total_seconds:" run.log` ≤ 600.
3. **Condition 3** — eval cadence structural check (eval lines ≤ epochs).

Pre-registered branches: (i) pair-mean ≥ 96.81 → improvement, commit; (ii) probe NO-LAUNCH → `invalid`/NaN, reparameterization cost-closure (the axis is closed on launch-pricing grounds, mirroring EXP-063); (iii) run ∈ [96.41, 96.73] → no-improvement; reparameterization gain absorbed by the heavy-aug recipe (absorption law extends to optimization-geometry interventions — external transfer 0-for-19); (iv) < 96.41 → no-improvement; branched optimization actively harmful here (diagnose via train-loss trajectory vs family); (v) infra → byte-identical relaunch ≤ 2.

Mechanism checks (informational): train smooth-loss at matched epoch counts vs family (does the branched net optimize FASTER, as the paper's mechanism predicts?); plateau eval count within 0.15 of best; P/B toll ratio from probe.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect 2,000–2,600MB
- num_epochs: expect ~118 at P_norm = 26 (~128 at 24)
- num_steps: probe band; num_params: NUM_PARAMS_ACB (sanity-pinned)
