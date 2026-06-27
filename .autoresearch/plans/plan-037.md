# Plan EXP-037: SE channel attention (r=16, all 9 blocks) with near-identity init
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-037.md

## Milestones

### Milestone 1: SE implemented with law-compliant init, sanity-checked
- [x] On branch `autoresearch/exp-037` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. Add `SEModule(nn.Module)` (channels, reduction=16): `fc1 = nn.Linear(C, C//16)`, `fc2 = nn.Linear(C//16, C)`; forward: `s = x.mean(dim=(2,3))` → `relu(fc1(s))` → `sigmoid(fc2(...))` → `x * s.view(N, C, 1, 1)`. In `__init__`: `fc2.weight` zero-init, `fc2.bias` fill 2.0 (sigmoid ≈ 0.881 constant at step 0 — near-identity vs deferral law), and set `skip_kaiming = True` on fc1 and fc2.
  2. In `BasicBlock.__init__`: `self.se = SEModule(out_channels)`; in forward, after `out = self.bn2(self.conv2(out))`: `out = self.se(out)` (before the shortcut add).
  3. In `ResNet._weights_init`: skip modules with `getattr(m, "skip_kaiming", False)` so the global kaiming pass does not randomize SE gates.
- [x] Sanity: AST parse OK; CPU-side param count via hand-calc check: total params must print **4,319,710** (= 4,286,026 + 33,684 SE); `git diff` touches only train.py. (Measured: params 4,319,710; fc2 weight absmax 0.0 / bias 2.0 after the kaiming pass; gate at init 0.8808; forward (4,10) OK)

### Milestone 2: Run 1 launched with gates, SE-dt GATE, and scaled watchdog
- [x] Launch gates: zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h. (Both runs: GATES_CLEAR poll 1)
- [x] Composite watchdog (15s ticks, 44): **SE-dt GATE** — at clean load (gates just cleared), if the first 3 stable windows (ticks 3–5) all read ≥ 26.5ms, kill (GATE_KILL: SE unaffordable, deficit ≥ −0.33 eats the plausible gain); contention 4 consecutive windows >29ms (scaled for expected ~24.5ms dt); STARTUP_KILL tick 10; NaN/inf; divergence eval <15% after ep5; wall cap 600s. (Gate PASSED: 24.0/24.9/24.0 — SE costs +1.7ms)
- [ ] Expected signatures: dt ≈ 24–25.5ms windows, ~123–132 epochs, VRAM ≈ 1620–1700MB, total ≈ 460–490s. Early-heat diagnostic (observation, not abort): ep1 eval should sit in the family band ~36–40 if near-identity init worked; a 20s-class ep1 (EXP-026-style init toll) flags deferral residue for analysis.

### Milestone 3: Completion and readout
- [x] Full run: rc=0, total ≤600s, eval_lines = num_epochs, params 4,319,710. (Two clean runs: 96.34 @ 129ep / 96.37 @ 128ep — below bar; fallback NOT triggered, gate passed)
- [ ] Pre-registered fallback (max 1): ONLY on GATE_KILL of Run 1 at clean load → Run 2 = SE in stage-3 blocks only (3 modules, `self.se` added only when out_channels == 256; cost ≈ 1/3, SENet ablations retain a large share of the gain in late stages). A completed-but-below-bar Run 1 gets NO fallback — that is a level read, proceed to analyze.

## Code Changes
- **train.py** (only file): SEModule class (~14 lines), one line in BasicBlock.__init__, one line in BasicBlock.forward, one-line guard in _weights_init. Why this tests the hypothesis: adds input-conditioned channel gating — the one mechanism class untried in 37 experiments with published in-domain LEVEL gains (+0.5–1.2 on CIFAR ResNets, Hu et al. CVPR 2018) — while engineered compliance with every measured law: near-identity init (deferral, EXP-018), early-dt gate (launch-bound pricing, EXP-034/026), default compile + bf16 unchanged (numerics, EXP-021), batch/momentum/aug untouched (noise, EXP-023/024).
- Risks/edge cases: (a) dt overrun ≥ +4ms → GATE_KILL, fallback to stage-3-only; (b) kaiming exclusion failure would randomize gates at init — covered by the skip_kaiming guard and the ep1 early-heat diagnostic; (c) channels_last broadcast of `x * s.view(N,C,1,1)` is standard and compile-safe; (d) torch.compile recompile risk is nil (static shapes, no attribute mutation at runtime).

## Configuration Changes
- None — all hyperparameters (LR, schedule, WD, LS 0.1, batch, augmentation) unchanged. New structural elements: SE r=16 (canonical, Hu et al.), fc2 zero-weight + bias 2.0 init (near-identity; sigmoid(2.0)=0.881). Rationale and arithmetic: knowledge/papers/squeeze-excitation-senet.md.

## Execution Environment
- Method: local composite background Bash (dual launch gates → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` → 44×15s watchdog → wait → rc + summary greps + eval tails), branch `autoresearch/exp-037`, GPU 0 only.
- Resources: VRAM ~1.7GB; 8 loader workers.
- Estimated runtime: ~460–490s total (dt 24–25.5ms → ~123–132 epochs). ~110s margin under the cap.
- Log output: `run.log` (no tee); post-hoc awk profile authoritative; delete run.log after analysis.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **SE-dt GATE (clean load)**: first 3 stable windows all ≥ 26.5ms → GATE_KILL (experiment-level dt refutation, NOT contention — gates ensured a clean GPU). Triggers the pre-registered stage-3-only fallback.
- **Contention**: 4 consecutive windows >29ms AFTER the gate window passes → kill, contaminated, rerun once after gates re-clear.
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = **96.81**. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>29)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>29: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤26ms. Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check: `grep "^num_params:" run.log` = **4,319,710**; training_seconds = 300.0; eval_lines = num_epochs; on a bar-pass confirm the plateau (last ~15 evals) sits above the baseline band, not a single-eval spike.
   - Timeout: greps instant; missing run.log = infrastructure failure.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: eval_lines ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **Measured SE dt cost** (profile mean − 22.4ms) and resulting epoch count — prices channel attention on H20 permanently, feeds the launch-bound law.
- **Early-heat trace**: ep1/ep5 evals vs family (≈38/64) — validates or refutes the near-identity init pattern as a reusable deferral workaround.
- **Plateau level & shape** (last ~15 evals) and final_test_loss vs family (~0.185) — basin LEVEL vs transit decomposition per the max-statistic law.
- **VRAM** vs 1613.0MB baseline.
