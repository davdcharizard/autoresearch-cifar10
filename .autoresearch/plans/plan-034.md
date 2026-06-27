# Plan EXP-034: Depth-for-width at matched compute — ResNet-26 at stage widths 56/112/224
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-034.md

## Milestones

### Milestone 1: Architecture change implemented and param-count verified
- [x] On branch `autoresearch/exp-034` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. `NUM_BLOCKS = 3` → `NUM_BLOCKS = 4  # ResNet-26 = 6*4+2`.
  2. Replace `WIDTH_MULT = 4` with `STAGE_WIDTHS = (56, 112, 224)  # 4x widths x sqrt(3/4): conv FLOPs ~1.02x, params +4.3% vs baseline`.
  3. `ResNet.__init__` signature: `width_mult=1` param → `stage_widths=(16, 32, 64)`; body `w1, w2, w3 = stage_widths` (drop the `16/32/64 * width_mult` line). Everything else in the class untouched (BasicBlock pad-shortcut handles 56→112→224 transitions identically).
  4. Model construction: `ResNet(NUM_BLOCKS, NUM_CLASSES, STAGE_WIDTHS)`; print line → `f"ResNet-{6 * NUM_BLOCKS + 2} (widths {STAGE_WIDTHS}) | params: {num_params:,}"`.
  5. NOTHING else changes: all recipe constants, transforms, loaders, schedule, compile+warmup, timed step, eval byte-identical to baseline.
- [x] Sanity: AST parse OK; CPU param count check `uv run python -c "from train import ResNet; print(sum(p.numel() for p in ResNet(4, 10, (56, 112, 224)).parameters()))"` must print **4,469,538** (hand-computed; same formula reproduces baseline's exact 4,286,026).
- [x] `git diff` shows only the 4 edit sites in train.py.

### Milestone 2: Run 1 launched with gates and depth-aware watchdog
- [ ] Launch gates (infra-errors): zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h.
- [ ] Composite watchdog (15s ticks, 44 max): standard kills (contention 4 consecutive windows >27ms; STARTUP_KILL tick 10; NaN/inf; wall cap 600s; divergence eval <15% after ep5) PLUS the **early-dt gate**: if the windowed dt (Δpct×3000/Δstep) exceeds **24.5ms on 3 consecutive windowed ticks within the first ~10 ticks** (clean load, no contention signature), kill as `GATE_KILL` — the design point is infeasible at matched compute, do not burn the full budget.
- [ ] Early readout target: dt ≈ 22.9–24ms (compute +2.1%, elementwise/launch overhead +~17% on 12 vs 9 blocks, partially fused by inductor).

### Milestone 3: Completion and readout
- [ ] Full run: rc=0, total ≤600s (est. ~475–510s; compile of the deeper graph may add ~5–10s startup, uncharged), epochs ≈ 130–139, eval_lines = num_epochs, params 4,469,538.
- [ ] Pre-registered fallback (counts as Experimental Adjustment, max one): if Run 1 dies on GATE_KILL with clean load, run ONCE at `STAGE_WIDTHS = (48, 96, 192)` (FLOPs 0.75×, dt ~19–20ms, ~3.3M params — tests the depth direction with an epoch surplus instead; the params drop is a recorded confound). If that also gate-kills (<24.5 not met — would indicate launch-overhead dominance), stop: outcome completed with the gate data, verdict no-improvement is not available without a metric → classify per analyze skill with whatever ran (precedent EXP-026: gate-killed variants carry no metric; the full-run variant's metric is recorded).

## Code Changes
- **train.py** (only file): 2 constants, 1 constructor signature/body, 1 construction+print line. Why this tests the hypothesis: the ONLY change is moving capacity from width into depth at ≈matched conv FLOPs (1.021×) and ≈matched params (+4.3%) — if composition depth adds decision-boundary expressivity per unit compute (EXP-032 diagnosis; EXP-008 mirror datum), the converged plateau LEVEL rises, which is the one thing the max-statistic rewards.
- Risks/edge cases: (a) 56 is 8-aligned but not 64-aligned — cudnn/tensor-core efficiency could be sub-linear → caught by the dt gate; (b) 12 sequential blocks add launch/BN/ReLU overhead beyond the FLOPs ratio → dt gate; (c) deeper nets shift early trainability — heat-preservation law says watch ep1 eval (family ~38%; a much lower ep1 with normal dt is signal, not abort); (d) compile time grows with graph size — lands in startup (uncharged), wall cap has ~100s margin.

## Configuration Changes
- NUM_BLOCKS: 3 → 4 (ResNet-20 → ResNet-26). Anchor: He-2015 CIFAR depth ladder is monotone 20→110 at fixed width; we sit at its minimum depth with the width dial already optimized (EXP-001/002/005/007 bracket 4× as the width optimum).
- STAGE_WIDTHS: (64, 128, 256) → (56, 112, 224) = ×0.875 ≈ √(3/4), chosen so blocks×width² (∝ conv FLOPs, per stage and total) is 1.021× baseline. All widths 8-aligned (112/224 are 16-aligned).
- All training constants byte-identical (peak 0.4, warmup 0.15, batch 512, WD 5e-4 selective, LS 0.1, TA+RE, bf16, compile DEFAULT).

## Execution Environment
- Method: local composite background Bash (dual launch gates → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` → 44×15s tick watchdog with dt gate → wait → rc + summary greps + eval tails), branch `autoresearch/exp-034`, GPU 0 only.
- Resources: VRAM ~1.7GB (slightly above baseline 1613MB — more activations at deeper/narrower shape, well within H20); 8 loader workers.
- Estimated runtime: ~475–510s total (300 charged + ~15–20s startup + ~145s evals/stalls). ~90–125s margin under the 600s cap.
- Log output: `run.log` (no tee); post-hoc awk profile authoritative; delete run.log after analysis.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill (STARTUP_KILL).
- **Early-dt GATE_KILL**: 3 consecutive windowed ticks >24.5ms within first ~10 ticks at clean load → kill (~90–150s spent); triggers the Milestone-3 fallback once.
- **Contention**: 4 consecutive windows >27ms → kill, contaminated, rerun once after both launch gates re-clear.
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code/architecture error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = **96.81**. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16 — only TRUE effects ≥ +0.3 clear the bar.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>27)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>27: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤24.5ms AND num_epochs consistent with the measured dt (epochs ≈ 139×22.4/mean_dt ±4). Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check: `grep "^num_params:" run.log` = **4,469,538** (fallback variant (48,96,192): **3,284,986**, same hand-calc method); training_seconds = 300.0; eval_lines (`tr '\r' '\n' < run.log | grep -c "eval ep"`) = num_epochs; model line prints `ResNet-26 (widths (56, 112, 224))`.
   - Timeout: verification greps are instant; treat missing run.log as infrastructure failure.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: eval_lines ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **dt at depth 26** (windowed mean from the awk profile) — the depth-overhead datum, recorded regardless of verdict; calibrates any future depth/shape probe.
- **ep1 eval** vs family ~38% — early-trainability signature of depth at fixed heat.
- **Plateau shape**: last-15-evals mean and test_loss level vs baseline family (~96.6 / ~0.185) — distinguishes level shift from transit effects.
- total_seconds, startup_seconds (compile growth), peak_vram_mb, num_epochs.
