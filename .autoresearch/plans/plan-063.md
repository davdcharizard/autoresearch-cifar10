# Plan EXP-063: Stream-parallel two-member ensemble (concurrency-funded diversity), probe-gated
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-063.md

Baseline 96.71 @ 1990397 (bar ≥ 96.81; family mean 96.57, σ 0.16). Hypothesis and branches per brainstorm-063. The launch decision is made UNCHARGED by a two-stream GPU probe: the project's HIGH-importance ensemble entry requires the gain-vs-dilution inequality be shown BEFORE running — the probe does exactly that.

## Milestones

### Milestone 1: Code changes + CPU sanity
- [x] train.py: two independent 4x ResNet-20 members (sequential construction under the fixed seed 42 → distinct inits), both channels_last + separately compiled; loader batch 1024 / 16 workers; timed step = H2D 1024 → split 512/512 → member forwards+backwards on two CUDA streams (event-ordered, joined before the two optimizer steps) → synchronize; per-member nesterov SGD + selective WD + shared time-keyed cosine lr_at (recipe constants byte-identical to baseline); `MeanEnsemble` module (logit mean) evaluated via `evaluator.evaluate(ensemble, device)` every 4th loop-epoch (loop-epoch = one 1024-batch pass = 48 steps; cadence is BELOW the once-per-epoch ceiling; ONLY the ensemble is ever evaluated — no member evals, keeping total validation ≤ 1/epoch); compile warmup 3 iters per member + 2 uncharged two-stream rehearsal iters (no optimizer.step)
- [x] CPU sanity (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp063_sanity.py`) — 10/10 pass: (a) two members constructed, params 4,286,026 EACH, inits differ (max |Δ| > 0.01 on conv1); (b) MeanEnsemble forward == exact mean of member logits; (c) batch split halves disjoint and shape-correct; (d) 3 joint CPU steps (no streams on CPU — sequential fallback path), both member losses decreasing; (e) lr_at unchanged vs baseline values at p ∈ {0.05, 0.15, 0.5, 1.0}

### Milestone 2: Two-stream GPU probe (uncharged, ~3 min) — THE LAUNCH GATE
- [x] Gate: GPU-0 compute apps == 0 AND host load < 40; record load — **gate passed at apps=0, load1=10.1** (cleanest branch of the criterion applies)
- [x] /tmp/exp063_gpu_probe.py: ran clean — **P1 = 22.48ms** (inside family band — probe itself clean), **P2 = 40.70ms**, **ratio = 1.810**, E = 4.26s (40×256 ensemble eval pass), VRAM 1,795MB
- [x] **Pre-registered launch criterion**: FAILED decisively — at load < 30 requires P2 ≤ 23.5; measured 40.70. **NO LAUNCH → verdict `invalid` (NaN), concurrency cost-closure.** Attribution diagnostic (uncharged, /tmp/exp063_eager_diag.py): eager-mode ratio 1.820 vs compiled 1.810 — identical → serialization is NOT a torch.compile artifact; the two streams contend on the serial kernel-dispatch chain (the same latency-bound resource that bounds the single-member step). Idle SMs are unreachable capacity on this stack.
- [ ] If LAUNCH: record probe-revised bands — steps ∈ [300000/(P2+1.5), 300000/(P2+0.1)]; loop-epochs = steps/48 ±2; evals = floor(loop-epochs/4) ±2; D0 ∈ [P2−1.3, P2+1.3] (two-sided: offset can invert, EXP-062); wall projection = 300 + startup + evals×(E+0.3) + stalls ≤ 600 — if projection > 580, thin evals to every 5th loop-epoch BEFORE launch

### Milestone 3: Gated composite run
- [ ] /tmp/exp063_composite.sh (exp046-standard single-phase, retargeted): dual gates (apps==0 AND load<60, poll 30s×240) → rm -f run.log → train background → watchdog 44×15s; GATE_KILL D0 > max(26, P2+2.5); contention 4 consecutive > max(26, D0×1.25); TAIL_THRESH=THRESH; STARTUP_KILL tick 12 (NOTE: two compiles — if startup > 50s expected from probe, extend to tick 14); NaN guard; divergence guard ensemble acc < 20 after loop-ep 12; WALL_CAP backstop
- [ ] Launch via Bash run_in_background (stdout → /tmp/exp063_composite_run1.log) + until-grep watcher + TaskOutput(block=true)
- [ ] Clean completion: RC=0, no kill markers, D0/windows/steps in bands

### Milestone 4: Verification (first-failure-stop; integrity gates Condition 1)
- [ ] Integrity: RC=0; step LEDGER in probe band (binding contamination gate); loop-epochs/evals consistent with cadence; `num_params` printed = 8,572,052 total (2 × 4,286,026 — the assert); `training_seconds: 300.0`; first-eval tripwire ≥ 30 (ensemble at ~2 member-epochs; relaxed — ledger binding); zero NaN; VRAM < 4,200MB
- [ ] Condition 1: best_test_acc ≥ 96.81 (ensemble metric — the trained system IS the ensemble, EXP-043 contract)
- [ ] Condition 2: total_seconds ≤ 600
- [ ] Condition 3: validation ≤ once/epoch (structural: every 4th loop-epoch, ensemble only)

## Code Changes

- **train.py** (only editable file):
  1. **Members**: `model_a`, `model_b` = `ResNet(NUM_BLOCKS, NUM_CLASSES, WIDTH_MULT)` constructed sequentially (RNG advances under the single fixed seed → independent inits; NO seed manipulation); both `.to(device, channels_last)`; `base_a/base_b` eager refs; `model_a/model_b = torch.compile(...)` separately. `ensemble = MeanEnsemble(base_a, base_b)` — 6-line nn.Module returning `(a(x)+b(x))/2`.
  2. **Loader**: `BATCH_SIZE = 1024` fetch (split into two 512 halves per step — each member sees a disjoint random half of each batch; over two loop-epochs each member covers ~one full dataset pass), `num_workers=16` (EXP-031 lever; demand ~45k img/s), persistent, pinned, drop_last.
  3. **Timed step** (charged semantics preserved exactly — t0 → ... → synchronize → dt): t0; H2D 1024 + channels_last; split `xa, xb = inputs[:512], inputs[512:]` (+ targets); shared `lr_at(progress)` written to both optimizers; record event on default stream; `with torch.cuda.stream(s1): s1.wait_event(ev); autocast fwd A; loss_a.backward()`; same on s2 for B; default stream `wait_stream(s1/s2)`; `opt_a.step(); opt_b.step()`; synchronize. Backward kernels follow the forward's recorded stream (PyTorch autograd stream semantics); per-step full join + synchronize makes the allocator cross-stream pattern safe.
  4. **Optimizers**: two independent `optim.SGD` with the baseline's exact two-group selective WD, nesterov, momentum 0.9, lr set per-step by the unchanged `lr_at`.
  5. **Warmup (uncharged)**: 3 compile-warmup iters per member (no optimizer.step) + 2 joint two-stream rehearsal iters (no step) so the stream path is exercised before the timed loop; zero_grad both; synchronize.
  6. **Eval block**: every 4th loop-epoch (counter check; ALSO always eval after the final partial epoch when budget exhausts): `evaluator.evaluate(ensemble, device)` — Eval untouched; print marks loop-epoch and member smooth train losses for diagnostics (no member evals). `num_params` print = sum over both members. Banner prints "2x stream-parallel members".
- **Why this tests the hypothesis**: members train concurrently on idle SMs; if P2 ≈ P1 the dilution is ~0.1 and the measured +0.3–0.5 logit-mean gain should surface above the family band.
- **Risks/edge cases**: GIL launch serialization (probe-detected); stream-autograd-compile interaction (rehearsal iters + probe dt signature = engagement proof); loader stalls at 1024 (wall-side only; 16 workers + eval thinning budgeted); two-compile startup (~25–45s, uncharged; STARTUP_KILL margin checked at M3).

## Configuration Changes
- BATCH_SIZE: 512 → 1024 fetch, split 512/member (per-member batch and ALL recipe constants unchanged — PEAK_LR 0.4, WARMUP_FRAC 0.15, cosine lr_at, momentum 0.9 nesterov, WD 5e-4 selective, LS 0.1, transforms; the gradient-noise scale per member is byte-identical to baseline)
- num_workers: NUM_WORKERS (8) → 16 (EXP-031 validated; feeds 1024-image steps)
- Eval cadence: every epoch → every 4th loop-epoch, ensemble only (ceiling-compliant thinning, EXP-031/043)
- New: two members, two streams, MeanEnsemble eval

## Execution Environment
- Method: local, GPU 0 only; probe first (M2 gate), then single gated run via /tmp/exp063_composite.sh; `uv run train.py > run.log 2>&1`
- Resources: GPU 0; VRAM ~3.2–4.0GB (2× model/optimizer/activations — well inside the goal's soft slack); host load gates (probe <40 with criterion split at 30, launch <60)
- Estimated runtime: probe ~3 min; run ≈ 520–580s total (300 charged + 25–45s two-compile startup + ~68 evals × ~2s + stalls); charged exactly 300.0s
- Log output: run.log (truth); /tmp/exp063_composite_run1.log (gate/watchdog); delete run.log after the experiment

## Abort Criteria
- M2 gate: NO LAUNCH if the pre-registered criterion fails (this is the experiment's primary falsification point — costs zero charged seconds)
- GATE_KILL: D0 > max(26, P2+2.5) at GATE_DECISION
- CONTENTION_KILL: 4 consecutive windows > max(26, D0×1.25)
- STARTUP_KILL: no step lines by tick 12 (extend to 14 if probe-measured startup > 50s)
- NaN in loss prints → kill (crash verdict); ensemble eval acc < 20% at/after loop-ep 12 → kill
- WALL_CAP backstop at watchdog window end

## Verification Protocol

### Verification Procedure
First-failure-stop; integrity gates Condition 1.

0. **Integrity** (2 min, run.log + composite log): RC=0, no kill markers; `num_steps` ∈ [300000/(P2+1.5), 300000/(P2+0.1)] (BINDING contamination gate); loop-epochs = steps/48 ±2; eval-line count = floor(loop-epochs/4) ±2 (+ final); `num_params: 8,572,052`; `training_seconds: 300.0`; first eval ≥ 30; zero NaN; VRAM < 4,200MB.
1. **Condition 1** — `grep "^best_test_acc:" run.log` ≥ 96.81 (baseline 96.71 + 0.1 via `exp-index.sh baseline`). **Near-bar protocol (EXP-052)**: any Run-1 read ≥ 96.81 triggers one byte-identical replicate; the PAIR MEAN decides. Max of the pair never decides.
2. **Condition 2** — `grep "^total_seconds:" run.log` ≤ 600.
3. **Condition 3** — eval cadence structural check (eval lines ≤ loop-epochs).

Pre-registered branches: (i) pair-mean ≥ 96.81 → improvement, commit; (ii) probe NO-LAUNCH → `invalid`/NaN, concurrency cost-closure (multiplicity axis closed on its last open funding source); (iii) run ∈ [96.41, 96.73] → no-improvement; decorrelation gain shrinks at converged member level — multiplicity axis closed completely; (iv) < 96.41 → no-improvement; stream interference or split-data regime damage (diagnose via member train-loss symmetry + step ledger); (v) infra → byte-identical relaunch ≤ 2.

Mechanism checks (informational): P2/P1 overlap ratio from probe; member smooth train losses at matched step counts vs family trajectory (dilution realized?); plateau eval count within 0.15 of best.

### Informational Metrics (Optional)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log` — expect 3,200–4,000MB
- num_epochs (loop-epochs): `grep "^num_epochs:" run.log` — expect ~270 at P2≈23 (48-step epochs)
- num_steps: probe band; num_params: 8,572,052
