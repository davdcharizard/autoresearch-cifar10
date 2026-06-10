# Plan EXP-031: Throughput-optimized Gradient Centralization (compiled + hoisted)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md

## Milestones

### Milestone 1: Code changes implemented and passing local checks
- [ ] Add a module-level centralization function + compiled handle near the top of `train.py` (after the imports / helper region, e.g. just after `lr_at_fraction`):
  ```python
  def _gradient_centralize(grads):
      # Gradient Centralization (Yong et al., ECCV 2020): return each weight grad
      # centralized to zero-mean over its fan-in dims (per output unit). Out-of-place
      # so torch.compile fuses mean+subtract into one kernel per tensor and there is
      # no in-place-mutation-under-compile writeback ambiguity.
      return [g - g.mean(dim=tuple(range(1, g.ndim)), keepdim=True) for g in grads]

  _gc_compiled = torch.compile(_gradient_centralize)
  ```
- [ ] **Hoist** the weight-param list ONCE, right after the optimizer is built (after train.py L198, before the `while` loop L213):
  ```python
  # EXP-031: pre-collect the conv/linear weight params (ndim>1, the 23 GC targets) once,
  # so the per-step GC neither re-iterates the BN/bias params nor rebuilds reduction dims.
  gc_params = [p for p in model.parameters() if p.ndim > 1]
  ```
- [ ] Insert the compiled GC call between `loss.backward()` (L237) and `optimizer.step()` (L238):
  ```python
          loss.backward()
          # EXP-031: throughput-optimized Gradient Centralization. Compiled+hoisted to
          # kill the ~1ms/step eager per-op host overhead that cost EXP-030 ~3 epochs.
          centralized = _gc_compiled([p.grad for p in gc_params])
          for p, cg in zip(gc_params, centralized):
              p.grad = cg
          optimizer.step()
  ```
- [ ] `git diff --name-only` shows ONLY `train.py`.
- [ ] AST parse clean (`uv run python -c "import ast; ast.parse(open('train.py').read())"`).
- [ ] Smoke check (`uv run python`, see below) passes: correctness + 23 targets + param count + compiled-not-slower.

### Milestone 2: Experiment launched and confirmed running
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Within ~60s: `run.log` shows `params: 4,299,866`, clean compile (model + the extra GC-fn compile on step 1), step lines, no NaN, loss decreasing.

### Milestone 3: Run completes; throughput-neutrality confirmed (THE KEY CHECK)
- [ ] Run exits 0 and prints the summary block.
- [ ] **Confirm throughput RESTORED** — `num_epochs ≈ 91` and `dt ≈ 8ms` (vs EXP-030's 88 ep / 9ms). This is the central hypothesis of the experiment: the compile+hoist fix removed the GC loop's host overhead. If epochs are still ~88 / dt ~9ms, the fix did NOT take (compile did not cut the overhead) → note as a failed fix, NOT a fair GC test.
- [ ] `total_seconds < 600`.

## Code Changes
- **train.py — add compiled GC helper (module scope)**: `_gradient_centralize(grads)` returns out-of-place centralized grads; `_gc_compiled = torch.compile(_gradient_centralize)` (DEFAULT mode — not reduce-overhead — to avoid CUDA-graph static-address pitfalls, since `optimizer.zero_grad()` defaults to `set_to_none=True` so grad tensors are reallocated each step).
- **train.py — hoist `gc_params` once** before the training loop: the 23 `ndim>1` conv/linear weights. Removes per-step iteration of the ~40 BN γ/β + bias params and the per-step ndim check.
- **train.py — call site between backward and step**: compute `centralized = _gc_compiled([p.grad for p in gc_params])` then reassign `p.grad = cg`. Out-of-place + reassign guarantees the centralized grad reaches `optimizer.step()` (no reliance on in-place writeback through compile). SGD reads `p.grad` at step time; reassigning the grad tensor object is the standard supported pattern and does not disturb momentum buffers (keyed on param, not grad identity).

  **Why this tests the hypothesis**: identical GC *math* to EXP-030 (per-output-unit zero-mean over fan-in) — ONLY the execution is optimized. EXP-030 tied baseline + improved loss at a 3-epoch handicap caused purely by eager per-op host overhead; `torch.compile` fuses the 23 mean+subtract pairs into one compiled call, collapsing that overhead. If epochs return to ~91, GC finally gets its throughput-neutral fair test.

  **Risks/edge cases**: (a) torch.compile may not fully eliminate the overhead in default mode (kernels still launched, just issued from compiled code without per-op Python dispatch) → dt may land ~8.3–8.5ms / ~89–90 ep; still a far fairer test than 88, document realized epochs. (b) The GC-fn compiles on step 1 inside the timed region (~1–3s one-time, charged to budget) — negligible (<<1 epoch). (c) `set_to_none=True` reallocates grads each step → addresses change → reduce-overhead/CUDA-graph would be INVALID here, hence default mode (correct choice). (d) Dynamo guards on shape/dtype not address → one compiled artifact reused across steps (no per-step recompile). (e) If compile errors on the grad-list input, fall back to the hoist-only eager form (brainstorm Idea 2) — record as an Experimental Adjustment. (f) Magnitude: even fully throughput-neutral, GC's gain may sit within the ±0.2pp noise floor on this shallow net → no-improvement (graceful).

  **Smoke test** (`uv run python`):
  1. `from train import ResNet, _gc_compiled` (and torch). Build `model = ResNet(3, 10, width_mult=4).cuda()`.
  2. Assert `sum(p.numel() for p in model.parameters()) == 4_299_866`.
  3. `gc_params = [p for p in model.parameters() if p.ndim > 1]`; assert `len(gc_params) == 23`.
  4. One forward+backward on a random `(8,3,32,32)` cuda batch with CE loss (use `model`, eager — compile not needed for the model in the smoke test).
  5. `centralized = _gc_compiled([p.grad for p in gc_params])`; for each returned `cg`, assert its per-output-unit mean ≈ 0: `cg.mean(dim=tuple(range(1, cg.ndim)))` all `< 1e-5` in abs.
  6. Confirm the 1-D params (BN γ/β, biases) are NOT in `gc_params` (their grads are never touched) — assert every `gc_params` tensor has `ndim >= 2` and the count of `ndim<=1` params with grads is > 0 (sanity that we correctly excluded them).
  7. Quick timing sanity (optional but record): time 30 iters of eager GC (`g.sub_(g.mean(...))`) vs 30 iters of `_gc_compiled` (after a warmup call) on the same grads; assert compiled wall-time ≤ eager wall-time (or note the ratio). Catches the case where compile is a net loss.

## Configuration Changes
- (none — pure execution optimization of EXP-030's GC. PEAK_LR 0.2, WARMUP_FRAC 0.05, batch 128, WD 1e-4, LS 0.1, Cutout 16, TA, Nesterov, momentum 0.9, cosine-to-0, params 4,299,866, seed 42 all unchanged. Same GC math as EXP-030.)

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background, with a `Monitor` watch on run.log for the summary + NaN/error + dt.
- Resources: single NVIDIA H20 (GPU 0). VRAM ~0.5 GB (out-of-place GC allocates 23 small weight-sized temporaries per step — negligible vs activations; expect peak_vram ≈ baseline).
- Estimated runtime: ~380–405s total. Must stay < 600s.
- Log output: `run.log` in project root via redirection; source of truth.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf loss** at any point → kill, treat as failed.
- **torch.compile error** on the GC function at step 1 (traceback in run.log) → kill; fall back to hoist-only eager GC (Idea 2) as an Experimental Adjustment and relaunch (counts as the code-fix retry).
- **Loss not decreasing after warmup** → kill.
- **No output / hang**: no new step lines for >120s → kill.
- **Wall-clock runaway**: process past ~580s → kill.
- CUDA OOM (not expected) → kill, treat as failed.

## Verification Protocol

### Verification Procedure
Baseline (from `exp-index.sh baseline`) = **96.22**, pass threshold **best_test_acc ≥ 96.32**. Run conditions in order; stop at first failure.

1. **Cond 1 — primary metric clears bar.** After completion:
   `grep -aE "^best_test_acc:|^peak_vram_mb:|^total_seconds:|^num_epochs:|^num_steps:|^num_params:" run.log`
   PASS iff `best_test_acc ≥ 96.32`. Empty `best_test_acc:` ⇒ crash (`tail -n 50 run.log`) → crash verdict.
2. **Cond 2 — clean completion within budget.** PASS iff summary block printed, `grep -c Traceback run.log` == 0, `total_seconds < 600`.
3. **Cond 3 — no constraint violations.** PASS iff: `git diff --name-only` lists only `train.py`; `num_params == 4,299,866`; eval-count == num_epochs (`grep -c "eval ep" run.log` == num_epochs); no new deps (pure torch + torch.compile); seed 42 unchanged.

**MANDATORY attribution note (epoch-wall + FLOPs-neutral-≠-wall-clock-neutral, EXP-015/024/030):** record `num_epochs` and mean `dt`. THIS IS THE CRUX — the experiment's whole purpose is to restore throughput vs EXP-030 (88 ep / 9ms):
- epochs ~91 & dt ~8ms → throughput RESTORED → GC gets its fair test → the accuracy delta vs baseline is a clean read on GC's intrinsic merit.
- epochs ~89–90 & dt ~8.3–8.5ms → PARTIAL restore → fairer than EXP-030 but note the residual ~1–2 epoch handicap.
- epochs still ~88 & dt ~9ms → the compile fix did NOT cut the overhead → this is a FAILED FIX, not a fair GC test; report as such and reconsider the approach (hoist-only / different impl).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -a "^peak_vram_mb:" run.log` — expect ~0.5 GB.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — expect ~91 / ~35,500 (the throughput-restore check vs EXP-030's 88 / 33,953).
- final_test_loss: `grep -a "^final_test_loss:" run.log` — compare to baseline 0.195 and EXP-030's 0.1934.
