# Plan EXP-049: Augmentation cooldown (EXP-034) + Gradient Centralization (EXP-031) combined
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-049.md

Baseline = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. This experiment combines the two best throughput-neutral near-misses unchanged: the EXP-034 augmentation cooldown (@0.10, the only ≥baseline result, 96.26) and the EXP-031 compiled+hoisted Gradient Centralization (loss 0.1894, top-1 96.14). Both are proven individually implementable with no code interaction; the bet is the two orthogonal sub-noise levers add to clear +0.1.

## Milestones

### Milestone 1: Code changes implemented and smoke-tested
- [ ] Add the 3 Gradient Centralization edits (module-level compiled centralizer, hoisted `gc_params`, call site between `backward()` and `step()`) — identical to EXP-031.
- [ ] Add the 4 augmentation-cooldown edits (`COOLDOWN_FRAC=0.10`, `train_tf_clean`, epoch-boundary `aug_cooled` swap, Cutout gate) — identical to EXP-034.
- [ ] Smoke check: `python -c "import ast; ast.parse(open('train.py').read())"` passes; `git diff --name-only` = `train.py` only.
- [ ] Smoke check: a short import/instantiate run confirms `num_params == 4,299,866` (unchanged), GC target count = 23 (`ndim>1` params), and `train_tf_clean` contains no `TrivialAugmentWide`.

### Milestone 2: Experiment running and throughput-neutral
- [ ] Launch `uv run train.py > run.log 2>&1` on the idle GPU; confirm process started and run.log is being written.
- [ ] Early signal: dt steady ~8ms (NOT 14-16ms — that would signal a CUDA-graph break) and ep1 test_acc in the normal ~45% range.

### Milestone 3: Run completes and is verified
- [ ] Run prints the summary block; `total_seconds < 600`; the `>>> aug cooldown ON` marker fired once near frac 0.90.
- [ ] Extract `best_test_acc`, `num_epochs`, `dt` distribution, `peak_vram_mb`; compare to bar 96.32.

## Code Changes
- **train.py** — two independent, individually-proven edit sets (no interaction between them):

  **A. Gradient Centralization (EXP-031, 3 edits):**
  1. Module-level, after `cutout_batch`:
     ```python
     def _gradient_centralize(grads):
         # Centralize each weight grad to per-output-unit zero-mean over fan-in dims.
         return [g - g.mean(dim=tuple(range(1, g.ndim)), keepdim=True) for g in grads]
     _gc_compiled = torch.compile(_gradient_centralize)  # DEFAULT mode: zero_grad reallocates grads each step
     ```
  2. After the model is built and before the training loop, hoist the GC targets once:
     ```python
     gc_params = [p for p in model.parameters() if p.ndim > 1]  # 23 conv/fc weights; skips BN/bias
     ```
  3. Call site, between `loss.backward()` and `optimizer.step()`:
     ```python
     loss.backward()
     centralized = _gc_compiled([p.grad for p in gc_params])
     for p, cg in zip(gc_params, centralized):
         p.grad = cg
     optimizer.step()
     ```
     Out-of-place + reassign (not in-place) — sidesteps the Inductor clone-writeback no-op ambiguity (EXP-031). GC runs every step, including the clean cooldown phase (it is a full-run regularizer).

  **B. Augmentation cooldown (EXP-034, 4 edits):**
  1. Hyperparameter block: `COOLDOWN_FRAC = 0.10` (disable strong aug for the final 10% of the time budget).
  2. After `train_tf` is defined, add the clean transform (full pipeline minus TrivialAugment):
     ```python
     train_tf_clean = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean, std),
     ])
     ```
  3. Before the training loop: `aug_cooled = False`. At the top of the epoch loop (after `epoch += 1; model.train()`), fire once at the epoch boundary:
     ```python
     if not aug_cooled and (total_training_time / TIME_BUDGET_S) >= (1 - COOLDOWN_FRAC):
         train_set.transform = train_tf_clean
         aug_cooled = True
         print(f"\n>>> aug cooldown ON at ep {epoch} frac {total_training_time / TIME_BUDGET_S:.2f}")
     ```
     The swap propagates to forked dataloader workers because `persistent_workers` is not set (default False → workers re-fork each epoch), verified in EXP-033/034.
  4. Gate the GPU Cutout call: `if not aug_cooled: inputs = cutout_batch(inputs, CUTOUT_SIZE)`.

  **Why this tests the hypothesis**: GC lowers loss / better-conditions the weights throughout training but its top-1 gain is masked by the aug-train↔clean-test mismatch; the cooldown removes that mismatch in the low-LR tail, giving GC's better-conditioned state a chance to surface as top-1. Both are throughput-neutral (dt~8ms, ~91 ep), so the comparison is confound-free.

  **Risks / edge cases**: (a) the two levers may simply not add (most likely → no-improvement ~96.2–96.3); (b) GC's `p.grad` reassignment is outside the model's reduce-overhead forward graph, so it must NOT change model dt — watch for a dt jump to 14-16ms (would indicate an unexpected graph break); (c) leave the tail LR schedule untouched (frozen near-zero) — EXP-035 showed reheating the clean tail regresses.

## Configuration Changes
- `COOLDOWN_FRAC`: (new) -> `0.10` (EXP-034's best window; later/shorter beats EXP-033's 0.15).
- Gradient Centralization: applied to all 23 `ndim>1` weight grads every step (EXP-031 settings, unchanged).
- No change to: width (k=4), depth, optimizer (Nesterov SGD m0.9, WD 1e-4), PEAK_LR 0.2, WARMUP_FRAC 0.05, LABEL_SMOOTHING 0.1, BATCH_SIZE 128, CUTOUT_SIZE 16, seed 42, compile mode (model stays `reduce-overhead`; GC uses a separate DEFAULT-mode compiled callable).

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, launched in background (Bash `run_in_background: true`).
- Resources: single NVIDIA H20. Shared node with GPUs 0/1; a foreign job intermittently occupies GPU 0 → check `nvidia-smi` and launch on the idle GPU (GPU 1 has been idle this session).
- Estimated runtime: ~300s training + ~2s startup + eval overhead ≈ 400s wall (< 600s limit).
- Log output: all stdout/stderr → `run.log` in the project root (the executor reads this; the background task's own output file stays empty due to redirection).
- Tool skill: none (local run).

## Abort Criteria
- Loss goes NaN/inf, or training diverges (debiased loss climbing for many steps).
- dt jumps to ~14-16ms and stays there (CUDA-graph break from the GC reassignment) → epochs would collapse; kill and diagnose rather than waste the budget.
- No output / log not advancing for > 3 minutes after launch.
- Total wall-clock approaching 600s without a summary block → kill (constraint breach).
- The `>>> aug cooldown ON` marker never fires by frac 0.95 → the cooldown gate is broken; the run is invalid for the hypothesis.

## Verification Protocol

### Verification Procedure
Run after the experiment completes; stop at the first failed necessary condition.

1. **Get baseline**: `bash .../exp-index.sh baseline experiment-indices/improve-cifar10-test-accuracy.tsv` → baseline = 96.22, so the bar is **96.32**.
2. **Necessary condition 1 — `best_test_acc >= 96.32`**:
   `grep -aE "^best_test_acc:" run.log` → parse the float. PASS iff `>= 96.32`; otherwise no-improvement. (timeout: run must have finished; if `best_test_acc:` is absent the run crashed → inspect `tail -n 50 run.log`.)
3. **Necessary condition 2 — clean completion within budget**:
   `grep -aE "^best_test_acc:|^total_seconds:|^num_epochs:|^num_params:" run.log`. PASS iff the summary block printed, `total_seconds < 600`, and `num_params == 4,299,866` (unchanged). No NaN/traceback in `run.log`.
4. **Necessary condition 3 — no hard-constraint violations**:
   `git diff --name-only` = `train.py` only; `prepare.py`/eval untouched; `evaluate()` called once per epoch (loop structure unchanged); no new deps (GC + cooldown use only torch/torchvision already present); seed 42 unchanged; deterministic — no seed hacking.
5. Confirm the cooldown fired: `grep -a ">>> aug cooldown ON" run.log` shows exactly one marker near frac ~0.90.
6. Remove `run.log` before the next experiment.

### Informational Metrics (Optional)
- best_test_acc / delta vs 96.22: `grep -aE "^best_test_acc:" run.log`.
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — confirm ~91 ep (throughput-neutral, no epoch confound).
- dt distribution: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect steady ~8ms.
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — compare to baseline 0.195 / EXP-031 0.1894 (does GC's loss gain persist in the combination?).
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`.
- pre-cooldown vs post-cooldown test_acc trajectory: the eval lines around the `>>> aug cooldown ON` marker (how much the clean tail climbed).
