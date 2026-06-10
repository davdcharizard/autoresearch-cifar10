# Plan EXP-066: Progressive resolution scheduling (train early @24×24, finish @32×32)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-066.md

## Closed-axis check
This is NOT a retune of any mapped scalar/schedule/aug/capacity/optimizer/norm lever. It introduces a NEW axis: input spatial resolution as a function of training-time fraction.

**Relation to the High-importance epoch-wall insight** (project-insights: "ANY change that adds non-trivial FLOPs OR sequential layers OR restructures the graph costs epochs → under-train"): this plan moves the OPPOSITE direction — it REDUCES conv FLOPs during the early phase (24×24 = 0.5625× the 32×32 FLOPs) to BUY epochs, exactly the mechanism behind the project's clearest win (EXP-003 GPU-Cutout, +0.58pp, throughput→epochs). No architecture/layer/width change (params unchanged); the global-avg-pool head is resolution-agnostic. The eval path is untouched (frozen 32×32) and the schedule ENDS at 32×32 so the converged model + BN stats match the eval distribution (FixRes-correct).

**Two identified risks, both early-gated**: (1) if the net is more LAUNCH-bound than compute-bound even under cudagraph (cf. EXP-058: narrower nets still rose dt), the 24×24 dt reduction may be small → fewer bought epochs → null (not a regression). (2) Two input shapes under `torch.compile(reduce-overhead)` could break or fall out of CUDA-graph capture (EXP-042 signature: dt 8→14-16ms) — caught at the ep8 gate.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py: add two constants near CUTOUT_SIZE (L28): `LOW_RES = 24`, `RESIZE_FRAC = 0.5`.
- [ ] train.py training loop (between the `targets = targets.to(...)` line ~L233 and the existing `inputs = cutout_batch(inputs, CUTOUT_SIZE)` at L231): replace the unconditional cutout call with a resolution-scheduled block (see Code Changes). Add a one-time `>> FULL-RES PHASE` marker print when the switch to 32 fires.
- [ ] Smoke: `python -c "import ast; ast.parse(open('train.py').read())"` OK; `git diff --name-only` == train.py only; confirm the resize runs BEFORE `compiled_model(inputs)` and OUTSIDE the autocast block; confirm cutout size scales with resolution.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate A (~ep8, still in the 24×24 phase): dt is NOT elevated to 14-16ms (would signal cudagraph break → abort per EXP-042). Record the realized 24×24 dt (hypothesis: < 8ms, showing the FLOP saving; even ≈8ms is acceptable = no speedup but no break). No NaN; eval test_acc climbing.
- [ ] Gate B (after frac≥0.5): confirm the `>> FULL-RES PHASE` marker fired and a second dt regime (~8ms for 32×32) appears. Confirm epochs accumulate faster than baseline if Gate A showed a 24×24 speedup.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract metrics, compare to baseline 96.45 / bar 96.55. Record num_epochs (expect > 91 iff the 24×24 phase realized a speedup) and the dt distribution (two regimes).

## Code Changes
- **train.py — constants (after L28)**: add
  ```python
  LOW_RES = 24      # EXP-066: early-phase training resolution (full eval res = 32)
  RESIZE_FRAC = 0.5 # fraction of the time budget trained at LOW_RES before switching to 32
  ```
- **train.py — training loop** (replace the single `inputs = cutout_batch(inputs, CUTOUT_SIZE)` line at L231, keeping everything else byte-identical):
  ```python
  # Progressive resolution (EXP-066): train early at LOW_RES to buy epochs, finish at 32
  # so the converged model + BN stats match the frozen 32x32 eval (FixRes-correct).
  cur_res = LOW_RES if (total_training_time / TIME_BUDGET_S) < RESIZE_FRAC else 32
  if cur_res != 32:
      inputs = F.interpolate(inputs, size=cur_res, mode="bilinear", align_corners=False)
      inputs = inputs.contiguous(memory_format=torch.channels_last)
      cur_cutout = max(1, round(CUTOUT_SIZE * cur_res / 32))
  else:
      cur_cutout = CUTOUT_SIZE
      if not full_res_announced:
          print(f"\n>> FULL-RES PHASE @ ep {epoch} (frac {total_training_time / TIME_BUDGET_S:.3f})")
          full_res_announced = True
  inputs = cutout_batch(inputs, cur_cutout)
  ```
  and initialize `full_res_announced = False` alongside the other pre-loop counters (near `best_acc = 0.0`, ~L219).
  - **Why**: the resize is a GPU bilinear downscale applied OUTSIDE the compiled forward and OUTSIDE autocast (on the float32 input), so reduce-overhead captures one stable CUDA-graph per phase-shape (24 then 32) — respecting the EXP-042 rule (no data-dependent branch INSIDE forward; shape is constant within a phase). Cutout scales proportionally (16·24/32 = 12 px) so the hole stays at the same 50%-side fraction. `.contiguous(memory_format=channels_last)` keeps the input layout consistent with the compiled graph's expected stride.
  - **Risks/edge cases**: (a) exactly one 24→32 transition at frac 0.5 → one one-time graph recapture (a single slow step, negligible vs Σdt); (b) if cudagraph breaks, dt doubles from step 1 of the low-res phase (Gate A abort); (c) BN running stats during the low-res phase are 24×24 — re-adapted by the ~48-epoch 32×32 tail (the whole point); the tail must be long enough (RESIZE_FRAC=0.5 leaves half the budget at full res).

## Configuration Changes
- `LOW_RES = 24`, `RESIZE_FRAC = 0.5`. Rationale: 24×24 is a moderate first probe (0.5625× FLOPs — a real saving without the aggressive detail loss of 16×16); a 50/50 split gives a substantial cheap-epoch early phase AND a long full-res tail (~48 ep) to re-sharpen features and re-fit BN to the 32×32 eval distribution. All else byte-identical to EXP-054 (k=4 WideResNet-20, AugMix-p0.5, cosine peak0.2/warmup0.05/Nesterov/LS0.1/WD1e-4, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi`; relaunch on contention per infra-errors — the Σdt budget REQUIRES an uncontended GPU).
- Estimated runtime: Σdt = 300s by construction. Wall: the low-res phase has LIGHTER eval-adjacent load but the same 92 evals; expect wall ≈ 560–595s. If the 24×24 phase buys epochs, num_epochs > 91 and there are MORE evals → watch the 600s wall (infra-errors recurring-breach: AugMix recipe is wall-tight; a small overrun on base variance is no-improvement-not-invalid per precedent, but monitor).
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- dt elevated to ≥13ms sustained from the start of the 24×24 phase (cudagraph break, EXP-042 signature): kill, record, treat as failed/crash.
- Loss NaN/inf or eval test_acc not climbing by ep5.
- F.interpolate or compile raises at runtime (e.g., reduce-overhead multi-shape error): capture traceback, treat as code error (1 retry to adjust, e.g., fall back to a single recapture or `mode="bilinear"` args) per the execute skill.
- dt drifts ≫ expected due to contention (GPU not idle): kill, relaunch on clean GPU.
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866`, `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric is trustworthy → no-improvement per EXP-061/065 precedent, NOT invalid.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only; prepare.py/eval untouched; evaluate() once/epoch (loop structure unchanged); no new deps (F.interpolate/torch already imported); seed 42 unchanged; ran uncontended (check dt regimes).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / cudagraph-break abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs` / `num_steps`: `grep -aE "^num_epochs:|^num_steps:" run.log` — **KEY diagnostic**: > 91 confirms the 24×24 phase bought epochs (the mechanism realized); ≈91 means dt was launch-bound (no speedup → expected null).
- dt distribution (two regimes): `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | sort | uniq -c` — expect a lower-dt cluster (24×24 phase) + an ~8ms cluster (32×32 tail). A single ~8ms cluster = no realized speedup; a 14-16ms cluster = cudagraph break.
- Full-res switch marker: `grep -a ">> FULL-RES PHASE" run.log` — confirms the schedule fired and at which epoch/frac.
- `final_test_loss`, `peak_vram_mb`: `grep -aE "^final_test_loss:|^peak_vram_mb:" run.log` (peak_vram may rise slightly — 2 cudagraphs).
