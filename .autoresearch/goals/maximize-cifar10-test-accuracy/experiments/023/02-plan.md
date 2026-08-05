# Plan EXP-023: EMA self-distillation (tail-gated dark-knowledge soft targets)

- **Created**: 2026-06-30

## Goal & Bar (refs, not duplicated)
Baseline **96.38** (commit 07c3760); bar **best_test_acc ≥ 96.48** AND beat the same-session α=0 control by > the ~0.1–0.2pp noise floor, replicated on a confirmation pair. Only `train.py` editable; prepare.py frozen; no new deps; no seed hacking; **≤1 eval/epoch**; GPU 1 (`CUDA_VISIBLE_DEVICES=1`); 300s training-compute budget; 600s wall cap. Chosen idea & hypothesis: `01-brainstorm.md`; technique ref: `knowledge/references/self-distillation-kd.md`.

## Strategy summary
Add a **tail-gated knowledge-distillation loss** whose teacher is the **EMA model already maintained** for eval (EXP-002): `L = (1−α)·CE_LS(student,y) + α·T²·KL(softmax(teacher/T).detach() ‖ softmax(student/T))`, teacher = EMA logits (eval mode, no-grad), KD active only when `progress ≥ KD_START` (tail-only, where the teacher is strong and most accuracy lands). Re-add the thrice-validated off-budget torch.compile warmup (+12%) to fund the one extra (uncompiled) teacher forward. This attacks the **loss/learning-signal axis** — the one major lane untouched in 22 experiments — which EXP-022 implicated when a wholesale backbone swap tied (ceiling is recipe/data-bound, not topology). Compared same-session vs an α=0 control; reduced-LS arm guards against over-softening; confirmed on a 2nd pair if a winner.

## Milestones

### Milestone 1: Code — compile recipe + KD term + env knobs implemented, correctness-smoked
- [ ] Add `import os`; env knobs (`USE_COMPILE`, `COMPILE_MODE`, `KD_ALPHA`, `KD_T`, `KD_START`, `LS`, `SMOKE_SECONDS`), all defaulting to the EXP-008 baseline (KD off, compile off, LS 0.2).
- [ ] Re-add the off-budget `torch.compile` warmup (separate `train_fwd` handle, bf16-autocast dummies, BN snapshot/restore, no step, `cuda.synchronize()` before timer) — identical to the EXP-014/021/022 recipe.
- [ ] Add the KD loss term in the train loop (teacher = `ema_model.module` eval-mode no-grad forward; forward-KL with detached teacher as target; ×T²; fp32 softmax); gate `kd_active = KD_ALPHA>0 and progress≥KD_START and ema_started`.
- [ ] Set `ema_model.eval()` once (teacher always eval); add `SMOKE_SECONDS` path forcing EMA+KD on after step 12 so the smoke measures the heavy (KD-on) per-step cost.
- [ ] **Verify**: `CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python /tmp/exp023_smoke.py` — all correctness checks pass (§A).

### Milestone 2: Throughput pre-smoke — confirm KD-on cost anneals
- [ ] Run `SMOKE_SECONDS=25 USE_COMPILE=1 KD_ALPHA=0.5` (forces KD on after step 12) → measure heavy-step `smoke_img_s` + all-heavy `projected_epochs`.
- [ ] **Verify**: all-heavy projected_epochs ≥ ~110 (the real run is tail-only ≈50% heavy, so it will land well above the 12610-step/130-ep gate). If heavy projection < 110, raise `KD_START` to 0.6 (less heavy exposure) and/or plan to compile the teacher. Log the number in 03-execute.md.

### Milestone 3: Official same-session triple — c0 / cA / cB
- [ ] Write a fresh `/tmp/exp023_orchestrate.sh` (EXP-021/022 retry-until-clean pattern, exp023 env + logs): c0=`USE_COMPILE=1` (KD off control); cA=`USE_COMPILE=1 KD_ALPHA=0.5 KD_T=4 KD_START=0.5 LS=0.2`; cB=`USE_COMPILE=1 KD_ALPHA=0.5 KD_T=4 KD_START=0.5 LS=0.1` (reduced-LS arm).
- [ ] Retry-until-clean GPU-1 (foreign compute-app >5GB on UUID a5c2d56c, util<30 pre-check; monitor img/s & foreign mid-run; abort+retry pair).
- [ ] **Verify**: all finish < 600s wall, valid best_test_acc; cA/cB `num_steps ≥ 12610` (else under-anneal → raise KD_START or compile teacher, re-run); `git diff --quiet -- prepare.py`.

### Milestone 4: Confirmation pair + bake-in (only if a KD cell is a candidate winner)
- [ ] If best KD cell ≥ 96.48 AND (KD cell − c0) > ~0.15pp AND num_steps ≥ 12610: run a 2nd same-session pair in REVERSED order (winner-cell then c0) on a clean window.
- [ ] **Verify replication**: the lead replicates (> noise AND ≥ 96.48). If it collapses to < ~0.1pp → low-control-draw artifact → no-improvement.
- [ ] **Bake-in (bare-run reproducibility)**: if confirmed, set the winning config (`KD_ALPHA`/`KD_T`/`KD_START`/`LS`, `USE_COMPILE=1`) as the DEFAULT env values in train.py and re-verify with a BARE `CUDA_VISIBLE_DEVICES=1 uv run train.py` reproducing ≥ 96.48 — the goal's verification runs bare `uv run train.py`, so env-only wins do not count.

## Code Changes
- **train.py** (sole editable file; currently the EXP-008 baseline):
  - `import os` after `import gc`.
  - **Env knobs** after the hyperparameter block: `USE_COMPILE = os.environ.get("USE_COMPILE","0")=="1"`; `COMPILE_MODE = os.environ.get("COMPILE_MODE","default")`; `KD_ALPHA = float(os.environ.get("KD_ALPHA","0.0"))`; `KD_T = float(os.environ.get("KD_T","4.0"))`; `KD_START = float(os.environ.get("KD_START","0.5"))`; `SMOKE_SECONDS = float(os.environ.get("SMOKE_SECONDS","0"))`. Replace the `LABEL_SMOOTHING = 0.2` constant's use with `LABEL_SMOOTHING = float(os.environ.get("LS","0.2"))` (so the reduced-LS arm is env-controlled; default 0.2 = baseline).
  - **Compile + off-budget warmup** (before `t_start_training`, after EMA setup): `warmup_seconds = 0.0` then `train_fwd = model`; if `USE_COMPILE`: `train_fwd = torch.compile(model, mode=COMPILE_MODE)`; snapshot BN running buffers; 3 fwd+bwd on local-RNG dummies INSIDE `torch.autocast("cuda", dtype=torch.bfloat16)`; `optimizer.zero_grad(set_to_none=True)`; no `optimizer.step()`; restore BN buffers; `torch.cuda.synchronize()`; `t_start_training` taken AFTER. Loop uses `outputs = train_fwd(inputs)`.
  - **EMA teacher**: after creating `ema_model`, call `ema_model.eval()` (teacher always in eval mode; `update_parameters` is mode-independent; epoch-end eval already uses eval). 
  - **KD term** in the train loop, inside the existing `torch.autocast`:
    ```
    outputs = train_fwd(inputs); loss = criterion(outputs, targets)
    kd_active = KD_ALPHA > 0.0 and ema_started and (progress >= KD_START or (SMOKE_SECONDS>0 and step>12))
    if kd_active:
        with torch.no_grad():
            t_logits = ema_model.module(inputs)         # EMA teacher, eval mode, no grad
        kd = F.kl_div(F.log_softmax(outputs.float()/KD_T, 1),
                      F.softmax(t_logits.float()/KD_T, 1), reduction="batchmean") * (KD_T*KD_T)
        loss = (1.0 - KD_ALPHA) * loss + KD_ALPHA * kd
    ```
    (`ema_started` is set once EMA warmup begins; KD_START 0.5 > EMA_WARMUP 0.15 so the teacher is always populated when KD fires. The teacher forward is UNCOMPILED — avoids the eval-recompile trap and needs no warmup; tail-gating bounds its cost.)
  - **EMA gate for smoke** (so smoke measures heavy cost): `ema_active = progress >= EMA_WARMUP_FRAC or (SMOKE_SECONDS>0 and step>10)`.
  - **SMOKE path**: effective `budget = SMOKE_SECONDS if SMOKE_SECONDS>0 else TIME_BUDGET_S` in while/break; collect `dt` for step>20; guard the per-epoch eval block with `if SMOKE_SECONDS==0`; after loop, if SMOKE print `smoke_img_s`, `projected_epochs`, `final_train_loss`, then `return`.
  - **DataLoader determinism**: pass `generator=torch.Generator().manual_seed(1234)` so c0/cA/cB share an identical data/aug stream (the KD effect is then isolated from data-order noise). Applied unconditionally (both arms).
  - **Recompile monitor**: per-epoch first-step dt print when `USE_COMPILE`.
  - **Summary prints**: add `use_compile`, `kd_alpha`, `kd_t`, `kd_start`, `label_smoothing`, `warmup_seconds`.
  - **Invariant**: defaults (`USE_COMPILE=0, KD_ALPHA=0, LS=0.2, SMOKE_SECONDS=0`) = the EXP-008 baseline functionally (KD branch inert, no compile, eval every epoch). Verified by smoke §A check 6.

## Configuration Changes
- `USE_COMPILE`: 0 → **1** (all cells; banked +12% off-budget, funds the teacher forward).
- `KD_ALPHA`: 0.0 (c0 control) → **0.5** (cA, cB). Pre-registered; not swept against test acc.
- `KD_T`: **4.0** (standard KD temperature, Hinton 2015).
- `KD_START`: **0.5** (tail-only — teacher strong + cost-bounded; reviewer-endorsed). Contingency 0.6 if under-anneal.
- `LS` (label smoothing): 0.2 (c0, cA) → **0.1** (cB reduced-LS arm — tests KD-vs-LS over-softening per Müller 2019).
- Everything else byte-identical to baseline (EMA 0.998, TTA 0.8, PCT_START 0.15, peak LR 0.4, wd 5e-4, batch 512, Cutout12+RandomErasing).

## Execution Environment
- **Method**: local, `CUDA_VISIBLE_DEVICES=1 <env> uv run train.py > /tmp/exp023_<tag>.log 2>&1` per cell (separate process). 
- **Resources**: single H20 GPU 1 (UUID a5c2d56c). VRAM non-constraint (KD adds one activation set; well under 98GB).
- **Estimated runtime**: smoke ~1.5min; official triple ~3×(300s + ~150s eval/startup/compile) ≈ 22min; confirmation pair ~15min. Total ~40min if a winner, ~24min if ties.
- **Log output**: `/tmp/exp023_*.log`; grep metrics, do not flood context.
- **GPU contention**: fresh `/tmp/exp023_orchestrate.sh` (retry-until-clean; foreign >5GB on UUID a5c2d56c or steady img/s<15000 → abort+retry pair). Idle holder PID 1723342 (~3.8GB, 0% util) tolerable.
- **Tool skill**: none (local).

## Abort Criteria
- Loss NaN/Inf or train loss rising after KD engages (progress≥KD_START) → KD destabilized training; stop, record (KD bug or α too high).
- cA/cB `num_steps < 12610` → under-anneal; raise KD_START to 0.6 or compile the teacher, re-run; do NOT report a sub-gate result as a KD verdict.
- Foreign >5GB compute on GPU-1 OR steady img/s < 15000 → abort+retry pair (orchestrator).
- Wall `total_seconds` > 600s → kill, treat as failure.
- No log output after 120s → likely compile hang; kill.

## Verification Protocol

### Verification Procedure

**§A — Correctness smoke** (`/tmp/exp023_smoke.py`, timeout 300s, GPU 1). All must pass:
1. KD loss is non-negative and → 0 when student logits == teacher logits (`F.kl_div` of identical distributions ≈ 0).
2. KD DIRECTION: with a deliberately-confident teacher and uniform student, the KD gradient pushes the student toward the teacher's argmax (loss decreases when student moves toward teacher) — confirms teacher-as-target (forward KL), not reversed.
3. KD term has a finite gradient w.r.t. student params; teacher path has NO grad (`t_logits.requires_grad is False`).
4. Compile aliasing + BN snapshot/restore + params-unchanged-after-warmup (as EXP-022 §A 5–6).
5. EMA teacher forward in eval mode uses running stats (deterministic across two calls on the same input).
6. **Baseline invariant**: defaults (KD off, compile off) → a 2-step train loop matches the unmodified loss path (KD branch not taken; loss == CE_LS only); forward shapes intact.
7. `≤1 eval/epoch` preserved: KD adds a forward in TRAINING only; the per-epoch `evaluator.evaluate` call count is unchanged (code inspection — eval outside batch loop, KD inside).

**§B — Necessary conditions** (goal file), after official + confirmation runs:
1. **Within budget**: `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_steps:" /tmp/exp023_*.log`; `total_seconds < 600`; empty ⇒ crash. Timeout 700s/run.
2. **Improves ≥ +0.1pp**: best KD cell `best_test_acc ≥ 96.48` (baseline 96.38 via `exp-index.sh baseline`) AND > same-session c0 by > ~0.15pp AND replicated on the confirmation pair AND the BAKED-IN bare `uv run train.py` reproduces ≥ 96.48. Any fail → no-improvement.
3. **Anneal gate**: KD cell `num_steps ≥ 12610` (= 130·97). A win below the gate is under-anneal; a tie at ≥ gate is a genuine ceiling datapoint.
4. **Genuine method change**: `git diff --quiet -- prepare.py`; only train.py modified; seed 42 fixed; KD α/T/start pre-registered (not test-tuned); eval once/epoch on EMA only (KD forward is in training, uses the already-maintained EMA — not an extra eval).

### Informational Metrics (Optional)
- `peak_vram_mb`, `num_epochs`/`num_steps`/`training_seconds` (anneal band), `warmup_seconds` (off-budget), `num_params` (unchanged 7,784,627), `smoke_img_s`/`projected_epochs` (sizing). Collected only if all NC pass.
