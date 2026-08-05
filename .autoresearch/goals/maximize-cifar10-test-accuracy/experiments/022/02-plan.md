# Plan EXP-022: Budget-sized pre-activation Wide ResNet backbone (size-gated)

- **Created**: 2026-06-30

## Goal & Bar (refs, not duplicated)
Baseline **96.38** (commit 07c3760); improvement bar **best_test_acc ≥ 96.48** AND beat the same-session DavidNet-compiled control by > the ~0.1–0.2pp noise floor, replicated on a confirmation pair. Only `train.py` editable; prepare.py frozen; no new deps; no seed hacking; ≤1 eval/epoch; GPU 1 (`CUDA_VISIBLE_DEVICES=1`); 300s training-compute budget; 600s wall cap. Chosen idea & hypothesis: `01-brainstorm.md`.

## Strategy summary
Replace the DavidNet backbone with a **pre-activation Wide ResNet** (the documented higher-ceiling CIFAR-10 conv net, ~97.1% w/ cutout), keeping every backbone-agnostic recipe win (EMA 0.998, tail flip-TTA, time-based one-cycle, LS 0.2, Cutout12+RandomErasing, bf16/channels_last, batch 512) and re-adding the twice-validated **off-budget torch.compile warmup** (+12%) to fund WRN's per-step cost. The experiment is **won or lost on size selection** (reviewer's #1 point): a throughput pre-smoke picks the largest WRN that anneals at **num_epochs ≥ 130**; sub-110-ep cells are under-anneal artifacts, not backbone verdicts. Compared same-session vs a DavidNet-compiled control, confirmed on a 2nd pair if a winner.

## Milestones

### Milestone 1: Code — WRN backbone + compile recipe + env knobs implemented, correctness-smoked
- [ ] Add `import os`; add env knobs (`MODEL`, `WRN_DEPTH`, `WRN_WIDTH`, `WRN_PEAK_LR`, `USE_COMPILE`, `COMPILE_MODE`, `SMOKE_SECONDS`).
- [ ] Add `PreActBlock` + `WideResNet` modules (pre-activation basic block, 3 stages, GAP head, `.tta` flip interface matching ResNet9).
- [ ] Branch model construction in `main()`: WRN (no whitening, no scale_out) vs DavidNet (unchanged: whitening + scale_out). Effective `peak_lr` = `WRN_PEAK_LR` when MODEL=wrn else `PEAK_LR`.
- [ ] Add off-budget `torch.compile` warmup block (separate `train_fwd` handle, local-RNG dummies, BN snapshot/restore, no optimizer.step) BEFORE `t_start_training`; use `train_fwd(inputs)` in the loop.
- [ ] Add `SMOKE_SECONDS` early-exit path (skip eval; print `smoke_img_s` + `projected_epochs`).
- [ ] **Verify**: `CUDA_VISIBLE_DEVICES=1 python /tmp/exp022_smoke.py` — all correctness checks pass (see Verification Procedure §A).

### Milestone 2: Throughput pre-smoke — pick WRN size AND stable LR (training-signal only)
- [ ] Run `SMOKE_SECONDS=25 USE_COMPILE=1` for WRN-16-4, WRN-22-4, WRN-16-8 on GPU 1 (clear window). Smoke includes EMA-update cost (forced on after step 10) so `projected_epochs` reflects the real per-step cost (review #5).
- [ ] Record `smoke_img_s` + `projected_epochs` (= `TIME_BUDGET_S / (97·mean_dt)`); **pick the largest WRN with projected_epochs ≥ ~135** (margin over the 130 gate, since EMA+eval add a little) as cA's size. Record the fallback (next-smaller) size.
- [ ] **LR by TRAIN-LOSS STABILITY only (never test acc — review #8)**: at the chosen size, smoke `WRN_PEAK_LR ∈ {0.4, 0.2, 0.1}` (pre-registered). Read the final debiased TRAIN loss + check no NaN/rising. Adopt the **largest LR whose training is stable** (loss monotone-ish decreasing, finite). Default expectation: removing scale_out=0.125 raises the effective gradient scale ~8×, so 0.4 may be too hot → 0.2/0.1 likely. This uses only the training-loss signal printed during the smoke; the frozen evaluator is NOT called.
- [ ] **Verify**: chosen size projected_epochs ≥ 135 (compiled, EMA-inclusive); chosen LR trains stably; log the size×LR table in 03-execute.md.

### Milestone 3: Official same-session pair — c0 (DavidNet+compile) vs cA (WRN-chosen+compile)
- [ ] Write a FRESH `/tmp/exp022_orchestrate.sh` (do NOT reuse exp021's literally — it sets DEPTH env + exp021 logs, review #11) with the correct env matrix: c0 = `MODEL=davidnet USE_COMPILE=1`; cA = `MODEL=wrn WRN_DEPTH=<d> WRN_WIDTH=<w> WRN_PEAK_LR=<lr> USE_COMPILE=1`; logs `/tmp/exp022_c0.log` / `_cA.log`.
- [ ] Retry-until-clean: pre-check no foreign compute-app >5GB on UUID a5c2d56c AND util<30 (2 checks); launch c0 then cA; monitor every ~20s for foreign >5GB OR steady img/s<20000; abort+retry the whole pair.
- [ ] **Verify**: both finish < 600s wall with valid `best_test_acc`; cA `num_steps ≥ 12610` (= 130·97, the hard anneal gate — review #6); else under-anneal → re-run cA at fallback size or record inconclusive. `git diff --quiet -- prepare.py`.

### Milestone 4: Confirmation pair + bake-in (only if cA is a candidate winner)
- [ ] If cA ≥ 96.48 AND cA − c0 > ~0.15pp AND cA num_steps ≥ 12610: run a 2nd same-session pair in REVERSED order (cAb then c0b) on a clean window.
- [ ] **Verify replication**: cAb − c0b > noise AND cAb ≥ 96.48. If it collapses to < ~0.1pp → low-control-draw artifact → no-improvement.
- [ ] **Bake-in for the frozen harness (review #1)**: if confirmed, set the chosen WRN config as the DEFAULT env values in train.py (`MODEL` default→`wrn`, `USE_COMPILE` default→`1`, `WRN_DEPTH/WRN_WIDTH/WRN_PEAK_LR` defaults→chosen) and re-verify with a BARE `CUDA_VISIBLE_DEVICES=1 uv run train.py` reproducing best_test_acc ≥ 96.48. The goal's verification runs bare `uv run train.py`, so the winner must be the default — env-only wins do not count.

## Code Changes
- **train.py** (sole editable file):
  - `import os` after `import gc`.
  - **Env knobs** after the hyperparameter block: `MODEL = os.environ.get("MODEL","davidnet")`; `WRN_DEPTH = int(os.environ.get("WRN_DEPTH","16"))`; `WRN_WIDTH = int(os.environ.get("WRN_WIDTH","4"))`; `WRN_PEAK_LR = float(os.environ.get("WRN_PEAK_LR", str(PEAK_LR)))`; `USE_COMPILE = os.environ.get("USE_COMPILE","0")=="1"`; `COMPILE_MODE = os.environ.get("COMPILE_MODE","default")`; `SMOKE_SECONDS = float(os.environ.get("SMOKE_SECONDS","0"))`.
  - **`PreActBlock(c_in, c_out, stride)`** (He 2016 pre-activation, Zagoruyko WRN form): `out = relu(bn1(x)); shortcut = conv_sc(out) if (stride!=1 or c_in!=c_out) else x; out = conv1(out); out = conv2(relu(bn2(out))); return out + shortcut`. Convs 3×3 bias-False; conv1 carries the stride; 1×1 conv_sc on the pre-activated `out` (standard WRN projection). No dropout (already heavily regularized).
  - **`WideResNet(depth, width, num_classes)`**: `n=(depth-4)//6`; widths `[16, 16w, 32w, 64w]`; `conv1` 3→16 (3×3, pad1, no BN — first block's bn1 pre-activates); `stage1` (stride1), `stage2`/`stage3` (stride2 first block); final `bn` + `relu` + `adaptive_avg_pool2d(1)` + `Linear(64w→10)`. `_forward_once` returns raw logits (NO scale_out). `forward` mirrors ResNet9's TTA pattern (`self.tta` flag, flip-average in eval). `apply(_weights_init)`: kaiming_normal fan-out on Conv2d, default Linear init.
  - **`main()` model branch**: if `MODEL=="wrn"`: `model = WideResNet(WRN_DEPTH, WRN_WIDTH, NUM_CLASSES).to(device, channels_last)`; SKIP whitening compute/load (no `self.whiten`); `peak_lr = WRN_PEAK_LR`. Else (davidnet): unchanged — `model = ResNet9(...)`, compute+load whitening, `peak_lr = PEAK_LR`. Optimizer built on `[p for p in model.parameters() if p.requires_grad]` (works for both). LR schedule lines use `peak_lr` (not the `PEAK_LR` constant).
  - **Compile + off-budget warmup** (before `t_start_training`, after EMA setup): init `warmup_seconds = 0.0` UNCONDITIONALLY first (review #12 — avoids unbound var when compile off). `train_fwd = model`; `if USE_COMPILE:` set `train_fwd = torch.compile(model, mode=COMPILE_MODE)`, snapshot BN running buffers (`{n: b.clone() for n,b in model.named_buffers() if "running_" in n or "num_batches" in n}`), `model.train()`, run **3** fwd+bwd on dummies from a LOCAL `torch.Generator().manual_seed(0)` (`torch.randn(BATCH_SIZE,3,32,32,generator=g)` → `.to(device, channels_last)`, `torch.randint(0,NUM_CLASSES,(BATCH_SIZE,),generator=g)`), **each fwd+loss INSIDE `with torch.autocast("cuda", dtype=torch.bfloat16):`** (review #3 — must compile the bf16 graph, not fp32, or the real bf16 compile lands in-budget), `optimizer.zero_grad(set_to_none=True)` before backward, **no optimizer.step()**, restore BN buffers via `copy_`, final `optimizer.zero_grad(set_to_none=True)`, then **`torch.cuda.synchronize()`** (review #4 — drain async warmup kernels so they are not charged to the first timed step), set `warmup_seconds = time.time()-t_w0`, print it. The loop body uses `outputs = train_fwd(inputs)`. `t_start_training = time.time()` is taken AFTER the sync.
  - **EMA gate** (review #5): replace the in-loop `if progress >= EMA_WARMUP_FRAC:` with `ema_active = (progress >= EMA_WARMUP_FRAC) or (SMOKE_SECONDS > 0 and step > 10)`; update EMA when `ema_active` — so the smoke's measured `dt` includes the per-step EMA copy cost (which scales with WRN size), preventing epoch over-projection. (In a real run `SMOKE_SECONDS==0`, so behavior is unchanged from baseline.)
  - **SMOKE path**: effective budget `= SMOKE_SECONDS if SMOKE_SECONDS>0 else TIME_BUDGET_S` in the `while`/`break` checks; collect per-step `dt` for `step>20`; guard the per-epoch eval block with `if SMOKE_SECONDS == 0` (no evaluator calls during smoke). After the loop, if SMOKE: compute `mean_dt` over collected samples; print `smoke_img_s` (= BATCH/mean_dt), `projected_epochs` (= `TIME_BUDGET_S / (len(train_loader)*mean_dt)`), and the final debiased train loss (for LR-stability read); then `return` before the final-summary block.
  - **Recompile monitor** (review #10): print `[compile] ep {epoch} first_step_dt: {dt*1000:.0f}ms` on the first step of each epoch when `USE_COMPILE` — a silent per-epoch recompile (which would masquerade as under-anneal) shows as a dt spike.
  - **DataLoader determinism across cells** (review #2): pass `generator=torch.Generator().manual_seed(1234)` to the `DataLoader` so the shuffle order + worker transform base-seed are IDENTICAL for c0 and cA, regardless of how many RNG draws each model's init consumes. Without this, DavidNet vs WRN init consume different amounts of global RNG → different data/aug stream → a 0.1–0.2pp delta could be order-noise not backbone. Applies equally to both cells (added unconditionally), so the comparison is clean; c0 is a same-session control (not the 96.38 baseline number), so changing the stream is fine.
  - **Summary prints**: add `model:`, `use_compile:`, `wrn_depth/width:` (when wrn), `peak_lr:`, `warmup_seconds:`.
  - **Invariant**: with defaults `MODEL=davidnet, USE_COMPILE=0, SMOKE_SECONDS=0`, the davidnet path is functionally identical to the committed baseline (whitening, scale_out, schedule on peak_lr==PEAK_LR==0.4, no compile, eval every epoch). Verified by smoke §A check 7.

## Configuration Changes
- `MODEL`: davidnet → **wrn** (cA); davidnet (c0 control).
- `WRN_DEPTH/WRN_WIDTH`: chosen by Milestone-2 smoke from {16-4, 22-4, 16-8} (largest with proj_ep ≥ 130).
- `USE_COMPILE`: 0 → **1** (both cells; banked +12%, off-budget — EXP-014/021 validated).
- `WRN_PEAK_LR`: selected in Milestone-2 smoke by **training-loss stability only** from the pre-registered set {0.4, 0.2, 0.1} — the largest LR that trains stably (finite, decreasing debiased train loss). NEVER selected by test accuracy (review #8: that would be test-set tuning, giving cA an unfair selection c0 doesn't get). Rationale: WRN with a GAP head and no scale_out has a ~8× larger effective gradient scale than DavidNet (which divides logits by 0.125), so DavidNet's 0.4 peak is likely too hot for WRN; the smoke determines this from training dynamics, not the benchmark.
- Everything else byte-identical to baseline (EMA 0.998, TTA 0.8, PCT_START 0.15, LS 0.2, wd 5e-4, batch 512, Cutout12+RandomErasing).

## Confounds & Attribution (review #7)
cA (standard WRN) differs from c0 (DavidNet) as a **bundle**: backbone topology + no whitening + no scale_out + GAP head (vs MaxPool) + possibly a different peak LR. This is **intentional** — the experiment tests whether a *different backbone family in its standard configuration* clears the ceiling, not an isolated one-layer swap. Removing whitening/scale_out is the RIGHT call (they are DavidNet-specific tricks; porting them onto WRN would add MORE confound, and the reviewer endorsed starting from clean standard WRN). Consequences for interpretation: (a) a **win** is a bundle win → triggers a follow-up attribution loop (e.g. add the whitening front-end onto WRN, or restore scale_out) to localize the gain; (b) a **null at ≥12610 steps** means this whole standard-WRN config does not beat DavidNet at 300s — a strong signal that the ~96.4 ceiling is recipe/data-bound, not specific to DavidNet's topology. Either outcome advances the goal's knowledge materially.

## Execution Environment
- **Method**: local, `CUDA_VISIBLE_DEVICES=1 uv run train.py > <log> 2>&1` with env knobs prepended per cell. Each cell is a separate process (no cross-cell compile contamination).
- **Resources**: single H20 GPU 1 (UUID a5c2d56c). VRAM non-constraint (DavidNet ~1.6GB; WRN-16-8 ~ a few GB of 98GB).
- **Estimated runtime**: smokes ~25s training + compile/startup each (~3×1.5min); official pair ~2×(300s train + ~150s eval/startup) ≈ 15min wall; confirmation pair another ~15min. Total ~40–50min if a winner, ~30min if cA ties.
- **Log output**: `/tmp/exp022_*.log` per cell (smoke, c0, cA, c0b, cAb). Read metrics via grep; do NOT flood context.
- **GPU contention**: write a FRESH `/tmp/exp022_orchestrate.sh` adapting the EXP-021 retry-until-clean PATTERN with the EXP-022 env matrix (c0 davidnet+compile, cA wrn+compile) and exp022 log names — do NOT reuse the exp021 script literally (it sets DEPTH env + exp021 logs, review #11). Pre-check no foreign compute-app >5GB on UUID a5c2d56c AND util<30 (2 checks); monitor every ~20s during the pair for foreign >5GB OR steady img/s<20000; abort+retry the whole pair. The persistent idle holder (PID 1723342, ~3.8GB, 0% util) is tolerable.
- **Tool skill**: none (local).

## Abort Criteria
- Loss NaN/Inf or diverging (debiased train loss rising past ep5) → the chosen `WRN_PEAK_LR` failed stability (should have been caught in the smoke); stop, drop to the next-lower pre-registered LR.
- cA `num_steps < 12610` (= 130·97) at the chosen size → under-anneal; abort, re-run at the fallback (smaller) size; do NOT report a sub-gate result as a backbone verdict (review #6).
- Foreign >5GB compute job on GPU-1 (UUID a5c2d56c) OR steady img/s < 20000 → abort+retry the pair (orchestrator); do not trust contended numbers.
- Wall-clock `total_seconds` > 600s for any cell → kill, treat as failure (review #9: WRN evals every epoch; screen wall on the FIRST official cA run and downsize if it projects to breach 600s).
- No log output after 120s → investigate (likely compile hang); kill.

## Verification Protocol

### Verification Procedure

**§A — Correctness smoke** (Milestone 1, `/tmp/exp022_smoke.py`, timeout 300s, GPU 1). All must pass:
1. `WideResNet(16,4)`, `(22,4)`, `(16,8)` build; forward `(4,3,32,32)`→`(4,10)`, all-finite, in train and eval mode.
2. Param counts within ±2% of reference (WRN-16-4 ≈ 2.75M, WRN-22-4 ≈ 4.30M, WRN-16-8 ≈ 11.0M) — sanity that depth/width wiring is correct.
3. TTA path: in eval with `.tta=True`, output == `0.5*(f(x)+f(x.flip(-1)))` (bit-equal to manual).
4. Gradient flows: 1 fwd+bwd → every trainable param has non-None finite grad (no dead block).
5. Compile aliasing: `torch.compile(m).parameters()` tensor-ids == `m.parameters()` ids (EMA/optimizer see the same tensors).
6. BN restore: snapshot running buffers, run 3 warmup fwd+bwd (train mode, no step), restore → buffers bit-equal to snapshot AND params unchanged (no step applied).
7. **DavidNet invariant**: `MODEL=davidnet` default builds ResNet9 with whitening + scale_out; a 2-step train loop runs; forward shape (4,10). (Confirms the davidnet path is intact.)
8. Eval-boundary: toggling `.tta` and train/eval does not trigger recompile of the uncompiled eval/EMA path (eval uses `ema_model.module`, never `train_fwd`).

**§B — Necessary conditions** (goal file), after the official + confirmation runs:
1. **Completes within budget**: `grep "^best_test_acc:\|^training_seconds:\|^total_seconds:\|^num_epochs:" /tmp/exp022_cA.log` returns values; `total_seconds < 600`; empty grep ⇒ crash (read `tail -n 50`). Timeout 700s per run.
2. **Improves over baseline ≥ +0.1pp**: cA `best_test_acc ≥ 96.48` (baseline 96.38 via `exp-index.sh baseline`). AND cA − c0(same-session) > ~0.15pp (above noise). AND replicated: cAb ≥ 96.48 AND cAb − c0b > noise. AND (review #1) the BAKED-IN bare `CUDA_VISIBLE_DEVICES=1 uv run train.py` (winning config as defaults) reproduces ≥ 96.48 — env-only wins do not count. If any fails → no-improvement.
3. **Anneal gate (experiment-specific, review #6)**: cA `num_steps ≥ 12610` (= 130·97, robust to the partial-final-epoch over-count in `num_epochs`). A win below the gate is an under-anneal artifact (re-run smaller / inconclusive); a tie at ≥ 12610 steps is a genuine ceiling datapoint.
4. **Genuine method change**: `git diff --quiet -- prepare.py` (byte-unchanged); only train.py modified; seed fixed at 42 (no re-roll); LR chosen by train-loss stability not test acc (review #8); eval block runs once per epoch (code inspection — eval is outside the batch loop).

### Informational Metrics (Optional)
- `peak_vram_mb`: `grep "^peak_vram_mb:" /tmp/exp022_*.log`.
- `num_epochs` / `num_steps` / `training_seconds`: confirms full budget + anneal band.
- `num_params`: WRN size vs DavidNet 7,784,627 — accuracy/capacity trade-off.
- `warmup_seconds`: off-budget compile cost (sanity it's not counted in training_seconds).
- `smoke_img_s` / `projected_epochs`: the sizing decision record.
