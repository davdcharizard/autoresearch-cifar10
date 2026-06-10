# Plan EXP-070: Dual (avg + max) global pooling readout
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-070.md

## Closed-axis check
Head/readout changes have a negative track record — multi-scale head (EXP-032, −1.5pp, "disrupts the feature HIERARCHY" by feeding mid-level layer2 features) and cosine/normalized head (EXP-039, −0.33pp, changed logit GEOMETRY). This plan is mechanically distinct: it keeps the SAME tuned layer3 (final-stage) features AND the SAME standard linear-logit geometry — it changes ONLY the spatial-aggregation statistic (adds a max-pooled descriptor alongside the existing avg-pooled one). EXP-032's hierarchy-disruption and EXP-039's geometry-change failure modes do not apply. It does NOT contradict any High-importance insight: it is dt-NEUTRAL (pooling is ~free; fc input 256→512 adds 2,560 params and a negligible matmul), so it respects the critical epoch-saturation constraint (every dt-RAISING capacity add — k5/k6/depth/fat-head/BlurPool/GhostBN — underfit; this one preserves ~91 ep). The input-normalization axis was found CLOSED this loop (frozen Eval uses std=(1,1,1) — train must match; brainstorm-070), so that idea was correctly dropped. cudagraph-safe: `adaptive_max_pool2d` is a static-shape op (no data-dependent branch) → no graph break under reduce-overhead.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py L107: `self.fc = nn.Linear(w3, num_classes)` → `self.fc = nn.Linear(2 * w3, num_classes)` (classifier input doubles to hold the concatenated avg+max descriptor; 512→10 at k=4).
- [ ] train.py `ResNet.forward` L131-133: replace
      `out = F.adaptive_avg_pool2d(out, 1)` / `out = out.view(out.size(0), -1)` / `return self.fc(out)`
      with: `a = F.adaptive_avg_pool2d(out, 1).flatten(1)`; `m = F.adaptive_max_pool2d(out, 1).flatten(1)`; `out = torch.cat([a, m], dim=1)`; `return self.fc(out)`.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` OK; quick shape check — instantiate `ResNet(3,10,width_mult=4)`, forward a `(2,3,32,32)` tensor, assert output shape `(2,10)` and print param count (expect 4,299,866 + 2,560 = 4,302,426); `git diff --name-only` == `train.py` only.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (the extra max-pool + 2× fc input must NOT raise dt or break the cudagraph to 14ms — adaptive_max_pool2d is static-shape, cudagraph-safe), no NaN, eval test_acc climbing normally. Confirm num_params printed as 4,302,426.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract best_test_acc, compare to baseline 96.45 / bar 96.55. Expect ~91 ep, dt 8ms, wall ~590s, num_params 4,302,426.

## Code Changes
- **train.py L107 (`ResNet.__init__`)**: `nn.Linear(w3, num_classes)` → `nn.Linear(2 * w3, num_classes)`. Why: the forward now concatenates the avg-pooled (w3-dim) and max-pooled (w3-dim) global descriptors into a 2·w3-dim vector, so the classifier input must double. Adds 2·w3·10 − w3·10 = w3·10 = 2,560 params (negligible).
- **train.py `ResNet.forward` (L131-133)**: replace the single `adaptive_avg_pool2d` + `view` with avg-pool AND max-pool, each flattened to (B, w3), concatenated along dim=1 to (B, 2·w3), then `self.fc`. Why this tests the hypothesis: GAP records only the MEAN channel activation over the 8×8 final map and discards the PEAK (most-discriminative spatial location), which max-pooling preserves; concatenating both gives the linear classifier a strictly richer, complementary global descriptor (the CBAM rationale) at ~zero FLOP cost. Risks/edge cases: (1) max-pool can be noisier than avg on small 8×8 maps (a single outlier activation dominates a channel) → possible small regression; (2) `adaptive_max_pool2d` must stay inside the compiled forward as a static-shape op (it is — fixed (B,256,8,8)→(B,256,1,1)) so no cudagraph break; (3) `.flatten(1)` is equivalent to the old `.view(size(0), -1)` for the pooled (B,C,1,1) tensor. No effect on the eager eval path (eval calls `model(inputs)` → same forward).

## Configuration Changes
- None (no hyperparameter changes). Architecture-only change to the pooling readout + fc width. num_params 4,299,866 → **4,302,426** (+2,560, expected; param count is not a fixed constraint — prior exps ranged 4.17M–9.7M). All else byte-identical to EXP-054 (AugMix-p0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`; Σdt=300s budget REQUIRES an uncontended GPU — relaunch on contention).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~590s (< 600s; the change is FLOP-negligible — pooling + tiny fc — so throughput is unchanged vs EXP-054's 593s). Monitor the 600s wall (recipe is wall-tight, 3 prior breaches) but this change adds no CPU/GPU cost.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf, or eval test_acc not climbing by ~ep5 (would indicate a broken readout — e.g., wrong concat dim or fc-width mismatch; but Milestone 1 shape-check catches this pre-launch).
- dt elevated to ≥13ms sustained (cudagraph break from the added max-pool — NOT expected; if seen, it indicates adaptive_max_pool2d forced a graph break or GPU contention): kill, diagnose; if a true graph break, record as a cudagraph finding.
- dt drifts ≫ 8ms (contention): kill, relaunch on clean idle GPU.
- Runtime error (shape mismatch in cat/fc): caught by the Milestone 1 shape smoke-test before launch; if it somehow reaches runtime, capture traceback, treat as code error (1 fix-retry per execute skill).
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse the float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,302,426` (the expected +2,560), and `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent, NOT invalid — this change is compute-free.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only; prepare.py/eval untouched; evaluate() called once/epoch; no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt). num_params change is an allowed architecture edit (param count not fixed), NOT a violation.
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / cudagraph-break abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged; the readout change is FLOP-negligible). Compare final_test_loss to EXP-054's 0.1968: if the richer descriptor genuinely helps, watch for BOTH top-1 and loss improving; if it is the recurring polish-vs-top1 or a max-pool-noise regression, top-1 stays flat/down. peak_vram ≈ baseline (tiny fc widening).
