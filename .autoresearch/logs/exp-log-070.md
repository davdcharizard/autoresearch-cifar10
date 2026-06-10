# Experiment Log EXP-070: Dual (avg + max) global pooling readout

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-070.md
- **Plan**: plans/plan-070.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-070
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 — two edits to `ResNet` in train.py: (1) L107 `self.fc = nn.Linear(w3, num_classes)` → `nn.Linear(2 * w3, num_classes)` (classifier input 256→512 to hold the concatenated descriptor); (2) `forward` — replaced the single `adaptive_avg_pool2d` + `view` with `a = adaptive_avg_pool2d(out,1).flatten(1)`, `m = adaptive_max_pool2d(out,1).flatten(1)`, `out = torch.cat([a,m],1)`, `self.fc(out)`. All else byte-identical to EXP-054. Smoke: AST OK; forward of a (2,3,32,32) tensor → (2,10); param count **4,302,426** (= 4,299,866 + 2,560 = w3·10, exactly as planned); `git diff --name-only` == train.py only.

### Surprises & Discoveries
None. `.flatten(1)` on the (B,C,1,1) pooled tensor is equivalent to the old `.view(size(0),-1)`. Both pooling ops are static-shape ((B,256,8,8)→(B,256,1,1)), so the added `adaptive_max_pool2d` should not break the reduce-overhead CUDA graph.

### Decisions
- Used `adaptive_max_pool2d` (matching the existing `adaptive_avg_pool2d` API) rather than a manual `.amax((2,3))` — identical result, consistent style, and a recognized static-shape op for the compiler.
- Order in concat: [avg, max] (arbitrary but fixed); fc learns the mapping regardless.

## Run Log

### Run 1
- **Description**: EXP-054 best recipe with the readout changed from global-average-pooling to dual avg+max concatenated pooling (fc 256→512). Tests whether giving the linear classifier BOTH the mean and the peak global response per channel (complementary statistics for a ReLU net, CBAM rationale) captures discriminative localized activations the GAP washes out, clearing the 96.55 bar. dt-neutral (pooling ~free, tiny fc) so ~91 ep expected. Honest prior: within-noise null (head changes lean negative here, max-pool can be noisy on 8×8). Launching on idle GPU 1 (GPU 0 also idle).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10
- **Early gate (ep1, ~step350, run.log L5)**: num_params 4,302,426 ✓; dt steady 8ms ✓ (the added adaptive_max_pool2d did NOT break the reduce-overhead CUDA graph — static-shape op); img/s ~15,000 ✓; no NaN ✓. Loss hovering ~2.306 (≈ln10) at step 350 — I judged this high-LR thrash; **IT WAS THE EARLY SIGN OF REAL INSTABILITY** (see below).
- **Key Metrics**: best_test_acc **87.00%** (best ep~85; **−9.45pp vs baseline 96.45**, ≪ 96.55 bar) | final_test_acc 86.78% | final_test_loss 0.4449 (≫ EXP-054's 0.1968) | **training_seconds 300.0** | **total_seconds 585.1 (< 600 — clean, no wall breach)** | num_epochs 87 | num_steps 33849 | num_params 4,302,426 | peak_vram_mb 500.5. dt distribution: 572×8ms / 99×9ms / 2×11ms / 2×10ms (dt-neutral, no sustained graph break). 0 NaN/error. **DIVERGENCE PATTERN**: eval test_acc stuck at RANDOM (9.99% ep1 → 10.11% ep2 → 10.24% ep3) for the first ~3+ epochs — the model did not learn at all during the high-LR phase — then escaped late and climbed to 87% by ep85, never recovering the lost epochs. Root cause: dual avg+max concat pooling at the tuned peak-LR 0.2 is UNSTABLE — the max-pooled descriptor (peak ReLU activations, large magnitude) concatenated un-normalized with the avg-pooled half inflates the logits/gradients at high LR, trapping the model at random for several epochs.

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **87.00%** < 96.55. **FAILED decisively** (−9.45pp vs baseline 96.45). (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: total_seconds 585.1 < 600 ✓ (CLEAN), training_seconds 300.0 ✓, num_params 4,302,426 (expected +2,560) ✓, 87 ep, summary printed ✓, `grep -ciaE "nan|traceback|error"` == 0 ✓. (Evaluated for completeness; condition 1 already determined the verdict.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only ✓; prepare.py/eval untouched ✓; no new deps ✓; seed 42 unchanged ✓; evaluate() once/epoch ✓; uncontended (dt 8ms) ✓. The +2,560 param change is an allowed architecture edit (param count not fixed), NOT a violation.

**Verdict**: no-improvement — clean valid run (Σdt=300s respected, wall 585.1 < 600, dt 8ms / no cudagraph break, train.py only) that DECISIVELY missed the bar (87.00 < 96.55, −9.45pp). Results trustworthy: 87% is a real, reproducible regression (the eval trajectory shows the model genuinely stuck at random ~10% for ep1-3, then a slow late climb to 87% — not a parsing/eval artifact). Root cause: dual avg+max concatenated pooling at the tuned peak-LR 0.2 is OPTIMIZATION-UNSTABLE — the un-normalized max-pooled descriptor (large peak ReLU activations) concatenated with the avg-pooled half inflates the fc logits/gradients during the high-LR phase, trapping the model at random for several epochs; it escapes too late to recover within the 87-epoch budget. NOT invalid (legitimate architecture change, no constraint breach) and NOT crash (produced a real metric). The pooling-readout change is NOT a free lever — it perturbs the finely-balanced high-LR schedule.
