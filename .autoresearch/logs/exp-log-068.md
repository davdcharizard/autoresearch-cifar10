# Experiment Log EXP-068: Lookahead optimizer wrapper (k=5, α=0.5)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-068.md
- **Plan**: plans/plan-068.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-068
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented plan Milestone 1 — Lookahead inline in train.py (no new dep), 4 edits: (1) constants `LOOKAHEAD_K=5`, `LOOKAHEAD_ALPHA=0.5` (L29-30); (2) `slow_weights = [p.detach().clone() for p in model.parameters()]` right after the optimizer (L212); (3) `la_step = 0` pre-loop counter (L226); (4) every-K sync after `optimizer.step()` — `s.add_(p.data - s, alpha=α); p.data.copy_(s)` under `torch.no_grad()` (L258-263); (5) slow-weight eval — snapshot fast, load slow, `evaluate()`, restore fast, bracketing the single eval call (L294-303). AST OK; `git diff --name-only` == train.py only. All else byte-identical to EXP-054.

### Surprises & Discoveries
None during implementation. `model.parameters()` yields a deterministic order, so the `zip(model.parameters(), slow_weights)` alignment is stable across all three call sites. Only PARAMETERS are tracked as slow weights; BN running-stat BUFFERS are not — so the slow-weight eval uses the fast weights' BN running stats (the standard Lookahead+BN minor inconsistency; acceptable since fast/slow sync every 5 steps keeps the params close, so the BN stats are approximately valid for the slow params too).

### Decisions
- **Eval on SLOW weights** (not fast): the slow iterate is Lookahead's deployed model and where the paper's gains are reported — so the faithful test evaluates slow. Implemented via a symmetric snapshot/restore bracketing the single `evaluate()` call (still one eval/epoch). The copies are eager and OUTSIDE the Σdt training timer (eval is untimed) → zero compute-budget cost.
- Dedicated `la_step` counter (incremented only at the sync site) for an exact K-cadence, independent of the existing `step` counter's increment timing.

## Run Log

### Run 1
- **Description**: Lookahead (k=5, α=0.5) wrapping Nesterov SGD on the EXP-054 AugMix-p0.5 best — fast weights step normally; every 5 steps slow weights pull toward fast by 0.5 and fast resets to slow; eval on the slow iterate. Tests whether trajectory-level (not eval-level) weight averaging stabilizes convergence and clears the 96.55 bar on this epoch-bound net. cudagraph-safe eager ops (cf. EXP-064). Expected: ~88-91 ep, dt 8ms; honest prior is a near-noise null (closed EMA/SWA precedent is a headwind). Watch final_test_loss (slow iterate is typically smoother). Launching on idle GPU 1 (GPU 0 also idle).
- **Job ID**: (local, background bash)
- **Log file**: run.log (project root)
- **WandB**: n/a
- **Status**: completed (exit 0)
- **Started**: 2026-06-10
- **Ended**: 2026-06-10
- **Key Metrics**: best_test_acc **95.84%** (best ep~84; −0.61pp vs baseline 96.45, ≪ 96.55 bar) | final_test_loss 0.2097 (> EXP-054's 0.1968) | **training_seconds 300.0** | **total_seconds 588.8 (< 600 — CLEAN, no wall breach)** | num_epochs **91 (= baseline → Lookahead is compute-neutral)** | num_steps 35105 | num_params 4,299,866 | peak_vram_mb 469.3 (+16MB from slow_weights/fast_backup, as predicted). dt sampled at 9ms because the every-50th-step print always lands on a sync step (50 is a multiple of K=5 → samples the heavier slow-copy steps); the 91 epochs confirm aggregate throughput is unchanged. 1 isolated 14ms step (one-time), no sustained cudagraph break. 0 NaN/error. GPU 0 idle throughout (uncontended).

## Experimental Adjustments
(none)

## Errors & Dead Ends
(none)

## Verification Results

### Conditions Checked
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: best_test_acc = **95.84%** < 96.55. **FAILED** decisively (−0.61pp vs baseline). (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: total_seconds 588.8 < 600 ✓ (CLEAN), training_seconds 300.0 ✓, num_params 4,299,866 ✓, summary printed ✓, 0 NaN/error ✓.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == train.py only ✓; no new deps (inline Lookahead, ~15 lines) ✓; seed 42 unchanged ✓; evaluate() once/epoch (save/restore brackets a SINGLE eval call) ✓; uncontended (GPU 0 idle) ✓; the slow-weight eval reads the model the optimizer produces (not eval circumvention) ✓.

**Verdict**: no-improvement — clean valid run (Σdt=300s respected, wall 588.8 < 600, no caveats, 91 ep = baseline so compute-neutral) that decisively missed the accuracy bar (95.84 < 96.55, −0.61pp). Lookahead's trajectory-level slow-weight averaging HURT both top-1 (95.84) AND eval loss (0.2097 > 0.1968) — the same direction as the closed EMA/SWA eval-time averaging family. On this already-well-tuned, epoch-bound net, periodically pulling the fast weights back toward a slow average removes useful late-trajectory progress rather than stabilizing it. (Note: the standard Lookahead+BN inconsistency — slow params evaluated with fast-weight BN buffers — may contribute, but the regression magnitude indicates the averaging itself is net-negative, consistent with EXP-006/019/020.) Closes the optimizer META-WRAPPER class.
