# Log EXP-051: Confidence-weighted CE — detached w = p_true^0.7, mean-normalized (GCE-style aug-noise filtering)
## Execution
- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-051.md
- **Plan**: plans/plan-051.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-051
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
M1 per plan: `GCE_Q = 0.7` constant added; timed-loop loss replaced with per-sample CE+LS (`reduction="none"`) weighted by detached, batch-mean-normalized `p_true^GCE_Q` (computed in a `torch.no_grad()` block: softmax → gather → float pow → mean-normalize); warmup loss mirrored identically with warm_y. Diff: 1 file, +16/−4 (constant + two loss sites). CPU sanity (/tmp/exp051_sanity.py): (a) params 4,286,026; (b) q=0 identity with plain mean CE+LS to 1e-6; (c) raw weights for p=(0.01, 0.5, 0.95) = (0.0398, 0.6156, 0.9647) — ~25× relative suppression of destroyed-label views vs genuine; batch mean(w)=1.000000; (d) per-row logit gradient of the weighted loss equals w_i × the unweighted row gradient EXACTLY (allclose) — GCE gradient geometry confirmed; (e) 6-step smoke at lr 0.01 decreasing 2.24 → 0.85. All passed first run.

### Surprises & Discoveries
- None. The detached-weight scaffold behaved exactly as the plan specified; no LS interaction surprises this time (the weight multiplies the loss VALUE per sample, so orderings are intuitive, unlike EXP-050's logit-side intervention).

### Decisions
- None beyond plan. `p_true.float()` used for the pow as planned (weight in fp32; ce vector from F.cross_entropy under autocast is already fp32).

## Run Log

### Run 1
- **Description**: Single gated 300s-budget run of train.py with confidence-weighted CE (detached w = p_true^0.7, mean-normalized) — the suppression-side test of EXP-050's destroyed-view hypothesis: augmentation-destroyed views (p_true → 0) lose their gradient vote (~25× suppression) while genuine samples keep ≈ full gradient; heat kept family-equal by mean-1 normalization. Signatures projected byte-identical. Pre-registered branches: (i) ≥96.81 improvement + hypothesis confirmed; (ii) mean band 96.42–96.72 → filtering absorbed, hypothesis unsupported, per-sample loss class closed; (iii) <96.42 → suppression negative, class closed both sides; (iv) gate/contention infra.
- **Job ID / PID**: composite background task bqlqhq0iy; train pid 1767871
- **Log file**: run.log (project root) + composite stdout (task output file)
- **WandB**: N/A
- **Status**: completed (RC=0, PROC_EXITED at tick 32)
- **Started**: 2026-06-11 03:36:52 (GATES_CLEAR poll 1: apps=0, load=32; GATE_DECISION D0=22.7ms, projected_epochs=136, contention_thresh=28.4ms — family dt, weighting throughput-free)
- **Ended**: 2026-06-11 ~03:44:42 (total_seconds 469.7)
- **Observations**: Pristine run: 29 post-gate windows all 21.7–22.8ms, slow_streak 0, no kills. Result is a LARGE active negative: best 95.32 = mean − 7.8σ. Plateau signature differs from 050's: still climbing at cutoff (95.09 → 95.32 over the last 8 evals, best AT the final epoch) with elevated test_loss (0.239 vs family ~0.185) — the undertrained shape, i.e., confidence weighting SLOWED effective learning. Mechanistic read: on clean-label CIFAR-10, low-p_true views are predominantly hard-but-GENUINE (the informative gradient population), not destroyed; suppressing them ~25× starves the model of exactly the examples that drive boundary refinement. Source: composite task bqlqhq0iy; run.log.
- **Key Metrics**: best_test_acc 95.32 | final_test_acc 95.32 | final_test_loss 0.2389 | training_seconds 300.0 | total_seconds 469.7 | startup 12.1 | peak_vram_mb 1613.0 | num_epochs 139 | num_steps 13,417 | params 4,286,026

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
- **Integrity pre-condition** — PASS. 29 post-gate ≥200-step windows all 21.7–22.8ms (mean ≈22.3 ≤ 23.5; none > 27). num_epochs 139 ∈ [136,142]. num_steps 13,417 within ~1% of the family ledger (11 steps / 0.08% below 13,428 — well inside the ±1% band). params 4,286,026 exact. training_seconds 300.0. eval count 139 ≤ 139. Numerics: trajectory family-shaped early (weights ≈ uniform at low confidence, as predicted); the depressed still-climbing tail is the intervention's effect, not a numerics defect. Source: composite task bqlqhq0iy; run.log.
- **Condition 1 (best_test_acc ≥ 96.81)** — FAIL. 95.32% — far below the replicate band. First-failure-stop: conditions 2–3 not evaluated for the verdict.
- **Condition 2 (within budget)** — skipped per first-failure-stop (informationally passes: RC=0, 469.7s ≤ 600).
- **Condition 3 (eval cadence)** — skipped per first-failure-stop (informationally passes: 139 ≤ 139).

### Informational Metrics
- final_test_loss 0.2389 (family ~0.185 — ELEVATED, opposite of EXP-050's 0.150: the two per-sample interventions produced opposite CE signatures and BOTH lost accuracy)
- num_epochs 139 / num_steps 13,417 — family (weighting throughput-free)
- peak_vram_mb 1613.0 (family)
- Plateau shape: still-climbing at cutoff, best at final epoch — undertrained signature (slowed effective learning), distinct from 050's converged-depressed shape

## Human Notes
(autopilot — none)
