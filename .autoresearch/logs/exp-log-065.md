# EXP-065: Warmup-phase augmentation lightening (head-side pressure-profile quadrant)

## Execution

Overall Status & Info:
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-065.md
- **Plan**: plans/plan-065.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-065
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary

Implemented plan-065 M1 on train.py (+46/−4 lines): module-level `AUG_GATE = torch.zeros(1).share_memory_()`; `GatedTA` / `GatedErase` wrapper classes holding the gate plus the byte-identical baseline transform instances, swapped into train_tf at the exact pipeline positions of the bare TA / RandomErasing; a one-time flip block at the top of the epoch loop (`AUG_GATE[0] = 1.0` + `AUG_ON ep N (p=...)` print at the first epoch boundary with progress ≥ WARMUP_FRAC). Everything else byte-identical — constants, lr_at, model, optimizer, compile warmup, timed step, eval cadence, summary. CPU sanity 10/10, including the load-bearing check: persistent workers (2 workers, prefetch 2) saw a mid-iteration gate flip after 5 batches — the fork-shared-memory propagation works exactly as the EXP-041 pattern promises.

### Surprises & Discoveries

- Worker prefetch drain measured at 5 batches in the sanity probe (2 workers × prefetch 2 + in-flight) — at run scale (8 workers, 97 batches/epoch) the flip lands within ~1/6 of an epoch of the boundary; immaterial against the pre-accepted 1-epoch boundary lag.
- No other surprises: the wrapper classes pickle cleanly under fork, and the real train_tf iterates under persistent workers without issue.

### Decisions

- Gate flip checked at epoch boundaries only (not per-step): keeps the timed loop byte-identical and aligns the distribution shift with the eval/BN cadence. Pre-registered in the plan.
- Light phase keeps crop+flip (EXP-033's light-aug floor) rather than going fully clean — going clean would conflate this quadrant with EXP-025's clean-data damage mechanism.

## Experimental Adjustments

- (none yet)

## Run Log

### Run 1

Metadata:
- **Job ID**: train pid 2668566 (composite gates cleared at poll 1, apps=0 load=7)
- **Log file(s)**: run.log (truth, deleted post-experiment per protocol); /tmp/exp065_composite_run1.log (gate/watchdog)
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-06-11 11:08
- **Ended**: 2026-06-11 11:17

Description:
- Single gated composite run of the aug-profile experiment: TA+RE off for the warmup phase (~21 light epochs at crop+flip only), on at full strength from the first epoch boundary past p=0.15 through budget end. Zero dt toll expected (aug is worker-side) → family bands binding: D0 ∈ [21.5, 23.5], steps ∈ [13,100, 13,600], ~137–140 epochs. Engagement proof = AUG_ON marker AND below-family early train loss followed by a visible loss rise within 2 epochs of the flip. Expected outcomes per pre-registered branches: pair-mean ≥ 96.81 (banked-alignment gain) | family-band null (profile-neutral) | ≤ 96.41 (EXP-018-class transition damage).

Observations:
- Pristine telemetry: RC=0, no kill markers; watchdog windows 21.7–22.7ms throughout; D0 in band (source: /tmp/exp065_composite_run1.log)
- **Mechanism fully engaged (marker + physical signature)**: exactly one `AUG_ON ep 22 (p=0.152)` — the first epoch boundary past WARMUP_FRAC; debiased train loss ran 0.76–0.79 over ep 15–20 (far below family's ~1.1 at full aug — the light phase trained visibly faster) then jumped to 1.14–1.21 within 2 epochs of the flip (source: run.log step lines ~1450–2250, AUG_ON line)
- Eval transient at the flip: ep 21–24 evals dipped (70.2/77.6/73.1/76.1 vs 80–83 before), recovered by ep 25 — hot-phase scatter, not divergence (source: run.log eval lines ep 20–26)
- Plateau tail flat at 96.25–96.38 over the last 8 evals; best == final == 96.38 (source: composite SUMMARY)

Key Metrics:
- best_test_acc: 96.38% @ ep 139 (final eval) — BELOW the family band floor 96.41 (mean −1.2σ) (source: composite SUMMARY)
- num_steps: 13,474 ∈ [13,100, 13,600]; num_epochs: 139; training_seconds: 300.0; total_seconds: 486.4; startup 12.6s; peak_vram_mb: 1,613.0; num_params: 4,286,026; zero NaN; 139 eval lines (once/epoch)

## Verification Results

### Conditions Checked

- **Integrity (gates Condition 1)**: PASS — RC=0, no kill markers; steps 13,474 in family band (BINDING — clean ledger); 139 epochs / 139 eval lines; params 4,286,026; training_seconds 300.0; zero NaN; VRAM 1,613 < 2,000; exactly one AUG_ON at ep 22 (p=0.152, correct boundary) WITH the pre-registered loss-rise engagement signature (below-family 0.76–0.79 before, 1.14+ within 2 epochs after). Mechanism engagement proven by physical signature, not marker alone (EXP-055 law).
- **Condition 1 (best_test_acc ≥ 96.81)**: **FAIL** — 96.38 (baseline 96.71, bar 96.81; family mean 96.57). Pre-registered branch (iii): < 96.41. First-failure-stop: Conditions 2–3 not evaluated for the verdict (informationally: total 486.4s ≤ 600; cadence structural pass).

### Informational Metrics

- Light-phase speedup realized: train loss 0.76 at ep 15–20 vs family ~1.1 — the banked-progress mechanism's premise was real, but the bank did not survive: final 96.38 sits −1.2σ, marginally below the family band. Single-draw caveat: −1.2σ is suggestive of mild transition damage (EXP-018-class, data side), not conclusive — but ≥-bar outcomes are decisively excluded.
- Eval transient at the flip recovered within 3 epochs; plateau reached normally.

## Errors & Dead Ends

### 2026-06-11 — Head-side aug lightening reads below family band
- Error: (research result, not an error) `best 96.38 = mean −1.2σ at byte-clean signatures`
- Root cause: the 21 light-aug epochs trained faster (loss 0.76 vs ~1.1) but the advantage washed out and inverted once full pressure arrived — consistent with the EXP-018 "turns on during peak heat" signature on the data side; the full-aug distribution needs the whole run, including the warmup.
- Source: run.log (deleted post-protocol; key lines quoted above); /tmp/exp065_composite_run1.log
- Do NOT retry: aug-pressure ramps/gates of any phase placement — the pressure-profile law is now four-quadrant complete (head light −0.19, tail light −0.46, tail clean −0.87, full = optimum).

## Errors & Dead Ends

## Human Notes

> (none — autopilot)
