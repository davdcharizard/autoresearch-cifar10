# Experiment Log EXP-033: Augmentation taper — original-ResNet light transform (crop+flip) for the final 12% of budget

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-033.md
- **Plan**: plans/plan-033.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-033
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
All four plan-033 edits applied on `autoresearch/exp-033` (cut from `autoresearch/dev` @ 1990397): (1) `AUG_TAPER_FRAC = 0.88`; (2) `light_tf` Compose = RandomCrop(32,4) + RandomHorizontalFlip + ToTensor + Normalize (the He-2015 original CIFAR recipe — baseline transform minus TrivialAugmentWide and RandomErasing); (3) `tail_set`/`tail_loader` with loader args identical to `train_loader` (batch 512, shuffle, 8 workers, pin_memory, drop_last, persistent_workers); (4) per-epoch loader selection: `epoch_loader = train_loader if total_training_time/TIME_BUDGET_S < AUG_TAPER_FRAC else tail_loader`. Timed step, LR schedule (full cosine anneal to 0), model, optimizer, eval (base_model once per epoch) all byte-identical to baseline. Sanity: AST OK; diff = 30 insertions / 1 deletion at exactly the 4 sites.

### Surprises & Discoveries
- None at implementation time. The second `datasets.CIFAR10(...)` re-reads cached files (data/ present — no download).

### Decisions
- Tail loader created eagerly at startup but its 8 persistent workers spin up lazily on first iteration (~2s, uncharged, at the taper boundary ~ep 122) — avoids paying double worker memory for the first 88% of the run.
- Kept 8 workers for both loaders (16 total once tail starts; launch is load-gated per EXP-032 infra entry).

## Run Log

### Run 1
- **Description**: Full run on GPU 0. Training byte-identical to baseline until 88% of the 300s charged budget (~ep 122); after that, epochs draw from the light-aug loader (crop+flip only — the original ResNet paper's CIFAR augmentation) while the cosine anneal completes as normal. Expected: dt 22.4ms / ~139 epochs / wall ~495s; an eval jump ≥ +0.15 within 2 taper epochs (EXP-025's fully-clean analogue was +0.35), then a SUSTAINED plateau (crop+flip pressure prevents EXP-025's overfit flatline). Hypothesis passes if best ≥ 96.81 from a sustained post-taper shift. Falsified by: no jump within 3 taper epochs, or test_loss reversing upward before run end.
- **Job ID**: local background composite, Claude task btkkacudn
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed (rc=0)
- **Started**: 2026-06-10 (gates passed at poll 1, load 6)
- **Ended**: 2026-06-10 (481s wall incl. composite overhead)
- **Observations**: Fully clean: 268 windows mean 22.3ms, 0 slow, load 6–13 throughout. Taper engaged ~ep 124 (train loss dropped 0.7695→0.5196 between ticks 28–29 as the light data arrived). Eval response: ep124 95.62 → ep125 96.10 (+0.48 jump, the alignment transient as hypothesized) — then FROZE: evals 96.1–96.25 and test_loss FLAT at ~0.197 for the final 15 epochs. No overfit reversal (loss never rose), but the anneal's final climb (baseline gains ~+0.4 over the last 12% with test_loss falling to ~0.185) did not materialize on light data. Post-taper plateau sits ~0.3 BELOW the baseline plateau.
- **Key Metrics**: best_test_acc 96.25 | final 96.22 | final_test_loss 0.1972 | training_seconds 300.0 | total_seconds 472.1 | startup 10.0s | VRAM 1613.0MB | epochs 139 | steps 13,472 | params 4,286,026 | taper jump +0.48 (ep124→125) | post-taper test_loss slope ≈ 0 (no overfit, no progress)

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
1. **best_test_acc ≥ 96.81 (bar = baseline 96.71 + 0.1)** — **FAIL**. `grep "^best_test_acc:" run.log` → 96.25%. −0.56 vs bar; −0.46 vs recorded baseline; −0.32 vs baseline mean ≈ 2σ — a REAL measured loss, outside the noise band.
   - Pre-condition (profile): **PASS** — 268 windows, mean 22.3ms, slow>27: 0; num_epochs 139 (exact); training_seconds 300.0; params 4,286,026; eval_lines 139 = epochs. Clean, trustworthy run.
2. **Completes within budget** — not evaluated (first-failure-stop). Informationally: rc=0, total_seconds 472.1 ≤ 600 ✓.
3. **Validation ≤ once/epoch** — not evaluated (first-failure-stop). Informationally: 139 = 139 ✓.

**Informational**: taper jump +0.48 (ep124 95.62 → ep125 96.10) — the light-aug alignment transient is real and ~40% larger than half of EXP-025's fully-clean +0.35... (comparable magnitude); overfit check: test_loss FLAT (~0.1972) over the final 15 epochs — crop+flip pressure DID prevent EXP-025's overfit reversal, but learning stalled: the heavy-aug anneal's final climb (~+0.4 at falling loss in baseline) requires the heavy distribution; post-taper plateau 96.2 vs baseline 96.6. The pressure-schedule axis is now bracketed at three points: full 96.6 / light 96.2 / zero collapse (EXP-025).

## Human Notes
(autopilot — none)
