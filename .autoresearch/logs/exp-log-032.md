# EXP-032: Multi-scale feature-aggregation classifier head

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md
- **Plan**: plans/plan-032.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-032
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Milestone 1 per plan-032: two edits to `train.py`'s `ResNet`. (1) `__init__`: `self.fc = nn.Linear(w3, num_classes)` → `nn.Linear(w2 + w3, num_classes)` (Linear(384,10)). (2) `forward`: keep both layer2 output (`out2`, 128ch@16×16) and layer3 output (`out3`, 256ch@8×8), global-avg-pool each, `torch.cat` → (B,384), then fc. Smoke test passed: params 4,301,146 (= baseline 4,299,866 + 1280), fc.weight (10,384), forward (8,10), backward clean, diff = train.py only, AST clean.

### Surprises & Discoveries
None — the change is a clean localized head modification; param count and shapes matched the plan's predictions exactly.

### Decisions
- Pooled features via `F.adaptive_avg_pool2d(.,1).flatten(1)` then `torch.cat([p2,p3],dim=1)` — keeps the existing global-avg-pool aggregation per scale (only the head's INPUT changes, not the pooling type), isolating the multi-scale variable from EXP-032's other candidates (GeM/avg+max, deferred).
- Kept `compiled_model = torch.compile(model, mode="reduce-overhead")` unchanged; the new forward graph recompiles once on step 1 (negligible, charged to budget).

## Experimental Adjustments

- **Early observation (NOT an abort)**: EXP-032 converges markedly SLOWER than baseline/EXP-031 at the same training fraction — eval ep1=19.3%, ep7=47.7% vs EXP-031's ep1=55.4%, ep7≈80%. Loss decreasing normally (2.1→1.45, no NaN), params 4,301,146 confirmed, dt 8-9ms. No abort criterion met (loss IS decreasing); letting it complete per plan to get a clean final metric + throughput reading. Likely heading to a regression — the direct layer2→classifier path appears to disrupt the tuned feature hierarchy / slow early learning. (source: run.log eval lines ep1-7, step lines ~step 2900/ep8/13%)

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID — background task)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Full 300s-compute-budget training of the k=4 WideResNet with a multi-scale classifier head (concat global-avg-pooled layer2 + layer3 → fc), single H20. Hypothesis: multi-scale features lift best_test_acc above the 96.32 bar at an unchanged ~91 epochs / dt~8ms / 4.30M params. KEY CHECKS: throughput-neutral (epochs ~91) confirms a fair test; best_test_acc vs the 96.32 bar.

Observations:
- Convergence was markedly slower than baseline/EXP-031 throughout (see Experimental Adjustments). The gap narrowed late but never closed: ep53≈90.7%, ep66≈93.0%, ep79≈94.5%, ep84=94.72% (best). The final cosine-anneal tail produced the usual lift but from a much lower trajectory — the run plateaued ~1.5pp below baseline.
- Run completed cleanly, exit 0, no NaN, no traceback. dt held at ~8ms (mean 8.12ms over sampled steps); one transient 14ms blip (step 31700) — noise. (source: run.log summary block + step lines)
- num_epochs=87 (vs baseline ~91) — a mild ~4-epoch throughput cost from the extra layer2 global-avg-pool + wider fc. This is a small compute confound, but it cannot explain a 1.5pp regression: budget-adders in this project cost ≤1 epoch worth of accuracy near the plateau (project-insights), so the dominant cause is the head change disrupting the tuned feature hierarchy, not the ~4 lost epochs.

Key Metrics:
- best_test_acc: **94.72%** (baseline 96.22, bar 96.32 → **−1.50pp vs baseline, −1.60pp vs bar**)
- final_test_loss: 0.2309 (baseline 0.195 → worse by 0.036)
- num_epochs: 87 (baseline ~91); num_steps: 33,677; mean dt: 8.12ms
- num_params: 4,301,146 (intended +1280 from the wider fc); peak_vram_mb: 493.7
- total_seconds: 403.3 (<600 ✓); training_seconds: 300.0; startup_seconds: 2.9
- (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Cond 1 — primary metric clears bar (best_test_acc ≥ 96.32)**: **FAIL.** best_test_acc = 94.72% < 96.32. Decisive regression of −1.60pp vs the bar and −1.50pp vs the 96.22 baseline. (source: `grep "^best_test_acc:" run.log` → 94.72%)
- **Cond 2 — clean completion within budget**: **PASS.** Summary block printed, `grep -c Traceback run.log` == 0, total_seconds 403.3 < 600. (Not gating once Cond 1 failed, but recorded for completeness.)
- **Cond 3 — no constraint violations**: **PASS.** `git diff --name-only` lists only `train.py`; num_params == 4,301,146 (the intended multi-scale-head change); eval-count == num_epochs (87 == 87); core torch only (no new deps); seed 42 unchanged. No constraint violated → verdict is no-improvement, NOT invalid.

**Throughput-neutrality attribution**: epochs 87 (vs baseline ~91), dt 8.12ms. The head change is slightly costly (~4 fewer epochs) — a mild compute confound — but the −1.5pp regression vastly exceeds what ~4 epochs near the plateau could account for. The regression is attributable to the multi-scale head's INDUCTIVE-BIAS effect (direct layer2→classifier path disrupting the tuned feature hierarchy / slowing convergence), as the early-epoch trajectory directly evidences, not to throughput.

### Informational Metrics

- peak_vram_mb: 493.7 (≈ baseline, as predicted — the extra pooled activation is tiny)
- num_epochs / num_steps: 87 / 33,677 (mild throughput cost vs ~91)
- final_test_loss: 0.2309 (worse than baseline 0.195 — the head hurts on loss too, consistent with a genuine convergence/inductive-bias regression rather than a top-1-only noise effect)

## Errors & Dead Ends

<!-- none yet -->

## Human Notes

> (none — autopilot)
