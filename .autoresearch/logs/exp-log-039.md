# EXP-039: Cosine / normalized-softmax classifier head

## Execution

Overall Status & Info:
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-039.md
- **Plan**: plans/plan-039.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-039
- **Commit**: (pending — committed on loop success)
- **PR**: (pending — created on loop success)
- **Outcome**: completed

## Implementation Notes

### Summary
Implemented the single planned change to `train.py` — the classifier head only. (a) In
`ResNet.__init__`, changed `self.fc = nn.Linear(w3, num_classes)` → `nn.Linear(w3, num_classes,
bias=False)` and added `self.logit_scale = 16.0` (a plain float, not a Parameter, so it is excluded
from weight decay and from `model.parameters()`). (b) In `forward`, after global-avg-pool + view,
L2-normalize the penultimate features (`F.normalize(out, dim=1)`) and the classifier weight rows
(`F.normalize(self.fc.weight, dim=1)`), then return `self.logit_scale * F.linear(feat, w)` — scaled
cosine logits. No other recipe element changed. Maps to plan Milestone 1.

### Surprises & Discoveries
- The Milestone-1 smoke test `train.ResNet(3,10)` instantiates with the DEFAULT `width_mult=1`, so it
  printed `params 272464` (a k=1 count) — NOT the k=4 model that actually trains. The forward shape
  check (out (4,10)) and scale (16.0) are valid regardless. Re-running with explicit `width_mult=4`
  confirmed the real training param count = **4,299,856** (baseline 4,299,866 − 10 = exactly the
  dropped fc bias). Train-mode CE+LS loss on a random batch is finite (≈2.37), logit absmax ≈1.75
  (= scale·cosθ in range, as expected for an untrained net).

### Decisions
- `logit_scale` kept as a fixed plain float 16.0 (not learnable, not a buffer) — the standard
  normalized-softmax temperature for ~10 classes; large enough that softmax+LS can saturate, avoiding
  the under-confidence failure mode, while staying compute- and convergence-neutral. The
  scale-saturation diagnostic is `final_test_loss` vs baseline 0.195 (inflated ⇒ scale too small).

## Experimental Adjustments

<!-- Appended incrementally over runs. -->

## Run Log

### Run 1

Metadata:
- **Job ID**: (PID recorded at launch)
- **Log file(s)**: /SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v2.9.6-opus-4-8/run.log
- **WandB**: N/A
- **Status**: completed (exit 0)
- **Started**: 2026-06-09
- **Ended**: 2026-06-09

Description:
- Training the WideResNet-k4 recipe unchanged except for the cosine/normalized-softmax classifier head
  (scale 16). Run on an idle H20 (`CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`). Expect
  throughput-neutral ~88–91 epochs at dt ~8ms (cosine head adds only two L2 norms on tiny tensors).
  Hypothesis: angular decision geometry lifts best_test_acc above the bar 96.32; honest most-likely
  outcome is within-noise (~96.0–96.3). Key diagnostic: final_test_loss not inflated vs 0.195.

Observations:
- Clean, uncontended run: dt distribution = 570 steps @ 8ms, 69 @ 9ms, only warmup outliers
  (44/73/80/84ms = compile/first steps) (source: run.log dt samples). GPU 1 stayed 0%/0MiB the whole run.
- Throughput NOT perfectly neutral: num_epochs 83 vs baseline ~91 (~9% fewer). The two `F.normalize`
  ops + extra `F.linear` cost ~sub-ms/step (more 9ms steps than baseline) → ~8 fewer epochs under the
  fixed 300s budget → mild under-training (source: run.log L summary, dt distribution).
- final_test_loss 0.2099 vs baseline 0.195 — mildly inflated, consistent with both the under-training
  (83 ep) and a slight scale-16 under-confidence (the planned scale-saturation diagnostic).

Key Metrics:
- best_test_acc: 95.89% (source: run.log `best_test_acc:`)
- final_test_loss: 0.2099 (source: run.log `final_test_loss:`)
- num_epochs: 83 | num_steps: 32336 | total_seconds: 405.4 | num_params: 4,299,856 | peak_vram_mb: 491.4
  (source: run.log summary block)

## Verification Results

### Conditions Checked

- **Primary metric clears bar (NECESSARY)**: best_test_acc 95.89% < bar 96.32 (and < baseline 96.22 by
  0.33pp) → **FAIL** → no-improvement.
- **Clean completion within budget (NECESSARY)**: summary block present, total_seconds 405.4 < 600,
  exit 0 → PASS.
- **No hard-constraint violations (NECESSARY)**: `git diff --name-only` = train.py only; seed 42
  unchanged; num_params 4,299,856 (−10 fc bias, expected); eval ≤1/epoch (one eval per epoch in the
  frozen eval call) → PASS. Fairness gate: dt ~8ms, uncontended — fair run (though 83 ep < 91 reflects
  the head's small per-step cost, not contention).

Verdict: **no-improvement** — clean, fair run; cosine head regressed top-1 by 0.33pp vs baseline.

### Informational Metrics

## Errors & Dead Ends

<!-- Append only. -->

## Human Notes

> (none — autopilot)
