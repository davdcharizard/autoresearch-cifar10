# Plan EXP-021: Stage-2 training-only companion classifier
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Implement an evaluator-invisible companion objective
- [x] Add a `Linear(128,10)` companion head after all inherited modules and initialization, preserving every inherited tensor bitwise by saving/restoring global CPU RNG around construction and overwriting the head from a dedicated seed-42021 CPU generator. Assert 1,290 new and 2,750,180 total trainable parameters.
- [x] Tap the output after block index 3 only when `return_companion=True`, compute `ReLU -> adaptive average pool -> companion_fc`, and return main logits, companion logits, and the 128-dimensional pooled feature. Preserve the default `model(inputs, drop_scale)` main-logits-only path used by the frozen evaluator, with no companion pooling or head call.
- [x] Factor the inherited hard/CutMix CE formula into one deterministic helper and compute `joint_loss = main_loss + 0.15*companion_loss` on every primary step using the exact same hard or area-corrected targets. On scheduled SAM steps use that identical joint objective for both the unperturbed perturbation gradient and perturbed descent gradient, with the head in optimizer/SAM/snapshot inventories.
- [x] Preserve architecture, inherited initialization, data/CutMix/drop-path RNG streams, LR, optimizer, SAM rho/cadence/RNG replay/BN suppression, BF16/channels-last, charged timer, evaluator, and max-selection semantics.

### Milestone 2: Add bounded mechanism and conditioning audits
- [x] Record only the core exact companion calls. A primary loss call occurs once per completed step; a replay loss call occurs once per applied SAM step; the head's internal training-forward count must equal their sum `step + sam_applied_batches`. Snapshot the head-forward counter before/after each evaluation and require equality, proving the default path never executes it.
- [x] At one-based primary steps `1` and multiples of 512, sample pooled stage-2 L2 norm into four fixed charged-progress bins `[0,.25)`, `[.25,.5)`, `[.5,.75)`, `[.75,1]` using fixed device sum/squared-sum scalars. Expected sample count is exactly `1 + floor((step-1)/512)`; report bin counts, mean, and RMS. Norm magnitude and even report-only reconstruction anomalies never gate or adapt the metric run.
- [x] Save initial companion parameters before the charged timer and report terminal displacement after charged training ends. Reconcile only optimizer/SAM ownership in deterministic smoke, preserve inherited CutMix/SAM counts, and report evaluation count, final-16 context, and final parameter/optimizer nonfiniteness.
- [x] Hard runtime integrity gates are limited to exact primary/replay/head/evaluator call formulas, finite training/state, correct parameter inventory, and the inherited SAM/CutMix restoration invariants. Feature norms and head displacement are interpretation-only; no scientific diagnostic magnitude may alter execution or verdict.

### Milestone 3: Pass exact semantics and a decisive accuracy-blind H20 preflight
- [x] Pass syntax/scope and deterministic CPU/GPU smoke for inherited state/main-logit/RNG parity; isolated repeatable head initialization; exact block-3 shape; default-vs-training forward contracts; hard/CutMix loss references including lambdas 0/1; auxiliary-only gradient reach; and no gradient into blocks 4-5/main classifier from auxiliary loss alone.
- [x] Prove one ordinary and one scheduled SAM update: both passes call the companion with identical hard targets and replayed masks, perturb all parameters to radius 0.05, update BN once, restore all parameters/flags exactly, and apply one Nesterov update from the second joint gradient. Inject a replay failure and require exception-safe restoration.
- [x] Run seven alternating-order parent/candidate rounds of 512 steps per arm using deterministic real-CIFAR fixtures in the exact EXP004 path proportions: 209 early ordinary, 205 early CutMix, 49 late ordinary, and 49 late SAM. Use primary step IDs 2..513 so exactly step 512 samples the cadence audit; smoke separately exercises step 1. Require parent drift `<=3%`, MAD/median `<=2.5%`, median weighted ratio `<=1.03`, p90 ratio `<=1.06`, projected `>=24,000` steps / `>=124` epochs, zero evaluator calls, and post-persistent-state live allocation growth `<=1 MiB`.
- [x] Validate source hashes, fixture/path/call counts, timing formulas, pair ordering, and JSON schema before timing. A rerun is allowed only if the harness exits/throws before a gate vector, GPU UUID/visibility changes, an external compute process appears on physical GPU 0 between start/end checks, or a preregistered source/schema assertion fails before ratios are evaluated. Any complete environment-clean numeric vector is decisive; thermal/clock variation or a narrow threshold miss never authorizes rerun.

### Milestone 4: Run exactly one fixed-seed metric experiment and classify it
- [x] Reconfirm physical GPU 0 and single-device visibility, remove stale `run.log`, and launch `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp021-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1` exactly once after all accuracy-blind gates pass.
- [x] Monitor only liveness and hard integrity. Never stop or adapt from finite loss, feature norms, loss ratios, head displacement, intermediate accuracy, or apparent dose. Preserve the raw log through durable transcription and independent result audit.
- [x] Require exit 0, 299.5-301.0 charged seconds, total below 600 seconds, exactly one evaluation per epoch, 2,750,180 parameters, exact core companion/SAM/CutMix reconciliation, main-only evaluation, and zero nonfinite state. Realized steps below 24,000 are a dose shortfall, not a hard-constraint violation or permission to retry.
- [x] Classify `best_test_acc >=95.50%` as formal improvement over EXP004's 95.40% when the goal's completion/budget conditions pass, regardless of realized step count; lower accuracy or a completed run outside the charged-time range is no-improvement, while untrustworthy scope/evaluator/device results are invalid and no-result crashes are crash. Only `steps>=24,000`, `best>=95.60%`, final-16 mean `>=95.50%`, and final within 0.15 points of best qualifies for later EMA composition.

## Code Changes
- **`train.py` only / isolated companion module**: add fixed constants `COMPANION_BLOCK_INDEX=3`, `COMPANION_CHANNELS=128`, `COMPANION_WEIGHT=0.15`, `COMPANION_INIT_SEED=42021`, and `COMPANION_AUDIT_EVERY=512`. Construct and initialize `companion_fc` after the inherited `self.apply`, saving/restoring global RNG and using an ephemeral dedicated generator, so all original state and future RNG remain identical.
- **`train.py` only / forward and loss contract**: default forward stays a single main-logit tensor. Training requests return main logits, companion logits, and pooled stage-2 features from the same backbone pass. A shared CE helper applies identical hard or CutMix target semantics; both SAM passes optimize the same joint loss.
- **`train.py` only / minimal audits and summary**: add training-only head/loss call counters, cadence-512 four-bin feature norms, terminal displacement/finiteness, and final-16 context. Do not add per-step loss sums or gradient-share reductions. Setup-time head initialization/snapshots occur before `t_start_training`; every training head/pool/loss, Python counter, sparse norm reduction/device add, SAM snapshot, and optimizer operation occurs inside `t0 -> cuda.synchronize -> dt` and is charged. After the loop, scalar host reads, displacement/finiteness scans, tail arithmetic, and summary printing are report-only; evaluation remains outside charged time and never executes the head.

## Configuration Changes
- Companion attachment: absent -> end of residual block index 3 (`128x16x16`), fixed full run.
- Companion head: absent -> `ReLU + global average pool + Linear(128,10)` with 1,290 parameters; no BN, convolution, MLP, dropout, temperature, or inference use.
- Companion coefficient: absent -> constant `0.15`; main coefficient stays 1.0. This is the proposal's conservative subordinate dose and cannot be tuned from timing, loss, gradients, or accuracy.
- Companion initialization: absent -> isolated CPU seed `42021`, leaving inherited state and RNG byte-identical.
- Composition rule: formal 95.50 pass is separate from stable mechanism support at best 95.60 / tail mean 95.50 / final gap at most 0.15.

## Execution Environment
- Method: local execution from repository root. Temporary smoke/preflight artifacts use `/tmp/exp021-*`; sole metric command is `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp021-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- Resources: physical GPU 0 only, exactly one visible NVIDIA H20 with approximately 97,871 MiB. Memory is soft; expected metric peak is close to EXP004's 1,190.5 MiB plus small companion state.
- Estimated runtime: deterministic checks and decisive preflight under 5 minutes; metric run about 450-480 seconds total with exactly 300 charged seconds and 600-second hard outer timeout.
- Log output: `/tmp/exp021_preflight.log` stores the accuracy-blind vector; repository `run.log` stores the sole metric result. Preserve raw evidence through transcription and independent audit, then delete before the next experiment. Use experiment-owned bytecode cache `/tmp/exp021-pycache`.
- Tool skill: none; local single-GPU execution.

## Abort Criteria
- Stop before metric launch for wrong GPU/visibility, tracked scope beyond `train.py`, syntax/diff failure, inherited state/logit/RNG mismatch, head inventory/initialization mismatch, wrong tap/default-forward/target semantics, SAM objective/restore/BN/RNG/optimizer mismatch, evaluator access, nonfinite state, or counter/allocation failure.
- Abort on the first environment-clean preflight if parent drift exceeds 3%, MAD/median exceeds 2.5%, median ratio exceeds 1.03, p90 exceeds 1.06, projected steps fall below 24,000, projected epochs below 124, live allocation grows by more than 1,048,576 bytes, or a structural field fails. Rerun eligibility is limited to the enumerated pre-vector exception, UUID/visibility change, external GPU process, or preregistered source/schema failure; preserve original evidence.
- During the metric run terminate on traceback, CUDA/OOM/device error, explicit structural integrity failure, nonfinite state, no progress for 90 seconds after startup, or the 600-second timeout. Finite scientific diagnostics and all accuracy values are non-adaptive.
- A valid completed run below 95.50% is no-improvement and cannot be retried with another weight, head, tap, phase, or seed in EXP021. A trustworthy accuracy pass remains a formal improvement even below 24,000 steps, but dose shortfall blocks the stronger mechanism/composition claim and cannot authorize retry. A completed charged-time miss is no-improvement; a scope/evaluator/device integrity violation is invalid; a crash/no result is crash.

## Verification Protocol

### Verification Procedure

1. **Parent, scope, syntax, and source checks** (30-second timeout):
   ```bash
   bash /root/david/.codex/plugins/cache/deoxys/tree-autoresearch/0.1.3/skills/shared/scripts/tree.sh show .tree-autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv 004
   git diff --name-only 1a8d0de
   git status --porcelain --untracked-files=all
   git diff --check
   env PYTHONPYCACHEPREFIX=/tmp/exp021-pycache uv run python -m py_compile train.py
   ```
   Require parent 95.40, only tracked `train.py` changed, no repository helper artifacts, and clean syntax/diff. `.tree-autoresearch/` is ignored loop metadata.

2. **GPU identity and visibility** (10-second timeout):
   ```bash
   nvidia-smi --query-gpu=index,name,memory.total,uuid --format=csv,noheader
   env CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; p=torch.cuda.get_device_properties(0); print(torch.cuda.device_count(), p.name, p.total_memory)'
   ```
   Require physical GPU 0 is NVIDIA H20 with about 97,871 MiB and exactly one visible device; record UUID.

3. **Deterministic semantic/integration smoke** (180-second timeout):
   ```bash
   env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp021-pycache uv run python /tmp/exp021_companion_smoke.py
   ```
   Both EXP004 and the candidate already have the same existing `if __name__ == "__main__": main()` import seam; no main-loop restructure is permitted or needed. Import the exact production helpers and replace the import-created evaluator before iteration. Materialize EXP004 source from commit `1a8d0de` in memory and hash it. Require exact inherited state/default-main-logit/post-forward RNG equality; global RNG preservation and repeatability of isolated head initialization; 2,750,180 total/1,290 new parameters; block-3 tap shape; default/training forward contracts; exact hard/CutMix loss; auxiliary-only gradient reach; joint gradients; optimizer/SAM/snapshot ownership; two-pass SAM target/RNG/BN/restore/update parity; failure restoration; step-1 and step-512 audit routing with closed-form count `1+floor((step-1)/512)`; final finiteness; and zero evaluator calls.

4. **Decisive accuracy-blind paired preflight** (300-second timeout):
   ```bash
   timeout 300s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp021-pycache PYTHONUNBUFFERED=1 uv run python /tmp/exp021_preflight.py > /tmp/exp021_preflight.log 2>&1
   ```
   Load byte-exact parent source from `1a8d0de` in memory and the candidate production helpers; record SHA-256 hashes; guard evaluators/test loaders. Use fixed real-CIFAR fixtures, equivalent initial inherited model/optimizer/RNG state, and seven alternating-order 512-step rounds per arm with 209 early ordinary, 205 early CutMix, 49 late ordinary, and 49 late SAM steps. Time exactly the production charged region, including training head/pool/joint loss, counters, one sparse norm sample, SAM, optimizer, and synchronization; use primary step IDs 2..513. Compute median, MAD/median, p90, parent drift, `floor(25560/R)` projected steps and `ceil(projected_steps/195)` projected epochs. Include a 32+1,024-step allocation trace after all persistent state. Before timing, validate formulas/schema and record GPU UUID/visibility/compute-process state; recheck those environment fields afterward. Emit raw path/round times, formulas, hashes, call/audit counts, allocation, finiteness, evaluator calls, and environment checks in one JSON vector, then apply fixed gates.

5. **Exactly one metric launch and integrity verification** (600-second timeout):
   ```bash
   rm -f run.log
   timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONPYCACHEPREFIX=/tmp/exp021-pycache PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1
   ```
   Require exit 0; training `[299.5,301.0]`; total `<600`; one evaluation per epoch; inherited config and 2,750,180 parameters; exact primary loss calls `=num_steps`, replay loss calls `=sam_applied_batches`, training head forwards `=num_steps+sam_applied_batches`, unchanged head-forward counter across every evaluation, zero nonfinite, complete inherited/appended summaries, and only `train.py` tracked. Report whether feature samples equal `1+floor((num_steps-1)/512)`; any mismatch disables conditioning inference but does not invalidate the metric. Head displacement is likewise report-only. Record whether realized steps meet 24,000 without using dose to override formal accuracy.

6. **Metric decision, durable evidence, and cleanup** (15-second parsing timeout):
   ```bash
   grep '^best_test_acc:\|^final_test_acc:\|^final_test_loss:\|^training_seconds:\|^total_seconds:\|^startup_seconds:\|^peak_vram_mb:\|^num_epochs:\|^num_steps:\|^num_params:' run.log
   ```
   Re-query EXP004=95.40. With goal integrity/budget valid, 95.50 or higher is improvement and anything lower no-improvement; a charged-time miss is no-improvement, scope/evaluator/device corruption invalid, and crash/no metric crash. Separately evaluate the composition conjunction (steps >=24,000, best >=95.60, tail mean >=95.50, best-final <=0.15) without changing the tree verdict. Transcribe all summary, final-16, feature-bin, displacement, dose, and counter values into `03-execute.md`; run independent result audit against raw evidence; then delete `run.log` and `/tmp/exp021-*` only after the audit.

### Informational Metrics (Optional)
- `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, `num_params`: final summary in `run.log`.
- Companion mechanism: total/new parameters, primary/replay/training-head/default-eval call reconciliation, terminal head displacement, ownership, nonfinite counts.
- Conditioning: cadence-512 stage-2 pooled-feature L2 count/mean/RMS in each charged-time quartile.
- Stability and exposure: final-16 values/mean/range/final/premium; steps/epochs versus EXP004 25,560/132; CutMix/SAM dose and paired latency.
