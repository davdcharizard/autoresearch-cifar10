# Plan EXP-027: CutMix-Off, RandAugment-On Refinement Window
- **Created**: 2026-08-06

## Milestones

### Milestone 1: Forkserver-safe temporal CutMix policy
- [x] Create branch `autoresearch/maximize-cifar10-best-test-accuracy-027` from the integration branch; confirm baseline `94.15%`, threshold `94.25%`, and `TIME_BUDGET_S=300`.
- [x] Modify only `train.py`: a module-level picklable collator reads one forkserver-context shared boolean snapshot per batch, applies the accepted 50% alpha-1 CutMix only while enabled, and returns explicit policy provenance.
- [x] At the first completed step at or above 70% counted time, set the shared flag false exactly once without breaking/rebuilding the strong loader; retain N1/M7 until the unchanged 80% weak-loader rebuild.
- [x] Pass compile, Ruff, format, diff/scope, model/optimizer/schedule/evaluator, direct target/RNG, forkserver-pickling, and shared-flag visibility checks.

### Milestone 2: Exact policy and lifecycle evidence
- [x] Build guarded `preflight_cutmix_bridge.py`; validate forced/natural policy-on hard/CutMix and policy-off hard semantics, source immutability, one flag read, no off-policy RNG draw, and surrounding CPU/CUDA RNG neutrality.
- [x] Create one EXP027-owned immutable 200-record post-N1/M7 pre-policy corpus before either arm, using the first unfiltered production-ordered batches and exact worker RNG states; atomically persist/fsync it, record its SHA-256, verify every tensor/state digest, and forbid any rematerialization after observing counts or trajectories.
- [x] Run a common accepted-policy prefix, clone exact model/optimizer state, then compare accepted CutMix-on and hard-N1/M7 continuations on identical immutable records; require finite complete state, loss-EMA ratio <=1.5, and no candidate-only >95% class concentration.
- [x] Pass one 20,000-collation real-loader lifecycle with production-style iterator recreation every `len(loader)` batches: 15,000 policy-on deliveries at 47.5-52.5% CutMix, one mid-epoch parent flip, propagation across any epoch boundary to all eight workers within 24 delivered batches, then 5,000 consecutive policy-off hard N1/M7 batches, unchanged worker PID set, exact strong shutdown, weak rebuild <5s, hard weak batch, weak shutdown, and zero live children.

### Milestone 3: Real-loader timing and exposure
- [x] On one idle H20, run one unscored conditioner plus five alternating fresh-process control/candidate pairs with >=100 warmups and the scaled 800-strong/200-weak production curriculum.
- [x] Require candidate/control counted mean <=1.01, every pair <=1.04, per-arm CV <3%, projected steps >=26,629, loader headroom/wait and wall/count gates, peak <650 MiB, no worker/allocation growth, weak rebuild <5s, and projected total <540s.
- [x] Record natural CutMix counts, exact request/drain timing, zero soft targets after drain, worker identities, policy identities, backend flags, and fsynced raw trials before any timing veto.

### Milestone 4: One scored run and ordered verification
- [x] Confirm only `train.py` differs, no stale run-log variant exists, and exactly one idle H20 is available; run once as `timeout --kill-after=5s 595s uv run train.py > run.log 2>&1`.
- [x] Establish exit zero and one complete finite ten-field summary before metric classification; require `best_test_acc>=94.25%`, `299.9<=training_seconds<=300.2`, and `total_seconds<600` in that order.
- [x] Verify >=26,629 steps, exactly one CutMix-off request in `[69.9,70.2]%`, no policy-on delivery after 70.5%, one 80% weak switch with eight workers stopped, hard weak targets, production policy counts, unique at-most-once-per-epoch evaluations, and no retry.

## Code Changes

- **`train.py`**: import `multiprocessing`; define `CUTMIX_OFF_FRACTION=0.70`, policy constants, and module-level `PhaseCutMixCollator`. Construct a synchronized boolean from `multiprocessing.get_context("forkserver")` inside guarded `main()` and pass the callable object to the initial strong loader. The collator snapshots the flag once, preserves the accepted forked-RNG CutMix semantics only when enabled, and returns `(inputs, targets, policy_enabled)`.
- **`train.py`**: conditionally unpack strong triples versus accepted weak pairs; count policy-on hard/CutMix, policy-off hard, post-request policy-on drain, and total strong batches. Immediately after the first completed step at/above 70%, set the flag false once and print request provenance without breaking the iterator, evaluating, rebuilding, reseeding, or changing LR. At 80%, retain the existing shutdown/rebuild and print complete realized phase counts/drain facts.
- **Ignored diagnostics**: experiment-local guarded preflight/timing controllers and JSON evidence only. They may import production functions but cannot change tracked files or create `run.log`.

## Configuration Changes

- `CUTMIX_OFF_FRACTION`: new fixed value `0.70`; it begins one 10%-of-budget hard-label N1/M7 bridge before the accepted 80% weak transition.
- Strong policy: accepted 50% alpha-1 CutMix remains unchanged until the request; after bounded prefetch drain it becomes 100% hard labels while crop/flip/N1/M7 stays unchanged.
- Unchanged: seed42, batch128, FP32 width-2 postactivation ResNet-20 (1,073,962 parameters), ordinary SGD momentum0.9/all-parameter decay1e-4, LR0.1 through80%, accepted cosine weak tail, evaluator, workers/prefetch, and 300-second counter.

## Execution Environment

- Method: local ignored CPU/forkserver semantic and lifecycle controller; local paired H20 timing controller; then one local production command.
- Resources: exactly one idle NVIDIA H20 near 97,871 MiB for GPU work, eight persistent forkserver workers, existing CIFAR-10 data, no new dependency.
- Estimated runtime: 8-15 minutes preflight/lifecycle, 5-7 minutes timing, and about 5.5 minutes production.
- Log output: diagnostics write experiment-local JSON; production alone writes root `run.log`, never `tee` or full-stream output.
- Tool skill: none; local autoresearch controllers supervise the run.
- Safeguards: every controller prepends project root, uses a guarded main, serializes/fsyncs evidence before assertions, launches deterministic CUDA children with `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and explicitly shuts every diagnostic loader.
- Planning feasibility: guarded `ipc_probe.py` confirmed a locked forkserver-context `Value` can be carried by a pickled persistent-worker collator, propagated from `True` to `False`, and cleaned up on this Python/PyTorch environment. Execution must repeat the proof with all eight production workers.

## Abort Criteria

- Abort on any tracked edit outside `train.py`; any model, optimizer, LR, seed, budget, evaluator, transform, CutMix alpha/probability, batch, worker/prefetch, 80% boundary, precision, or weak-tail drift.
- Abort if the flag is created outside the guarded main/forkserver context, read more than once per collator call, mutated by a worker, or implemented with manager/Python/NumPy/CUDA RNG; abort on a 70% loader break/rebuild/reseed or new evaluation.
- Abort on target/provenance/RNG/source mismatch, corpus/digest mutation, non-finite or incomplete continuation state, candidate-only concentration, loss-EMA ratio >1.5, unpersisted veto evidence, or forced/filtered/rematerialized corpus data.
- Abort if any worker misses the flag, a policy-on tag appears beyond 24 post-request deliveries or 70.5%, any soft target appears after drain, worker PIDs change/grow/leak, policy-on CutMix falls outside 47.5-52.5%, or weak lifecycle fails.
- Abort on timing/exposure/loader/memory/wall miss, GPU contention, OOM/CUDA/worker error, production timeout, non-finite loss, or incomplete summary. Do not rescue with another boundary, rebuild, probability, alpha, seed, prefetch, threshold, or pure accepted fallback.

## Verification Protocol

### Verification Procedure

1. Query the baseline and budget (30s): `bash /root/david/.codex/plugins/cache/deoxys/linear-autoresearch/3.0.5/skills/shared/scripts/exp-index.sh baseline .autoresearch/goals/maximize-cifar10-best-test-accuracy/04-results.tsv` and `rg '^TIME_BUDGET_S\s*=' prepare.py`; require `94.15`, commit `7c1e7d8`, and 300 seconds.
2. Run `uv run python -m py_compile train.py`, `uv run ruff check train.py`, `uv run ruff format --check train.py`, `uv run pre-commit run --all-files`, and `git diff --check` (90s). Require `git diff --name-only 7c1e7d8` to print only `train.py`; assert exact constants, one policy draw, one shared snapshot, model parameters, optimizer group, evaluator, and unchanged schedule/source structure.
3. After creating each ignored controller, run its `--help` before GPU work to prove forkserver import safety. Query `nvidia-smi --query-gpu=index,name,memory.total,compute_mode --format=csv,noheader` and `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` before every GPU command; require one idle H20.
4. Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 900s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/027/preflight_cutmix_bridge.py` (<=15m). Require all Milestone-2 semantic, exact-corpus continuation, 20,000-collation, propagation, and lifecycle gates; the final JSON report must exist and be status `pass`.
5. Run `CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 480s uv run python .autoresearch/goals/maximize-cifar10-best-test-accuracy/experiments/027/timing_cutmix_bridge.py` (<=8m). Require all Milestone-3 five-pair gates and a fsynced passing timing report.
6. Require no `run*.log`, reconfirm scope/static checks and idle GPU, then execute the Milestone-4 production command exactly once. Monitor only bounded status/tail excerpts.
7. After process completion, first require exit zero and exactly ten finite summary fields. Then compare `best_test_acc` with `94.25`; a lower finite result is `no-improvement` and cannot be rerun. If it passes, require the fixed counted/total time conditions; protocol failures are `invalid` and no-result infrastructure failures are `crash`.
8. Independently parse all request/switch/evaluation records. Require one request in `[69.9,70.2]%`, bounded drain ending by 70.5%, one switch in `[79.5,80.5]%`, eight stopped workers, no post-drain/weak soft target, >=26,629 steps, 1,073,962 parameters, valid count identities, registered proportions, unique evaluation epochs, one scored run, and the expected report/log hashes.
9. A `>=94.25%` result satisfies the user-defined moving-baseline rule and is committed, but the report must call a marginal one-seed gain weak evidence for the net three-phase curriculum. It may not claim an effect-size estimate or isolate pasted-pixel removal from hard-target removal; no rerun or seed selection is permitted.

### Informational Metrics (Optional)

- Final metrics: parse `final_test_acc`, `final_test_loss`, `training_seconds`, `total_seconds`, `startup_seconds`, `peak_vram_mb`, `num_epochs`, `num_steps`, and `num_params` from the complete summary.
- Mechanism trajectory: parse 70% request counts/drain, 80% switch accuracy, first weak accuracy, best/final epoch and accuracy, final NLL, CutMix/hard-N1/M7 counts, and best-final gap from `run.log`.
- Evidence provenance: record immutable corpus SHA, semantic/safety/lifecycle/timing report hashes, paired timing ratios/projection, worker PID facts, and production log SHA.
