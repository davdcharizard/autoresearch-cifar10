# Plan EXP-013: Fixed-scale-40 cosine-normalized classifier
- **Created**: 2026-08-06

## Adversarial Review Resolution

- Bias freezing is pinned inside `PreActWideResNet.__init__` immediately after the unchanged initializer, before `main` builds optimizer/SAM/EMA ownership. `num_params` retains its historical stored-count meaning; a new `num_trainable_params` line reports the frozen-bias difference.
- The early trace shares each single-loader materialized batch between arms and diagnoses only initial conditioning. No durable matched EXP-011 trace exists at 25%/50%, so slow-start versus null transfer over the full horizon remains explicitly unresolved rather than inferred from 200 steps.
- Preflight totals and timeout are explicit. Every pre-metric terminal rejection is recorded as a `crash`/`NaN` tree leaf; there is no quiet abandonment or proxy free look.
- The `>=25000` projected-step gate deliberately admits a band below the 25,300 mechanism target because the frozen goal's formal accuracy verdict is primary. If projected or realized dose is below 25,300, the run may still be a genuine formal improvement but cannot support a full-dose mechanism claim.

## Milestones

### Milestone 1: Implement one isolated classifier-geometry change
- [x] Add fixed `COSINE_SCALE=40.0`, `COSINE_TEMPERATURE=0.025`, and `COSINE_EPS=1e-6`; no learned scale, alternate scale, margin, label smoothing, loss change, or post-metric tuning is allowed.
- [x] Preserve `self.fc = nn.Linear(256,10,bias=True)` and the parent's complete initialization order, then call `self.fc.bias.requires_grad_(False)` inside `PreActWideResNet.__init__` immediately after `self.apply(self._weights_init)`. Keep its name/shape in the state dict and optimizer group but exclude it through the existing `requires_grad` SAM filter.
- [x] Replace only the final affine forward with FP32 `40 * F.linear(F.normalize(features), F.normalize(fc.weight))` under an autocast-disabled block. Preserve pooled feature shape, raw classifier weight ownership, CutMix losses, and the full backbone path.
- [x] Print classifier type/scale/temperature/epsilon, stored and trainable parameter counts, frozen-bias status, and the unchanged parent configuration.

### Milestone 2: Add non-selective geometry and state audits
- [x] At setup, snapshot initial raw classifier row-norm min/mean/max and assert ten finite nonzero rows plus exactly one frozen parameter named `fc.bias`, with exact-zero bias.
- [x] After every existing once-per-epoch evaluation, regardless of live/EMA source and after any EMA online-state restoration, print the online raw row-norm min/mean/max and bias max. This is outside charged training and cannot change scale, stop a finite run, or select a checkpoint.
- [x] At terminal time, report initial/final online and EMA raw row norms, online/EMA off-diagonal row-cosine min/mean/max, raw and normalized-direction online-to-EMA distances, bias max, trainable/stored parameter counts, and SAM/optimizer/EMA ownership checks.
- [x] Treat nonfinite state, any row norm `<=COSINE_EPS`, nonzero bias, missing/extra frozen parameter, state-coverage/restoration failure, or an EMA/optimizer/SAM ownership mismatch as integrity failure. Finite norm shrinkage is informational and never triggers mitigation or retry.

### Milestone 3: Pass deterministic correctness and integration smokes
- [x] On CPU FP64/FP32, verify logits equal `40*cosine`, remain in `[-40,40]` within tolerance, are invariant to positive feature/individual-row scaling, and remain finite for zero/near-zero features under epsilon.
- [x] Materialize EXP-011 from `d68f73a`, import parent/candidate without invoking `main`, and prove every stored initialized tensor plus post-construction CPU/CUDA RNG state is identical. Require only `fc.bias` trainability to differ after candidate construction.
- [x] Verify CutMix pixels, targets, lambda, loss formula, dedicated RNG states, global RNG, and drop-path draw count remain parent-identical for a scripted trace; cosine forward consumes no RNG.
- [x] On full WRN BF16/channels-last GPU 0, require FP32 finite logits/loss, finite nonzero gradients for all 43 trainable tensors, no bias gradient, correct channels-last backbone path, and one valid optimizer update.
- [x] Instantiate the candidate and then execute the actual `main` ownership order (optimizer from all stored parameters, `sam_parameters` from `requires_grad`, snapshots, EMA, exclusion checks) before re-running production-faithful SAM perturb/replay/BN/restore. Then exercise cadence-31 EMA updates and one evaluate/swap/restore; require exact state restoration, one BN/optimizer update, balanced cadence, and zero RNG/coverage/restore failure.

### Milestone 4: Run the one-shot accuracy-blind GPU-0 preflight
- [x] Confirm physical GPU 0 is the approximately 97,871 MiB NVIDIA H20 and expose only it with `CUDA_VISIBLE_DEVICES=0` before every GPU command.
- [x] Run one paired 200-batch early-optimization trace from bit-identical initialization. Iterate one seeded eight-worker DataLoader once; for each materialized transformed CPU batch, clone the same tensor/targets into both arms, use equal dedicated CutMix streams, and replay the same global CUDA drop-path state before each arm. Record parent/candidate CE at steps 1/25/50/100/200, candidate row norms, and finite-gradient status. This diagnoses only initial conditioning; mid-run slow-start versus null transfer remains unresolved. The trace is informational except nonfinite/integrity failure and cannot select a scale or trigger recipe changes.
- [x] Run five alternating parent/candidate BF16/channels-last timing rounds. Per arm per round measure 100 ordinary, 40 CutMix, 20 production-faithful SAM steps, and 40 evaluation forwards, for across-five-round totals of 500/200/100/200 respectively. Separately exercise 30 cadence-31 EMA samples and one exact EMA swap/restore per arm. Measure training paths and evaluation separately.
- [x] Weight charged training by EXP-011 counts `10512/25798` early clean, `10345/25798` CutMix, `2470/25798` late ordinary, and `2471/25798` SAM. The first complete measurement is decisive with no timing rerun: require parent drift `<=0.03`, paired-ratio `MAD/median <=0.005`, candidate/parent median `<=1.03`, projected steps `>=25000`, projected total `<600s`, and all correctness assertions. A projection below 25,300 is explicitly accepted only for formal-accuracy testing and precludes a projected full-dose mechanism claim. Peak allocation is informational; allocation failure blocks.

### Milestone 5: Execute one fixed-seed metric run
- [x] Remove stale `run.log`, reconfirm physical GPU 0, and launch exactly once with `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`.
- [x] Monitor process/GPU/log liveness without pruning on loss, training accuracy, row norms, or intermediate test accuracy. Abort only on exception, CUDA/OOM, nonfinite/integrity failure, SAM/EMA restoration failure, 120 seconds without process/GPU/log progress, or outer timeout.
- [x] Preserve raw run/preflight evidence through analysis and tree insertion. Durably transcribe the complete summary, classifier trajectory/audits, CutMix/SAM/EMA audits, source counts, final-16 EMA values/mean/range/progress, and preflight. Use Claude as sole implementation/result adversarial reviewer; on Claude failure pause for the user, never fall back.

### Milestone 6: Verify and classify
- [x] Apply integrity-first precedence: any classifier/state/RNG/evaluation integrity signature is `invalid`/`NaN`; otherwise any nonzero exit is crash/`NaN`; only exit 0 can be improvement/no-improvement.
- [x] Require exit 0, charged seconds `[299.5,301.0]`, total `<600s`, one evaluation per epoch, historical `num_params=2748890` stored parameters, new `num_trainable_params=2748880`, complete summary, only `train.py` changed, and no error signature.
- [x] Formal improvement is `best_test_acc >=95.71%` versus parent 95.61, regardless of informational mechanism targets. Below 95.71 is one valid no-improvement and never a scale change or retry.
- [x] Mechanism-supporting evidence requires at least 25,300 steps, at least 155 EMA updates, final-16 EMA mean `>=95.64%`, exact bias/norm/ownership audits, and intact CutMix/SAM/EMA dose. A shortfall limits causal interpretation but does not override a valid formal tree verdict.

## Code Changes

- **`train.py`**: Add fixed cosine constants; preserve linear construction but freeze/ignore zero bias; compute FP32 normalized classifier logits; add setup/per-epoch/terminal classifier audits and complete failure reporting. No other model, data, loss, optimizer, schedule, CutMix, SAM, EMA, or evaluator logic changes.

No other tracked file may change. Parent snapshots and accuracy-blind harnesses live only under `/tmp`; raw evidence remains until analysis/tree insertion and is removed before the next experiment.

## Configuration Changes

- `COSINE_SCALE=40.0`: one literature-anchored but sweep-selected transfer point; not an unbiased +0.29 expectation and never tuned locally.
- `COSINE_TEMPERATURE=0.025`: exact reciprocal audit value.
- `COSINE_EPS=1e-6`: numerical protection only; row norms at or below it invalidate the mechanism.
- `fc.bias.requires_grad=False`: bias remains stored/zero for construction, state, and EMA parity but is unused in forward.
- All parent hyperparameters, including the 300-second budget, remain read-only and unchanged.

## Execution Environment

- Method: local CPU correctness, local GPU-0 integration/early-trace/timing preflight, then one local fixed-seed metric run.
- Resources: physical GPU 0 only through `CUDA_VISIBLE_DEVICES=0`; NVIDIA H20 approximately 98 GB; existing `uv` environment; no dependency changes.
- Estimated runtime: about 3 minutes for correctness/preflight plus approximately 450 seconds for the metric run; every command bounded, metric run capped at 600 seconds.
- Logs: `/tmp/exp013_*` for transient harness/preflight output and repository-root `run.log` for the metric run, with exact durable transcription before cleanup.
- Tool skill: local execution; no remote submission.

## Abort Criteria

- Wrong physical/visible GPU; tracked change outside `train.py`; change to frozen evaluator, data stream, seed-42 streams, budget, validation cadence, backbone, loss, optimizer hyperparameters, LR/drop path, CutMix, SAM, or EMA semantics.
- Any alternate/learned scale, use of affine bias/logits, angular margin, label smoothing, post-metric scale selection, or second metric launch.
- Formula, dtype, bound, scaling-invariance, initialization/RNG parity, frozen-bias, gradient, CutMix, SAM, BN, EMA, optimizer-identity, coverage, swap, or restoration assertion failure.
- One-shot preflight parent drift `>0.03`, ratio dispersion `>0.005`, median latency ratio `>1.03`, projected steps `<25000`, projected total `>=600s`, allocation failure, or nonfinite early trace. Finite early loss/norm differences cannot abort or alter the recipe.
- Metric-run exception, CUDA/OOM/nonfinite/integrity error, 120 seconds without process/GPU/log progress, or 600-second timeout. Intermediate metrics and finite norm shrinkage cannot trigger abort.
- A terminal correctness/preflight rejection produces no metric launch and is recorded as `crash` with metric `NaN`; it is never a quiet abandonment. Straightforward code/harness errors may be fixed only within the execution skill's bounded retry policy, but a completed gate measurement is never rerun.

## Verification Protocol

### Verification Procedure

1. Query parent with `tree.sh show .../04-results.tsv 011`; require metric 95.61, formal threshold 95.71, and reference final-16 mean 95.493125 from EXP-011's report.
2. Before every GPU command run `nvidia-smi -i 0 --query-gpu=name,memory.total --format=csv,noheader` and a `CUDA_VISIBLE_DEVICES=0` PyTorch device query; require one visible NVIDIA H20 and approximately 97,871 MiB.
3. Run `python -m py_compile train.py`, `git diff --check`, `git diff --name-only d68f73a`, and `git status --short --untracked-files=all`; require exactly `train.py` as tracked code change and no unexpected repository-local harness/log. Run all Milestone-3 smokes under bounded timeouts.
4. Materialize `git show d68f73a:train.py` under `/tmp`, hash-check it against a fresh `git show`, import parent/candidate under distinct non-`__main__` names, and verify shared read-only `prepare`. Run the single early trace and exact per-round preflight under `timeout 420s` on GPU 0 without querying test accuracy; apply Milestone-4 gates without rerun. An outer timeout before a complete measurement is a pre-metric `crash`/`NaN`, not permission to resize and rerun the workload.
5. After passing preflight, remove stale `run.log` and launch once: `timeout 600s env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 uv run train.py > run.log 2>&1`. Preserve exit status and apply integrity-first classification.
6. Require `rg -c '  eval ep' run.log == num_epochs`, live+EMA sources equal epochs, and no match from `rg -n -i 'traceback|cuda error|out of memory|classifier_audit_failed|ema_audit_failed|runtimeerror|(^|[^a-z])(nan|inf)([^a-z]|$)' run.log` after investigation.
7. Extract the full summary, classifier configuration/norm trajectory/final audits, CutMix/SAM/EMA audits, and final 16 EMA-source evaluations. Formal improvement is best `>=95.71`; separately classify mechanism support against Milestone 6 and disclose literature selection/dimension transfer limits.
8. Run Claude-only raw result review, correct any evidence wording, finish analysis/tree insertion, verify commit scope, then delete `run.log`, parent snapshot, and all temporary harnesses before advancing.

### Informational Metrics

- Summary: final accuracy/loss, training/total/startup seconds, peak VRAM, epochs, steps, stored/trainable parameters.
- Classifier: initial/per-epoch/final online raw row norms; EMA raw norms; pairwise row cosines; online/EMA raw and normalized-direction distances; bias and ownership status.
- Parent preservation: CutMix applied/eligible, SAM applied/eligible/first boundary, EMA updates/parity/decays/distances, evaluation sources/restores/failures.
- Stability: final-16 EMA values, mean/range/progress span, final accuracy/loss, best epoch.
