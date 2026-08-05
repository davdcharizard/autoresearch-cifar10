# Plan EXP-045: ResNet-D Projection Shortcuts
- **Created**: 2026-07-27

## Milestones

### Milestone 1: Implement exact two-transition shortcut averaging
- [x] Create EXP045 branch from accepted `a7c42dc`; modify only `train.py` and freeze evaluator/`prepare.py`.
- [x] Register parameterless2x2/stride2 average pools only for stride2 blocks, change only those shortcut projections to stride1, and preserve identity/layer1/main paths.

### Milestone 2: Prove topology, phase aggregation, gradients, and state
- [x] Add ignored evaluator-free `preflight.py`; prove exact two-pool membership/shapes, common parameter/buffer bytes/order/RNG, and independent four-phase forward/gradient oracles.
- [x] Prove unchanged block-local main branches and identity shortcuts, ordinary Nesterov updates, replay, accepted controls, and report shortcut/main RMS ratios without tuning.

### Milestone 3: Protect exposure
- [x] Run interleaved paired full-step H20 timing; require CVs, every retention>=0.9746439096, median projected>=127 passes, and peak<2,048MiB.

### Milestone 4: Sole score
- [x] Reaudit baseline94.48/source/resource/log and run exactly `timeout 600s uv run train.py > run.log 2>&1` once.
- [x] Require valid300-second completion,1,003,482 params, correct cadence/transitions; improvement needs best>=94.58 and mechanism support also needs>=127 realized passes.

## Code Changes

- **`train.py` / `PreActBlock.__init__()`**: construct the existing `self.shortcut` with stride1 when block stride is2, otherwise the accepted stride. Immediately afterward register `self.shortcut_pool = nn.AvgPool2d(kernel_size=2,stride=2)` iff stride==2, else `None`. Defaults remain padding0/ceil_modeFalse/count_include_padTrue/divisor_overrideNone. Pool has no state/RNG.
- **`train.py` / `PreActBlock.forward()`**: preserve `preactivated=relu(bn1(x))`. If `self.shortcut is None`, use raw `x` exactly. Otherwise pool `preactivated` only when `shortcut_pool` exists, then apply the existing projection. Main `conv1/conv2` path remains byte-identical.
- **ignored `experiments/045/preflight.py`**: no test/evaluator construction; loads exact accepted source and candidate, prints before assertions, never writes score output.

## Configuration Changes

- `layer2[0]`/`layer3[0]` shortcut: 1x1 stride2 phase sample -> AvgPool2d(2,2) plus same1x1 stride1.
- `layer1[0]` remains direct1x1 stride1; four identity shortcuts (`layer1[1]`, `layer2[1]`, `layer3[1]`, `layer3[2]`) remain raw-x; main strides remain `[1,2,2]` at stage entries.
- Parameters remain1,003,482 across52 parameter tensors (97 state-dict entries including BatchNorm buffers), with identical names, bytes, order, initialization, decay groups, and seed trajectory.
- All accepted data, mixup/RandAugment, head, objective, LR, optimizer, evaluator, budget, and cadence remain exact.

## Execution Environment

- Method: offline local semantic/timing gates then one local score; no remote/network/gh/install/W&B.
- Resources: one idle NVIDIA H20, local CIFAR, installed `uv`,8 persistent workers.
- Estimated runtime: semantics<=180s, timing<=240s, score about345s and killed at600s.
- Log output: preflight stdout; sole score project-root `run.log`, removed before and after loop.
- Tool skill: `/research-execute` only.

## Abort Criteria

- Abort before timing on any scope/state/RNG/topology/shape/phase/main-path/identity/gradient/update/replay/control failure; repair only verifier or implementation defects without changing pool/kernel/order/transition/scale.
- Abort before score on nonfinite timing, arm CV>5%, pair-ratio CV>1%, any retention below floor, median projected<127, peak>=2,048MiB, or contention. Diagnostics cannot select gain/kernel/transition.
- Abort score on timeout/nonzero/OOM/worker/nonfinite/no-output60s/malformed summary/wrong state or cadence. Never rerun a valid score.

## Verification Protocol

### Verification Procedure

1. Run/persist/assert: index baseline query; `git rev-parse HEAD`; `git branch --show-current`; SHA-256 of current and `git show a7c42dc:{train.py,prepare.py,pyproject.toml}`; `git diff --name-only a7c42dc` exactly `train.py`; frozen-file `git diff --exit-code`; `git diff --check`; ignored harness; local data/absent log/compilation; `printenv CUDA_VISIBLE_DEVICES` (record unset as such); Python CUDA count/name exactly `1/NVIDIA H20`; `nvidia-smi -L`, utilization/memory/process queries showing the sole H20 idle. Require baseline94.48 at `a7c42dc`, threshold94.58, and EXP045 branch.
2. Run `timeout 180s uv run python .autoresearch/goals/maximize-cifar10-test-accuracy/experiments/045/preflight.py semantics`; resolve root in `sys.path`, stub Eval, block dataset construction, load accepted via exact git source.
3. Diff/AST allow only shortcut construction/forward edits. Require exactly pools at `layer2.0`/`layer3.0`, projection conv strides `[1,1,1]`, main entry conv strides `[1,2,2]`, expected shape table, exactly1,003,482 params/52 parameter tensors/97 state-dict entries, and no pool state.
4. From cloned seed42 states require all named parameter/buffer keys/bytes/order, optimizer membership/options/state, pooled-head weights, post-construction CPU/CUDA RNG exact. Pool forward/backward consumes no RNG.
5. On fixed transition tensors, capture production pool input/output and projection output. Compare to independent reshape `[N,C,H/2,2,W/2,2]` mean over phase axes then independent1x1 einsum within FP64 `1e-10/1e-12` and FP32 `2e-5/2e-7`. Four phase impulses must each contribute exactly quarter; accepted shortcut selects only `(0,0)`.
6. Instantiate accepted/candidate transition blocks with identical weights/input. Capture conv2 main output before addition and require byte equality. Require shortcut difference on phase-sensitive fixtures, operand shape equality, finite sum, layer1 direct preactivated projection, and identity blocks byte-equal raw `x` shortcut.
7. Bind gradient hooks to the actual production `preactivated` tensor before pool and actual projection output. With fixed independent upstream `U`, pooled `a`, projection weight `W`, require `dW[o,c]=sum_nij U[n,o,i,j]*a[n,c,i,j]`; `da[n,c,i,j]=sum_o U[n,o,i,j]*W[o,c]`; and each of four production `dz` phase values equals `da/4`, in FP64/FP32 bounds. The accepted control must put full `da` on phase(0,0) and exact zero on the other three. Verify fresh/preseeded Nesterov updates and every parameter/buffer; replay early/hard fixtures.
8. Report candidate/accepted shortcut RMS and main/shortcut RMS ratios for both transitions, plus logit/loss/gradient deltas, without gates or tuning. Require only finite nonzero treatment.
9. Re-prove accepted constants/data/loss/head/optimizer outside diff, mixup RNG, LR, strict65% transition, exhausted RandAugment, finite guard, seed42, one CUDA path, sole backward/step, and unique every-fifth-plus-final evaluator condition.
10. Run `timeout 240s uv run python .../experiments/045/preflight.py timing`. Use pinned host fixtures/full production post-loader body and warm>=20 each arm/regime. Execute the exact 8-window block `AE,CE,AH,CH,CH,AH,CE,AE` twice (16 retained windows total, each>=50). In each block pair indices `(AE0,CE1,AH2,CH3)` and reverse `(AE7,CE6,AH5,CH4)`, yielding exactly four local early ratios, four hard ratios, and four combined retentions.
11. Before every window create/load only that arm on GPU from the same preregistered parameter/buffer bytes and identical empty or fixed-preseeded optimizer state; never carry evolution or copy evolved state between arms. Restore identical input/RNG. Keep one live GPU model at a time; synchronize/reset peak stats immediately before every candidate window and sample immediately after. Print all16 windows, arm CVs, four regime ratios/CVs, four retentions, median projection, peak. Require arm CV<=.05, ratio CV<=.01, every retention>=0.9746439096, median projection>=127, peak<2,048MiB.
12. Re-run/persist every step1 source/resource/process audit immediately before score, remove stale log, run exact score once in the same environment. Require exit0, `Device: cuda`, one summary,300.0-300.1 counted,<600 wall,1,003,482 params, one ordered mixup/RandAugment transition, unique cadence, no errors.
13. Passes=`steps*256/50000`. Goal improvement requires best>=94.58; mechanism support also passes>=127. Final accuracy/loss descriptive. Normal-exposure miss closes exact two-pool treatment and immediate one-transition/kernel/order/gain/main-branch variants.

### Informational Metrics (Optional)

- `run.log`: best/final/loss, timing, VRAM, epochs/steps/passes, params, transitions/evaluations.
- Preflight: topology/oracle errors, RMS ratios, timing windows/CVs/retention/projection/peak.
