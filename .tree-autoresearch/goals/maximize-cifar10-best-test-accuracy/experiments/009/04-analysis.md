# Report EXP-009: Late-Stage Identity SE Preflight Reject
- **Created**: 2026-08-05

## Goal

Improve CIFAR-10 `best_test_acc` from EXP-002 at 95.23% through an additive representation mechanism. Formal improvement required 95.33%; the preregistered mechanism target was 95.53%. Global best remains EXP-004 at 95.40%.

## Idea & Hypothesis

After four failed children accumulated at EXP-004, EXP-009 forked from EXP-002 to test channel recalibration without the later SAM package. Claude selected identity-centered SE over ECA because memory and modest step loss were not binding, while SE offered fuller cross-channel capacity. Four 128/256-channel gates were expected to become active, retain at least 26,000 steps, and reach 95.53%.

## Approach

Only `train.py` changed. Four raw-parameter SE modules add 25,408 parameters. Dedicated generator initialization preserves every parent draw; zero final layers and `2*sigmoid` preserve the initial parent function. Pooled `conv2` descriptors are standardized across channels and processed in an explicit FP32 region. Gate parameters use a separate 5x-LR/no-decay group. Sampled training-only device buffers audit activity without changing evaluation state.

## Execution

Static, inventory, construction, CPU/BF16 initial-output, staged-gradient, optimizer-ownership, and inference-mode checks passed. The 200-step physical-GPU-0 smoke showed every gate was active and finite without broad saturation. A production-faithful parent/candidate latency benchmark then failed the fixed gate, so no full training or evaluation was launched and no `run.log` exists.

## Results

- **Primary metric**: unavailable (parent: 95.23%; global best: 95.40%)
- **Inventory**: 4 modules, 16 gate tensors, 25,408 gate elements, 2,774,298 total parameters
- **Live-gate mean deviations**: 0.202099 / 0.137746 / 0.121413 / 0.034505
- **Fractions beyond two BF16 ULPs**: 0.915283 / 0.880506 / 0.885749 / 0.647850
- **Saturation fractions**: 0.001388 / 0.000081 / 0.000076 / 0; nonfinite counts all zero
- **Parent latency median / p90 / mean**: 10.087693 / 10.285463 / 10.192185 ms
- **Candidate latency median / p90 / mean**: 12.177348 / 12.979396 / 12.286809 ms
- **Median ratio / projected steps**: 1.20715 / about 23,154, versus limits 1.075 / 26,000

The mechanism itself was live, so this is not a BF16 dead-zone failure. The limiting issue is execution granularity: four FP32 pool/standardize/bottleneck/sigmoid paths add many small charged operations, producing 20.7% median overhead despite less than 1% parameter growth. The result discredits only this fixed late-four FP32 package under the time budget; it says nothing about SE accuracy or cheaper channel recalibration.

- **Key Learning**: Active late-stage FP32 SE adds 20.7% median latency to the compact WRN, overwhelming its sub-1% parameter cost before accuracy can be tested.

## Verification

- **Conditions**: Implementation and live-gate gates passed; parent-relative latency/exposure failed; full-run and accuracy conditions were skipped.
- **Review Notes**: Physical GPU 0 was the H20 and the unrelated 3,384 MiB co-tenant was stable. The gate was preregistered. No metric, retry, or configuration selection occurred.
- **Verdict**: crash
- **Verdict Basis**: Mechanical tree encoding for no result produced. This was a preflight reject, not a code crash or negative accuracy result.
- **Tree placement**: failed leaf from EXP-002 on `br-000`, commit `762b609`; best remains EXP-004.

## Unexplored Avenues

- Final-stage-only SE or a narrower bottleneck could reduce launches/work, but would be a new fixed package and must not be treated as a retry of this node.
- BF16 excitation could be cheaper, but it reopens the precision/dead-zone issue and needs its own integrity design.
- ECA avoids dense channel MLPs but has a much smaller mechanism ceiling and still incurs several launches per block.

## Next Steps

- **High confidence**: Prefer a single-kernel or existing-convolution representation change rather than another multi-launch attention module.
- **Medium confidence**: Explore architecture reallocation from EXP-002, where capacity can move without CPU work or many small GPU kernels.
- **Low confidence**: Revisit final-stage-only SE only after an isolated latency model shows a credible path below 7.5% overhead.
