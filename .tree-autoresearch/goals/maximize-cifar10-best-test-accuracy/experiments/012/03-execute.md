# EXP-012: CutMix-complementary GPU Cutout

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-012
- **Base Node**: 011
- **Commit**: 3b5b48d
- **Outcome**: failed - exit-0 research no-improvement

## Implementation Notes

### Summary

Added a setup-owned `ComplementaryCutout` implementation to `train.py`: a reference-geometry 1,024-entry FP32 channels-last mask bank, private seed-43 device generator, preallocated centers/indices/selected-mask/area buffers, GPU-resident dose and center-support audits, and an in-place input mutation. The call is the lexical `else` of the unchanged CutMix decision inside the existing early `progress < CUTMIX_END` branch. Terminal reporting now checks complement equality, realized masked area, full center support, and bank invariants; evaluation log lines include charged time/progress for stable-tail analysis.

### Surprises & Discoveries

The parent already places its synchronization after augmentation, forward/backward, optimizer, and EMA work. Keeping Cutout's area sum and center histogram as device tensors therefore charges the complete mechanism without introducing a per-batch `.item()` synchronization. The fixed loader uses `drop_last=True`, so the deliberately strict full-batch input-shape assertion is compatible with every production batch.

### Decisions

Used FP64 only for the scalar accumulated masked-pixel count while masks, inputs, and per-image areas remain FP32; this preserves exact integer accumulation at the full expected dose without changing the multiplication dtype. Center support uses a preallocated integer histogram and `scatter_add_`, avoiding allocation-heavy `bincount`. The raw Claude implementation review is preserved in `03-implementation-review.md`; Claude passed the implementation and identified no required correction.

After the GPU smoke exposed the local `torch.sum` overload requirement, the final implementation uses an explicit `dim=(0,)` and preallocated scalar output. Claude re-reviewed the complete post-fix diff and returned `PASS`, explicitly clearing it for the metric run.

## Experimental Adjustments

- **No implementation deviation requiring a recipe change**: Claude's two notes were non-blocking (the full-batch assertion relies on the frozen `drop_last=True`; the image-count audit is redundant but harmless). (ref: `03-implementation-review.md`)

## Run Log

### Run 1

Metadata:
- **Job ID**: N/A (local CPU correctness harness)
- **Log file(s)**: terminal capture; `/tmp/exp012_correctness.py`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 UTC
- **Ended**: 2026-08-06 UTC

Description:
- Accuracy-blind exhaustive CPU verification of the mask bank, reference clipping, mean fill, outside-pixel identity, labels, layout, deterministic replay, parent RNG isolation, and a forced 64-batch CutMix/complement sequence. It imports candidate code without invoking `main`.

Observations:
- Passed all 1,024 masks with area min/mean/max 64/196/256. The forced sequence produced 33 CutMix and 31 Cutout batches, zero late calls, identical parent/candidate CutMix generator states, unchanged targets, and channels-last output. (source: terminal `cpu_correctness=PASS`)

Key Metrics:
- masks: 1,024 unique; CutMix/Cutout: 33/31; RNG isolation/target identity/channels-last: 1/1/1 (source: terminal `cpu_correctness=PASS`)

### Run 2

Metadata:
- **Job ID**: N/A (local GPU smoke)
- **Log file(s)**: terminal capture; `/tmp/exp012_gpu_smoke.py`
- **WandB**: N/A
- **Status**: completed after two bounded harness/code corrections
- **Started**: 2026-08-06 UTC
- **Ended**: 2026-08-06 UTC

Description:
- Full-WRN BF16/channels-last correctness smoke on the sole visible physical GPU 0. It checks private CUDA RNG replay/isolation, stable helper allocation, finite loss and gradients, an optimizer update, drop-path draw parity, production-faithful SAM perturb/replay/BN/restore, EMA updates, and exact evaluation swap restoration.

Observations:
- Passed on NVIDIA H20 after fixing the scalar-reduction API and a harness allocation-baseline placement. Across 1,102 helper calls, steady-state allocation growth was zero; EMA completed two updates and one exact restoration with no failure. (source: terminal `gpu_smoke=PASS`)

Key Metrics:
- loss: 2.534332; SAM gradient norm: 1.867053; EMA updates/restores: 2/1; isolated peak allocated: 1,604.964 MiB (source: terminal `gpu_smoke=PASS`)

### Run 3

Metadata:
- **Job ID**: N/A (local paired GPU preflight)
- **Log file(s)**: `/tmp/exp012_preflight.log`; harness `/tmp/exp012_preflight.py`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 UTC
- **Ended**: 2026-08-06 UTC

Description:
- Single decisive accuracy-blind five-round parent/candidate latency preflight on physical GPU 0. Arm order alternates by round; each round measures 100 parent-clean/candidate-Cutout steps plus fixed CutMix, late ordinary, and production-faithful SAM workloads. Both arms additionally execute 30 cadence-31 EMA updates and one exact swap/restore.

Observations:
- All preregistered gates passed without rerun: parent drift 0.016205 <=0.03, paired-ratio MAD/median 0.001261 <=0.005, median ratio 1.007941 <=1.01, projected exposure 25,594.756 >=25,500, and projected total 451.457s <600s. (source: `/tmp/exp012_preflight.log` `PREFLIGHT_JSON`)
- Each arm completed 200 CutMix steps, 130 SAM steps, 30 EMA updates split 15/15, one exact restore, and zero restore/coverage/nonfinite/RNG failures. Parent/candidate CutMix generator states remained equal. (source: `/tmp/exp012_preflight.log` `PREFLIGHT_JSON`)

Key Metrics:
- parent weighted medians: 10.717045, 10.751521, 10.779879, 10.890719, 10.778080 ms; candidate: 10.801798, 10.836898, 10.933435, 10.991041, 10.800266 ms (source: `/tmp/exp012_preflight.log`)
- paired ratios: 1.007908, 1.007941, 1.014245, 1.009212, 1.002058; median 1.007941; MAD/median 0.001261 (source: `/tmp/exp012_preflight.log`)

### Run 4

Metadata:
- **Job ID**: N/A (local helper benchmark)
- **Log file(s)**: terminal capture; `/tmp/exp012_helper_bench.py`
- **WandB**: N/A
- **Status**: completed after one harness correction
- **Started**: 2026-08-06 UTC
- **Ended**: 2026-08-06 UTC

Description:
- Direct timing of 1,000 exact Cutout helper calls after 100 warmups on physical GPU 0, including center RNG, index selection, mask multiplication, center histogram, and masked-area audit.

Observations:
- Deterministic helper benchmark passed with no steady-state allocation growth. (source: terminal `helper_bench=PASS`)

Key Metrics:
- total: 71.120236 ms; mean: 71.120236 us/call; allocation delta: 0; helper-only peak: 8.021 MiB (source: terminal `helper_bench=PASS`)

### Run 5

Metadata:
- **Job ID**: local PID 1573645 (launcher/session 3613)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-06 06:35 UTC
- **Ended**: 2026-08-06 06:42 UTC

Description:
- One fixed-seed local metric run of EXP-012 on physical GPU 0 after all deterministic and paired timing gates pass. It tests whether reference-geometry Cutout on every early non-CutMix batch raises the formal best accuracy above 95.71% while producing a stable final-16 EMA plateau. The run retains the complete EXP-011 model, optimizer, CutMix, SAM, EMA, budget, and evaluation cadence.

Observations:

- Startup printed CUDA, 2,748,890 parameters, the complete fixed parent configuration plus `cutout_size=16 cutout_seed=43`, the 300-second budget, and 195 batches per epoch. The process and worker pool are live. (source: `run.log` startup lines 1-5; local process check)
- The sole metric launch exited 0 after a complete 300.0-second charged run. It produced 131 evaluations for 131 epochs, split 105 live and 26 EMA, with 26 exact EMA swaps/restores. No traceback, CUDA/OOM, timeout, NaN/Inf, runtime error, overlap, or audit-failure signature was present. (source: `run.log` lines 1-286; evaluation/error scans)
- CutMix stopped at the frozen 75% boundary with 10,151/20,461 applications; complementary Cutout executed on all other 10,310 eligible batches, so `10151+10310=20461`. The 2,639,360 Cutout images had 517,469,257 masked pixels, mean area 196.058612, all 1,024 centers, correct 64/196/256 bank geometry, and `complement_ok=1`. (source: `run.log` lines 268-269)
- SAM began at step 20,462/progress 0.7500 and applied 2,458/4,915 = 0.5001. EMA produced 158 updates from progress 0.7518-0.9991 with exact 79/79 ordinary/SAM parity; restoration, coverage, nonfinite, and RNG failure counts were all zero. (source: `run.log` lines 270-275)
- The run reached `best_test_acc=95.52%`, 0.09 points below parent EXP-011 at 95.61 and 0.19 below the formal 95.71 threshold. Realized 25,376 steps were 124 below the 25,500 dose floor, so this is a valid no-improvement with a mechanism-dose shortfall, never a retry. (source: `run.log` lines 277-286; tree parent card)
- Final-16 EMA evaluations at epochs 116-131 spanned charged progress 0.860187-1.000000 and were `95.37, 95.33, 95.36, 95.35, 95.35, 95.38, 95.38, 95.40, 95.39, 95.46, 95.43, 95.48, 95.48, 95.50, 95.51, 95.52`; mean 95.418125 and range 95.33-95.52. All 16 observations occupy the preregistered `<95.59` falsified tail bin and the mean is 0.075 points below EXP-011's 95.493125 reference plateau. Because the conjunctive step-dose gate failed, this observed bin does not establish a clean full-dose causal falsification of Cutout efficacy. (source: `run.log` lines 237-267; `/tmp/exp012_final16.txt`)
- Preflight projected 25,594.8 steps but the run realized 25,376, a 218.8-step miss (0.86% of realized steps) and 124 below the fixed dose floor. The preflight included the SAM path, so the remaining gap is unreconciled production timing drift; it is handled conservatively by the dose gate rather than retroactively changing or retrying the measurement. (source: `/tmp/exp012_preflight.log`; `run.log` line 285)
- After Claude's final `PASS`, analysis report completion, commit `3b5b48d`, and tree insertion, raw `run.log`, parent snapshot, preflight log, and all temporary harnesses were removed before advancing. All exact values and adversarial recomputations remain in this durable record and `04-analysis.md`.

Key Metrics:

- `best_test_acc`: 95.52%; `final_test_acc`: 95.52%; `final_test_loss`: 0.1557 (source: `run.log` lines 277-279)
- `training_seconds`: 300.0; `total_seconds`: 450.1; `startup_seconds`: 1.1 (source: `run.log` lines 280-282)
- `peak_vram_mb`: 1,228.4; `num_epochs`: 131; `num_steps`: 25,376; `num_params`: 2,748,890 (source: `run.log` lines 283-286)
- CutMix: 10,151/20,461 = 0.4961; Cutout: 10,310 calls / 2,639,360 images / mean area 196.058612 / all-step share 0.406289; SAM: 2,458/4,915 = 0.5001 (source: `run.log` lines 268-270)
- EMA: 158 updates, 79/79 parity, 105 live + 26 EMA evaluations, 26 exact restorations, all integrity failure counts zero (source: `run.log` lines 271-275)
- Final-16 EMA plateau: mean 95.418125, range 95.33-95.52, charged progress 0.860187-1.000000 (source: `run.log` lines 237-267)

## Verification Results

### Conditions Checked

- **Primary accuracy improvement**: failed. `best_test_acc=95.52%` is 0.09 points below parent EXP-011 at 95.61 and 0.19 below the required parent+0.10 threshold of 95.71. Formal verification stops here. (source: `run.log` line 277; `tree.sh show ... 011`)
- **Successful bounded execution**: skipped by formal verification after the primary necessary-condition failure. Prior classification evidence records exit 0, 300.0 charged seconds, 450.1 total seconds, 131/131 evaluations, complete summary, and no integrity signature. (source: Run 5 observations)
- **Mechanism dose and parent integrity**: skipped by formal verification after the primary necessary-condition failure. Prior classification records a step-dose shortfall at 25,376 <25,500, while complement, area, EMA sample, restoration, RNG, and state audits passed. (source: Run 5 observations; `03-result-review.md`)
- **Claude adversarial result integrity**: completed before formal verification. After one causal-wording correction, Claude returned `PASS` for the exit-0 no-improvement classification and exact durable values. (source: `03-result-review.md`)

### Informational Metrics

- `best_test_acc=95.52%`, `final_test_acc=95.52%`, `final_test_loss=0.1557`
- `training_seconds=300.0`, `total_seconds=450.1`, `startup_seconds=1.1`
- `peak_vram_mb=1228.4`, `num_epochs=131`, `num_steps=25376`, `num_params=2748890`
- CutMix `10151/20461=0.4961`; Cutout `10310` calls, mean area `196.058612`; SAM `2458/4915=0.5001`; EMA `158` updates with `79/79` parity
- Final-16 EMA observed tail: mean `95.418125`, range `95.33-95.52`, progress `0.860187-1.000000`; observed bin falsified but full-dose causal inference unavailable

## Errors & Dead Ends

### 2026-08-06 - CPU harness missing repository import path
- Error: `ModuleNotFoundError: No module named 'prepare'` before candidate import completed.
- Root cause: Loading `train.py` by absolute module spec does not implicitly add its parent directory to `sys.path`.
- Source: `/tmp/exp012_correctness.py` initial run, import line 13.
- Do NOT retry: Do not import the candidate by absolute spec without explicitly placing the read-only repository root on `sys.path`.

### 2026-08-06 - Scalar reduction `out=` required an explicit dimension
- Error: `TypeError: sum() received an invalid combination of arguments - got (Tensor, out=Tensor)` on the first GPU Cutout call.
- Root cause: This PyTorch overload accepts `out=` only when the reduction dimension is explicit; the allocation-stable scalar audit used the unsupported shorthand.
- Source: `/tmp/exp012_gpu_smoke.py` first run; candidate `train.py` `ComplementaryCutout.apply_`.
- Do NOT retry: Use `torch.sum(selected_areas, dim=(0,), out=batch_masked_pixels)` for an allocation-stable scalar reduction on this environment.

### 2026-08-06 - GPU smoke allocation baseline preceded persistent input allocation
- Error: Stable-allocation assertion reported `(19968000, 23113728)`, an exact 3,145,728-byte increase.
- Root cause: The harness captured `allocated_before` before creating its persistent `[256,3,32,32]` FP32 working input; the helper itself had already completed successfully.
- Source: `/tmp/exp012_gpu_smoke.py` second run, allocation assertion.
- Do NOT retry: Allocate every persistent benchmark tensor and complete warmup before capturing the steady-state allocator baseline.

### 2026-08-06 - Helper benchmark used unsupported `torch.ones` memory format
- Error: `TypeError: ones() received an invalid combination of arguments` before the candidate helper was called.
- Root cause: This PyTorch build accepts `memory_format` for `torch.empty` but not `torch.ones`.
- Source: `/tmp/exp012_helper_bench.py` initial run, input construction.
- Do NOT retry: Construct channels-last filled tensors with `torch.empty(..., memory_format=...).fill_(value)`.

## Human Notes

> The user requires physical GPU 0 and Claude as the sole adversarial reviewer. Never substitute a non-adversarial or non-Claude reviewer; pause for the user if Claude is unavailable.
