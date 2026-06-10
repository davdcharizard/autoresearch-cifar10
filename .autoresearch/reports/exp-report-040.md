# Report EXP-040: cuDNN conv-algorithm autotuning (throughput → more epochs)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-040.md
- **Plan**: plans/plan-040.md
- **Log**: logs/exp-log-040.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s training-compute budget on
a single H20, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32**.
This experiment probed the only untried axis after 40 experiments — per-step THROUGHPUT under the
dt-gated budget — to test the pivotal open question: is the net convergence-bound at ~91 epochs (more
epochs would lift top-1) or epoch-saturated?

## Idea & Hypothesis
Chosen from brainstorm-040: set `torch.backends.cudnn.benchmark = True` so cuDNN benchmarks and caches the
fastest convolution algorithm for the fixed input shapes (128×3×32×32, `drop_last=True`). The budget
gates on `Σ per-step GPU dt`; the net is launch-bound (~8ms/step ≈ 13% of H20 bf16 peak), so any dt
reduction converts directly into more epochs at the same 300s. Selected as the lowest-risk, legitimate
throughput lever (same mechanism as the accepted EXP-007 torch.compile enabler), informative in all
outcomes. Hypothesis: cuDNN picks faster conv algos → dt < 8ms → >91 epochs → if convergence-bound, acc
clears 96.32. Honest most-likely outcome: small dt change (compile may already be conv-optimal) → modest
gain or within-noise null.

## Approach
Single-line change to `train.py`: `torch.backends.cudnn.benchmark = True` added inside `main()` right
after the device print (L156), before model construction and `torch.compile(mode="reduce-overhead")`.
No other change — model, params (4,299,866), data, optimizer, LR schedule, seed (42), and eval all
untouched, so num_epochs change is purely throughput and any acc change isolates the convergence question.
No deviations from plan-040.

## Execution
One run, no retries. Launched on idle H20 GPU 1 (`CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1`)
while a separate autoresearch instance (`v2.9.5-gpt-5-5`) ran on GPU 0 — GPU 1 stayed uncontended.
Clean run: dt = 718 steps @ 8ms, 6 @ 9ms, only warmup outliers (13/30ms). Ran to the 300s budget cut:
94 epochs / 36422 steps, total_seconds 433.1, exit 0. No errors.

## Results

- **Primary metric**: best_test_acc 95.91% (baseline: 96.22, delta: **−0.31**, −0.32%)
- **Observations**:
  - **dt UNCHANGED at 8ms** — `cudnn.benchmark` did NOT lower per-step time. This is the decisive signal.
  - **peak_vram_mb 491→971 (≈2×)** — proof that benchmark mode DID select different conv algorithms
    (higher-memory ones); they were simply no faster. The conv dt floor for these fixed shapes was
    already reached by `torch.compile(reduce-overhead)`.
  - num_epochs 94 vs baseline ~91 — within run-to-run throughput jitter, NOT a real throughput gain.
  - final_test_loss 0.2039 vs baseline 0.195 — mildly inflated; the run landed in the low side of the
    ±0.25pp augmented-base noise band.
- **Analysis**: The hypothesis's premise (cuDNN benchmark cuts dt) was FALSIFIED → outcome (c) from the
  plan: cuDNN-autotune is a throughput no-op under torch.compile here. Because dt did not drop and the
  +3 epochs are jitter, this run did NOT cleanly test convergence-bound. A weak secondary observation:
  the run reached MORE epochs (94) than baseline yet scored LOWER (95.91 vs 96.22) — but with the
  ±0.25pp base jitter this is within noise and cannot distinguish convergence-bound from epoch-saturated.
  Result fits the recent sub-baseline null cluster (036: 95.89, 037: 96.04, 038: 95.47, 039: 95.89).
- **Key Learning**: `torch.compile(reduce-overhead)` already selects throughput-optimal conv algorithms
  for this launch-bound net's fixed shapes; `cudnn.benchmark` only swaps in equal-speed higher-memory
  algos (no dt gain). The conv-dt floor is reached — cheap throughput levers can't buy epochs here.

## Verification
- **Conditions**: NECESSARY primary-metric condition FAILED (95.91 < bar 96.32, < baseline 96.22).
  Clean-completion and no-constraint-violation conditions passed (total_seconds 433.1 < 600, exit 0,
  diff = train.py only, seed 42, eval lines 94 == num_epochs 94 ≤1/epoch, num_params 4,299,866 unchanged).
- **Review Notes**: Results trustworthy — clean uncontended run (steady 8ms dt, GPU 1 isolated from the
  GPU-0 neighbor), 94 ep ≥ baseline so not contention-shortened, metric cross-consistent with the
  dt/loss/vram evidence. No integrity concern (a backend flag, not seed/data manipulation).
- **Verdict**: no-improvement
- **Verdict Basis**: valid, fair run; necessary primary-metric condition failed (metric below baseline).

## Unexplored Avenues
- **A throughput lever that actually cuts dt** is still untried — `cudnn.benchmark` failed because compile
  is already conv-optimal. Candidates that COULD move dt: `torch.compile(mode="max-autotune")` (Triton+
  cuDNN autotuning; risk: autotuning runs in the timed first steps → compile-tax-in-budget), or offloading
  the eager GPU Cutout (~6 kernels/step) to the dataloader (reliably removes work from the timed path;
  risk: reward-hacking optics + wall-clock). Both were deprioritized in brainstorm-040 for risk.
- The **convergence-bound vs epoch-saturated** question remains UNRESOLVED — it needs a working dt
  reducer to add real epochs. This experiment only closed the cuDNN-autotune sub-lever.

## Next Steps
- **Treat the cheap-throughput axis as largely closed**: cuDNN benchmark is a no-op; the conv dt floor is
  reached under compile (confidence: high for cheap flags). Only an aggressive kernel re-autotune
  (max-autotune) or moving work off the timed GPU path could cut dt, and both carry budget/optics risk.
- **Step back from the "buy epochs" framing toward accuracy-per-epoch at iso-dt**: after 41 experiments,
  the plateau is bounded by capacity (closed both ways), augmentation (closed), schedule (closed),
  regularizers (closed), optimizer (closed), and both head sub-levers (closed). The remaining genuinely
  untried micro-levers are training-dynamics details that don't add compute: e.g. **BN momentum / eps
  tuning** or **a Fixup/zero-γ-style init variant beyond EXP-026's** (confidence: low-medium — likely
  polish, but unmapped). (confidence: medium that these are the only clean remaining knobs.)
- **If pursuing dt reduction**: run a quick OFFLINE diagnostic of `max-autotune` compile time + steady dt
  (compile + time 50 steps, NOT a full budgeted run) to decide whether the compile-tax is repayable
  before committing a real experiment (confidence: medium it's worth the measurement).

## Exit Action Results
<!-- No exit actions defined in the goal file. -->
- None defined.
