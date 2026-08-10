# Proposal 03: State-Restored TorchInductor Training (Retired at Feasibility)

## Decision

**Do not advance this seed to planning or a full experiment.** The exact candidate call fails before graph capture in the frozen local environment:

```python
compiled_model = torch.compile(
    raw_model,
    backend="inductor",
    mode="default",
    fullgraph=True,
    dynamic=False,
)
```

With Python `3.14.6` and PyTorch `2.9.1+cu128`, it immediately raises:

```text
RuntimeError: torch.compile is not supported on Python 3.14+
```

This is a decisive protocol-level no-go, not a timing-gate miss. The goal forbids changing dependencies, `pyproject.toml`, or `uv.lock`; changing the interpreter/framework would therefore be outside scope. A different backend, eager execution after compiler failure, error suppression, or a no-op wrapper would not test the proposed TorchInductor mechanism and must not be substituted. No GPU timing run is warranted.

The remainder pins the intended candidate and its gates so the retirement is technically auditable rather than an underspecified rejection.

## Mechanism and Evidence

The accepted EXP-010 run processes 26,898 batch-128 updates in 300 counted seconds and reaches 94.15%. EXP-013's warm H20 decomposition attributes 2.408 ms/step to model forward and 8.220 ms/step to model backward, together 10.628 ms or 97.57% of the measured 10.893 ms CUDA-stage sum. By contrast, the synchronized wall step is only 0.034 ms above the summed CUDA stages. Loader wait is 0.145 ms median, the optimizer stage is under 0.8%, and peak allocation is 598.7 MB on a 97,871 MiB H20.

The only defensible compilation mechanism is therefore lower GPU forward/backward execution time through generated/fused kernels. PyTorch 2.9 documents TorchInductor as the default GPU compiler, AOTAutograd as capturing backward, and fusion as Inductor's principal optimization. CUDA-graph or Python-launch savings alone cannot materially affect this workload because the locally measured host gap is only 0.034 ms. Sources: [compiler architecture](https://docs.pytorch.org/docs/2.9/torch.compiler.html), [fusion overview](https://docs.pytorch.org/docs/2.9/torch.compiler_get_started.html), and [`torch.compile` API](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html).

If supported, compilation would be nominally training-math preserving. Its accuracy hypothesis is indirect: exact restoration makes the accepted seed-42 initial state and data trajectory the starting point, while faster counted steps yield more optimizer updates and images before the same elapsed-time LR boundaries. A candidate retaining the same 94.15%-quality dynamics while raising exposure by at least 3% would process at least 27,705 steps, about 70.9 rather than 68.9 dataset passes. The pre-registered metric hypothesis would be `best_test_acc >= 94.25%`, with no claim that compilation itself regularizes the model.

## Exact Candidate Scope

- Preserve the complete accepted EXP-010 recipe: width-2 postactivation ResNet-20, batch 128, all-parameter SGD decay `1e-4`, momentum `0.9`, p=0.5 alpha-1 CutMix on N1/M7 views through the 80% boundary, then the hard weak tail, seed 42, evaluator, checkpoints, and 300-second counted timer.
- Compile **only the top-level training model forward** after constructing `raw_model` and its SGD optimizer. Training calls `compiled_model(inputs)`; eager cross-entropy, `loss.backward()`, LR assignment, zero-grad, and `optimizer.step()` remain outside the compiled Python frame. AOTAutograd may still generate the backward for the captured model graph, while the negligible loss and optimizer stages do not introduce changing-LR guards.
- Pin `backend="inductor"`, `mode="default"`, `fullgraph=True`, and `dynamic=False`. Batch size and training tensor shapes are fixed, so static specialization is appropriate. `fullgraph=True` turns a model graph break into a hard failure. `default` tests balanced Inductor fusion; `reduce-overhead` is rejected because it principally targets small-batch CUDA-graph/Python overhead, and `max-autotune` is a distinct higher-startup search policy rather than an interchangeable rescue.
- Never enable `torch._dynamo.config.suppress_errors`. Compiler failure is candidate failure. Never retry through eager, another backend, a reduced submodule scope, or a different mode.
- Keep `raw_model` as the optimizer's parameter owner and as the evaluator argument. The compiled callable shares those parameters for training, while `evaluator.evaluate(raw_model, device)` preserves the exact eager fixed-evaluation path and avoids a second eval-shape/inference compilation. Each epoch still calls `raw_model.train()` before the next compiled training invocation because `Eval.evaluate` leaves it in eval mode.

## Lazy Compilation and Exact Restoration

Compilation is lazy, and PyTorch warns that the first one or two invocations can be much slower. If the environment supported the exact call, perform three fixed-shape synthetic training steps before `t_start_training`; the first materializes forward/AOT backward code and the next two prove stable cache reuse. Compilation and this warmup remain excluded from the 300-second training counter but included in total wall time.

Immediately before creating synthetic inputs, snapshot all mutable experiment state:

1. Clone every tensor in `raw_model.state_dict()` without changing device. This includes all weights, biases, BatchNorm affine tensors, `running_mean`, `running_var`, and `num_batches_tracked`.
2. Deep-copy `optimizer.state_dict()`, including parameter-group LR/decay/momentum and the initially empty momentum state.
3. Save `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()`, and Python `random.getstate()`. Generate synthetic batch-128 FP32 inputs and integer targets only after these snapshots.
4. Record that every parameter gradient is initially `None`.

After the third synchronized warmup step, restore `raw_model.load_state_dict(model_snapshot)`, restore `optimizer.load_state_dict(optimizer_snapshot)`, set every parameter gradient to `None` with `optimizer.zero_grad(set_to_none=True)`, restore Python/CPU/all-CUDA RNG states, delete synthetic tensors and loss/output references, and synchronize once. Do not reconstruct the model: the compiled callable and optimizer must retain the same parameter objects.

Before admitting timing, prove exact restoration with `torch.equal` for every model-state tensor, exact equality of optimizer parameter groups and state keys/tensors, equality of Python/CPU/CUDA RNG states, all gradients `None`, and the original training mode. In particular, a reset of weights without BN buffers is invalid: three warmup forwards mutate all BN running statistics and counters. A reset without optimizer state is also invalid because three SGD steps create momentum buffers.

DataLoader workers must not be started or iterated by warmup. This prevents synthetic compilation from consuming the shuffled sampler, CutMix gate, RandAugment, or worker RNG trajectory. `t_start_training` is set only after all restoration assertions pass and immediately before the unchanged real training loop.

## Compiler Integrity Diagnostics

Run the correctness preflight in a fresh process with `TORCH_LOGS="graph_breaks,recompiles,perf_hints"` and save only the compact diagnostics in the experiment artifact. Require:

- the exact compile call and all three warmup steps exit successfully;
- zero graph-break diagnostics (also enforced structurally by `fullgraph=True`);
- zero recompilation diagnostics on warmup calls two and three;
- no cache-limit, eager-fallback, skipped-function, or CUDA-graph failure presented as the source of a speed claim;
- a single static batch-128 training specialization, with no new compilation after real training begins;
- finite eager/compiled loss and gradients on cloned, state-aligned models, with FP32 output/loss/gradient agreement under declared tight tolerances before the exact-state restoration test.

PyTorch documents guards and `TORCH_LOGS="recompiles"` for detecting guard failures, while `fullgraph=True` is the prescribed graph-break detector. Sources: [recompilation guide](https://docs.pytorch.org/docs/2.9/compile/programming_model.recompilation.html), [graph-break guide](https://docs.pytorch.org/docs/2.9/compile/programming_model.graph_breaks_index.html), and [troubleshooting](https://docs.pytorch.org/docs/2.9/torch.compiler_troubleshooting.html).

## Paired H20 Feasibility Gate

Had the CPU-side call succeeded, the candidate would still need a paired one-H20 gate before a full run. Use five fresh-process, alternating-order control/candidate pairs with identical batch-128 synthetic tensors and state-aligned models. After compilation/restoration and a fixed warm stabilization, collect at least 500 steps per arm. Use CUDA events separately around transfer, forward, loss, backward, and optimizer, plus synchronized wall timing around the full step. Report each pair and the median paired ratios.

Advance only if all conditions hold:

- compiled `(forward + backward)` CUDA-event time is at most `0.950x` eager in at least four of five pairs and in the aggregate paired median;
- compiled total CUDA-stage time and synchronized full-step wall time are each at most `0.970x` eager in the paired median;
- projected fixed-budget exposure is at least 27,705 steps (`1.03x` EXP-010's 26,898), not merely a lower Python dispatch measurement;
- pairwise candidate forward+backward ratios have coefficient of variation at most 1.5%, no non-finite values occur, and peak VRAM remains below the H20 limit;
- cold-process startup through successful restoration plus the accepted 300-second training and observed evaluation overhead projects below 540 seconds, leaving 60 seconds before the absolute 600-second kill threshold.

The forward/backward threshold is primary. A candidate that passes only wall timing, CUDA-graph, or host-overhead metrics fails, even if projected steps look favorable. Conversely, compile time is excluded from counted training exactly as the harness defines, but it is not free under the wall limit.

## Failure Mechanisms

- **Observed hard incompatibility:** PyTorch 2.9.1 rejects Python 3.14 before TorchDynamo capture. This alone retires the candidate under frozen dependencies.
- **Opaque convolution ceiling:** cuDNN may already dominate the 19-convolution forward/backward; Inductor can fuse surrounding pointwise/BN work yet fail the 5% GPU-stage gate or regress kernel choice.
- **Misattributed launch savings:** CUDA graphs can reduce Python launch overhead, but the measured 0.034 ms gap provides essentially no local ceiling. A wall-only gain is insufficient evidence.
- **Warmup contamination:** BN running buffers, `num_batches_tracked`, SGD momentum, gradients, or RNG left after lazy compile would change optimization and data augmentation, confounding both accuracy and exposure.
- **Guard churn:** training/eval mode, input shapes, or scalar state can trigger recompilation. Keeping evaluation eager and optimizer/LR logic outside scope narrows this risk, but diagnostics must prove it absent.
- **Compiler numerics:** different fusion/reduction order or selected kernels can perturb the fixed-seed trajectory. The full run would measure the net method, but only after tight local parity and exact restoration.
- **Cold-start wall failure:** compilation can take seconds to minutes and is outside the training timer but inside the 600-second total cap.
- **Evaluation substitution:** passing the compiled wrapper to `Eval.evaluate` could compile a separate batch-256 inference graph and alter the fixed evaluation path. The exact design forbids it.

## Recommendation

Retire state-restored TorchInductor training from EXP-013. The idea has a relevant measured target and a defensible experiment design, but no legal implementation exists in this repository's Python 3.14.6 / PyTorch 2.9.1 environment. Treating eager execution, another backend, dependency changes, or suppressed compiler errors as a fallback would violate both the mechanism and the hard constraints. Reconsider only in a future goal whose fixed environment natively supports the exact compile call.
