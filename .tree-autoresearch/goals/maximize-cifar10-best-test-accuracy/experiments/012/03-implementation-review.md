# Claude Adversarial Implementation Review: EXP-012

- **Reviewer**: Claude Code 2.1.220
- **Mode**: read-only adversarial review of the `train.py` diff against `d68f73a`
- **Verdict**: PASS

Claude found no correctness, RNG, layout, audit, or performance defect warranting a block.

## Findings

- **Complement semantics**: The Cutout call is the lexical `else` of the CutMix draw inside `progress < CUTMIX_END`, so every eligible non-CutMix batch receives Cutout and CutMix batches remain untouched. The parent CutMix CPU/CUDA streams and patch sequence are unchanged.
- **RNG**: The device-local seed-43 generator is consumed only by uniform center draws in `[0,32)`, without advancing global or CutMix streams.
- **Geometry and fill**: The bank implements clipped half-open `[center-8,center+8)` masks. Exact invariants of 1,024 unique masks, min/mean/max area 64/196/256, and the `(1,1,1)` normalization guard correctly establish normalized-zero dataset-mean fill.
- **Dtype and layout**: Mask bank and selected buffer are FP32 channels-last, inputs are asserted FP32 channels-last, and the in-place multiply preserves input layout.
- **Charging**: All RNG, selection, multiplication, histogram, and area-accounting work occurs before the existing per-step synchronization. No per-batch host synchronization was introduced.
- **Audit strength**: Complement equality, bank invariants, realized mean area, and complete center support detect skipped/double application and dose corruption; errors feed the fatal integrity path.
- **Parent preservation**: Model, optimizer, SAM, EMA logic, and evaluation cadence are unchanged. Evaluation output only adds charged time/progress fields.

## Non-blocking Observations

- The full-batch shape assertion depends on the frozen loader's `drop_last=True` and therefore fails loudly if that contract changes.
- The image-count equality is tautological, but adjacent complement and dose checks provide real detection power.

## Final Review After GPU Correction

- **Verdict**: PASS
- **Trigger**: Re-review after changing the allocation-stable scalar reduction to the environment-supported `torch.sum(selected_areas, dim=(0,), out=batch_masked_pixels)` overload.
- **Claude conclusion**: The scalar result shape matches the preallocated zero-dimensional output; FP32 per-batch sums and FP64 cumulative sums remain integer-exact at this dose. Claude reconfirmed geometry, private RNG isolation, lexical complement/SAM separation, charged-time placement, terminal audit behavior, channels-last layout, and EXP-011 preservation, and explicitly cleared the candidate for the metric run.
