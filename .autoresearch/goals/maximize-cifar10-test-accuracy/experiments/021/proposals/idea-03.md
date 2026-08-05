# Idea-03: AdaptiveConcatPool readout head (avg+max global pool)

## Summary
Change the global-readout, the ONE structural component never touched in 20 experiments. Currently the net does `MaxPool2d(4)` over the final 4×4/512 feature map → flatten(512) → `Linear(512,10)`. Replace it with **AdaptiveConcatPool**: concatenate global average-pool and global max-pool of the 4×4/512 map → a 1024-d vector → `Linear(1024,10)` (bias-free, ×SCALE_OUT preserved). Global average pooling adds a translation-robust, all-spatial-locations summary that pure max-pool discards; concatenating both gives the linear head strictly more information (max = "is the feature present anywhere", avg = "how much / how broadly"). This is a fast.ai-popularized head trick and a near-throughput-free change (one extra global reduction + a 2× wider final Linear, negligible at 512→10).

## Reasoning
- **Untouched axis.** Every prior experiment changed convs, optimizer, schedule, aug, or normalization; the readout (global pool + head) has been the fixed `MaxPool4 → Linear` since EXP-001. A genuinely-untried axis is more likely to carry residual signal than re-probing saturated ones.
- **Information-theoretic argument.** Max-pool is a hard "any-location" OR; average-pool is a soft spatial mean. On a 4×4 map (16 locations) the two are materially different statistics; concatenating both strictly dominates either alone for a downstream linear classifier (the head can learn to ignore a channel if unhelpful). EXP-018's brainstorm explicitly flagged AdaptiveConcatPool as an untried readout lever.
- **Throughput-cheap.** No new spatial convs; the only cost is one extra global reduction and doubling the final Linear's input (512→1024 weights into 10 outputs = 5120 extra params, ~0). num_epochs effectively unchanged → no under-anneal risk.

## Sources
- fast.ai `AdaptiveConcatPool2d` (concat of AdaptiveAvgPool + AdaptiveMaxPool) — standard library head trick.
- EXP-018 brainstorm — listed AdaptiveConcatPool head as an untried architectural-readout lever (experiments/018/01-brainstorm.md).
- Global-average-pooling lineage (Network-in-Network, Lin et al. 2014) — GAP as a regularizing readout.

## Estimated Effort
Low. Replace `self.pool`/`self.fc` with a concat-pool readout and `Linear(1024,10)`; adjust `_forward_once` to `cat([avgpool(x), maxpool(x)], 1)`. Re-init the new Linear with the existing kaiming path. Smoke: output shape [N,10]; num_epochs unchanged; best==per-epoch-max; verify no NaN under bf16.

## Risk Assessment
- **Magnitude risk (primary)**: the head is a thin linear layer on an already-good 512-d feature; adding avg-pool may be redundant with what max-pool + BN already capture → likely a small or null effect on this saturated net (similar prior to the channel-attention null EXP-019, which also added a head-side recalibration).
- **Interaction with SCALE_OUT / EMA**: the wider head changes logit magnitude slightly; keep ×0.125 and verify EMA(use_buffers) still tracks the new Linear cleanly.
- **Low ceiling vs idea-01**: changes the readout, not the function class depth — less likely than compile-funded depth to break a generalization ceiling. Included as the cheap, genuinely-untried-axis bet.
