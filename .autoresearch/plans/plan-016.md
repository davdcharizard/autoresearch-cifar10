# Plan EXP-016: TTA (horizontal flip) in EMA model
## Code Changes
- Add `_forward_features` method to ResNet that does the core computation
- Override `forward` to call `_forward_features` twice when not training (original + hflip), average logits
- Training forward is unchanged (no TTA during training)
## Verification
best_test_acc >= 95.83%
