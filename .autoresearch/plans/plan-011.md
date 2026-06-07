# Plan EXP-011: Dropout(0.3) before FC
- **Created**: 2026-05-28
## Code Changes
Add `self.dropout = nn.Dropout(0.3)` in ResNet.__init__ and apply before FC in forward.
## Verification
best_test_acc >= 95.83%
