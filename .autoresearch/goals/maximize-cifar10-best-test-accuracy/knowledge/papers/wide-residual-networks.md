# Wide Residual Networks
- **Authors**: Sergey Zagoruyko, Nikos Komodakis
- **Venue**: BMVC 2016
- **URL**: https://arxiv.org/abs/1605.07146

## Key Contributions

- Shows that increasing residual-network width can be more accuracy- and compute-efficient than adding extreme sequential depth on CIFAR.
- Defines widening as scaling feature-plane counts and studies the resulting quadratic parameter/compute growth.
- Finds strong results with shallow two-convolution residual blocks, while many reported recipes also use preactivation, dropout, stronger weight decay, and longer schedules.

## Relevance

The accepted model has only 269,722 parameters and uses about 330 MB on an H20, so width is a credible untested representation lever. Published headline gains cannot be transferred directly: this repository uses post-activation ResNet-20, a 300-second horizon, Option-A shortcuts, and a distinct augmentation/schedule composition. Width must be judged as a net capacity-versus-update-exposure trade under the fixed timer.

## Local Feasibility

An EXP-007 synthetic synchronized-step diagnostic measured width 16/24/32 at 7.515/9.251/10.852 ms for batch 128. Width 2 has exactly 1,073,962 parameters and projects roughly 26.6k full-run steps after calibrating to EXP-004, versus 38,358 accepted steps.
