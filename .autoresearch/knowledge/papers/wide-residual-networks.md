# Wide Residual Networks

- Source: https://arxiv.org/abs/1605.07146
- Authors: Sergey Zagoruyko, Nikos Komodakis
- Relevance: CIFAR residual-network architecture scaling.

## Key Takeaway

Widening residual networks can improve CIFAR accuracy and efficiency compared with simply increasing depth. In this repo, WRN-style changes are promising after cheap recipe upgrades, but they need careful runtime checks under the fixed 300 second budget.

## Use In This Project

Consider compact WRN variants such as 16 or 22 layers with width factor 2 in later experiments if training-recipe changes plateau.
