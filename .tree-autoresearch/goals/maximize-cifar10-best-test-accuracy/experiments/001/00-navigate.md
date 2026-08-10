# Navigation - Experiment 001

## Search Intent
Establish the first modernized training direction from the unchanged ResNet-20 baseline, prioritizing a high-confidence accuracy gain within the fixed five-minute training budget.

## Chosen Base
**BASE** on `br-000` @ 91.51% - exploit

The fresh tree contains only BASE, which is the sole extendable node and has no failed children. Experiment 001 therefore extends the measured baseline at commit `7646ab4`.

## Alternatives Considered
- None - the fresh tree has no alternate successful nodes or branches.

## Policy Influence
The default policy favors the candidate with the best expected payoff after considering momentum, failed children, and unexplored directions. With only BASE available, it unambiguously selects BASE.
