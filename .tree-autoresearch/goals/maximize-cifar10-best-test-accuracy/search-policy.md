# Search Policy

Read during the navigating phase before every base-node choice. This file is
soft guidance — the agent weighs it by judgment. Edit it to steer the search
(e.g. "prefer forking over deep extension", "abandon a branch after 3
consecutive failures", "keep two directions alive at all times").

## Policy

Decide by your own reasoning: weigh each branch's momentum, failed-children
pileups at the tips, and unexplored directions visible in `tree.sh view` /
`tree.sh branch`, and pick the extendable node with the best expected payoff.

## Hard constraints (optional hook)

For constraints that must be enforced deterministically rather than followed
by judgment, place an executable `search-policy-hook` next to this file. It is
run as `search-policy-hook <TSV_PATH>` on every base choice: stdout lines
starting `DENY:` or `ALLOW-ONLY:` (space-separated exp_ids) are hard-enforced
— a violating choice is rejected. All other stdout is advisory text for the
agent. A hook that exits non-zero (or no hook at all) imposes no hard
constraints. To ban a whole branch, have the hook expand it to that branch's
extendable node ids (one line over the TSV's `branch` column).
