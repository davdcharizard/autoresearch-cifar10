# Tree Autoresearch Agent Instructions

<!-- This file is the map for the tree-autoresearch loop inside the target
     project. It is seeded as CLAUDE.md (auto-loaded by Claude Code) with
     AGENTS.md kept as a symlink to it (auto-loaded by Codex), so whichever
     agent is operating inside the target project root picks the file up
     automatically. Every phase skill reads it first to learn `Current phase:`,
     `Active goal:`, `Active experiment:`, `Current branch:`, and `Base node:`.
     It must not be modified under any circumstances. -->

## Current Autoresearch State

Active goal: maximize-cifar10-best-test-accuracy  <!-- goal slug -->
Active experiment: 023  <!-- 3-digit ID e.g. 001, or "none" -->
Current phase: `brainstorming`
Current branch: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-023
Base node: 001

## Routing for agents

**What is tree autoresearch?** A research loop whose experiments form a **tree**: every experiment is a node grown from a chosen **base node** (the root `BASE` or any past node with verdict `improvement`). Successes extend a branch or fork a new one; failures stay as terminal reference leaves. The phases are goal → navigating → brainstorming → planning → executing → analyzing, each owned by one skill.

**If continuing autoresearch from a previous session:** Invoke `/using-autoresearch` first — it sets the session mode (copilot/autopilot, see [Autonomy](#autonomy)) and routes to the correct phase. Do not invoke phase skills directly in a new session without going through this entry point first.

| Phase | Skill | Meaning |
|-------|-------|---------|
| `goal` | `/research-goal` | Set or confirm the research goal (metric, direction, constraints, verification) |
| `navigating` | `/research-navigate` | Inspect the tree, apply the search policy, choose the base node (`tree.sh base`) |
| `brainstorming` | `/research-brainstorm` | Lineage review, literature review, idea generation, evaluation, selection |
| `sweeping` | `/research-sweep` | Compressed parameter-sweep experiment (brainstorm routes here via the `--sweep` advance flag) |
| `planning` | `/research-plan` | Creating concrete execution plan |
| `executing` | `/research-execute` | Implementing changes and running the experiment |
| `analyzing` | `/research-analyze` | Analyzing results vs the parent node, recording the node via `tree.sh insert`, routing onward |

After analysis is complete, the loop always routes back to the goal phase: `Active experiment:` resets to `none`, `Current phase:` to `goal`. `Active goal:` is preserved across loops — the goal persists until the user explicitly changes it. Every experiment is recorded in the tree TSV regardless of verdict, and every experiment's code is committed on its own permanent branch.

## The Tree Manager (tree.sh)

`${CLAUDE_PLUGIN_ROOT}/skills/shared/scripts/tree.sh` is the sole writer of the tree store (`goals/{slug}/04-results.tsv`) and the sole performer of `tree-autoresearch/*` git operations. NEVER create, move, merge, rebase, or delete `tree-autoresearch/*` branches by hand, and never edit the TSV directly.

Queries (read-only, use freely): `view` (whole-tree overview), `branch` (branch list + tips), `show <id>` (node card), `children <id>`, `log <id>` (lineage history). Mutations (phase-bound): `base` (navigating), `insert` (analyzing).

## Git Model

- The base branch `tree-autoresearch/{slug}-base` is created once from local `main` at goal creation and never moved by the loop. **No syncing with main, ever** — no fetch, rebase, or merge for the tree's lifetime; commits are node anchors and must stay stable. To re-root on a newer `main`, delete `.tree-autoresearch/` and start fresh.
- Every experiment runs on its own permanent branch `tree-autoresearch/{slug}-exp-{NNN}`, created at the base node's commit by `tree.sh base`. It is never deleted — every node's code stays inspectable forever.
- Tree branches (`br-000`, `br-001`, …) are tree-store labels, not git branches; each branch's head experiment is reported by `tree.sh branch`. On success nothing moves in git — the exp branch IS the new tip; `tree.sh insert` just asserts the commit descends from the base node and records the row. On failure, nothing is integrated either. PRs happen only if the goal's exit actions ask for them.

## Goal Switching

To switch autoresearch goals at any time call:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/skills/shared/scripts/update-loop-state.sh" "<PROJECT_ROOT>" --switch <slug>
```

The current goal's state is always preserved so you can always switch back later to continue it. After switching, check out the branch named in the RESUME output.

## Loop Discipline

These rules apply at ALL times. They are non-negotiable. A common failure mode is forgetting to re-invoke phase skills and missing key steps in the loop.

- ALWAYS invoke phase skills via the Skill tool — do not execute phases from memory. Re-read the skill every time, even if you have seen it before in the previous loop.
- ALWAYS write the required phase artifact in `experiments/{NNN}/` before advancing e.g. `00-navigate.md`, `01-brainstorm.md`, `02-plan.md`, `03-execute.md`, `04-analysis.md`. The script rejects advancement if required artifacts are missing.
- ALWAYS call `update-loop-state.sh` (at `${CLAUDE_PLUGIN_ROOT}/skills/shared/scripts/update-loop-state.sh`) to advance the loop. It auto-advances, checks artifacts, and tells you the next skill. Run with no args to see usage. Trust the script to determine the next state.
- NEVER skip phases. The script enforces the order: goal → navigating → brainstorming → planning → executing → analyzing → goal. For a parameter sweep the flow goes from brainstorming → sweeping → analyzing. You cannot override it.

## Directory Structure

```
<project-root>/.tree-autoresearch/
├── CLAUDE.md                          # Agent entry point / map
├── AGENTS.md                          # Symlink → CLAUDE.md (for Codex auto-load)
├── protected-files.json               # Hook-enforced file protection patterns
├── bin/                               # Self-contained user-runnable scripts (dashboard)
├── .session/
│   └── session-<session-id>.yaml      # Per-session sentinel — carries `mode: copilot|autopilot`
├── .prompts/
│   └── history-<session-id>.jsonl     # Per-session user-prompt observability log (hook-appended)
├── .goal-state/                       # Per-goal loop state (script-owned ground truth)
│   ├── {slug}.state                   # phase + experiment (written ONLY by update-loop-state.sh)
│   └── {slug}.base                    # pending base: base node, current git branch, experiment (written ONLY by tree.sh)
├── goals/                             # Goal-major: one self-contained directory per goal (slug-named)
│   └── {slug}/
│       ├── 01-definition.md           # Goal: metric, direction, frozen protocol, constraints, verification
│       ├── 02-system-understanding.md # Measured system breakdown + problem bottleneck
│       ├── 03-experiment-learnings.md # Base-INDEPENDENT lessons only (lineage-local memory lives in the tree)
│       ├── 04-results.tsv             # Canonical tree store (written ONLY by tree.sh)
│       ├── search-policy.md           # π(T) soft guidance for base-node choice (user-editable)
│       ├── search-policy-hook         # Optional executable: hard DENY/ALLOW-ONLY constraints
│       ├── knowledge/                 # Per-goal external knowledge (README index, venues.md, papers/, references/)
│       └── experiments/               # Per-goal experiments, numbered from 001 (resets per goal)
│           └── {NNN}/
│               ├── 00-navigate.md     # Navigation record: chosen base + reasoning, alternatives, policy influence
│               ├── 01-brainstorm.md   # Lineage review, diagnosis, candidate ideas, chosen idea (or ## Sweep Plan)
│               ├── 02-plan.md         # Execution plan: milestones, code changes, environment, criteria
│               ├── 02-sweep.md        # Sweep path only: trial table + confirmed winner (replaces 02-plan.md + 03-execute.md)
│               ├── 03-execute.md      # Execution log: decisions, job details, errors & dead ends
│               └── 04-analysis.md     # Final analysis: results, verdict vs parent, key learning, next steps
└── project-notes/
    ├── project-insights.md            # Cross-goal STRATEGIC wisdom
    └── infra-errors.md                # Cross-goal infrastructure error memory
```

## Key Rules

- **State updates**: Use `update-loop-state.sh` to change the state fields above. Do NOT Edit or Write this file directly — it is protected by a PreToolUse hook.
- **Project context**: The project-root `CLAUDE.md` provides project orientation and is auto-loaded by Claude Code.
- **Local-only**: `.tree-autoresearch/` is gitignored — never committed. Code changes are committed on the experiment's own branch during analysis (every verdict).
- **Session restore**: Read this file → `Current phase:` determines which skill to invoke → `Active goal:` locates the goal directory (`goals/{slug}/`) and its tree store → `Active experiment:` locates experiment artifacts under `goals/{slug}/experiments/{NNN}/` → `Current branch:` is where the working tree should sit.
- **Frozen protocol**: A goal's measurement process and constraints never change mid-tree. Changing the eval methodology means a new goal and a fresh tree.
- **Scope enforcement**: `.tree-autoresearch/protected-files.json` defines hook-enforced protection for out of scope files from modifications.

## Autonomy

| Mode | Behavior |
|------|----------|
| `copilot` | Agent pauses at every phase boundary via `AskUserQuestion` and uses the researcher as a resource on ambiguous calls. In the goal phase the agent presents the current goal and lets the user continue, switch, or create a new one; in the navigating phase the agent recommends a base node and the user confirms. |
| `autopilot` | Agent runs the loop autonomously, never stopping mid-loop. Narrow pause exception: blockers only the user can resolve (credentials, permission errors, admin-only infrastructure). Navigation decides silently within the search policy's constraints. With no active goal, autopilot prompts the user — the agent never picks a goal on its own. |

Mode is scoped to the current agent session and mode reminders are injected continuously via hooks (Claude Code only) to steer the agent. The mode is immutable once set — to change it, start a new session.

## What to read, and when

| Situation | Read |
|-----------|------|
| Session start / resume | This file → `Current phase:` tells you which phase to resume, make sure session mode is set when continuing progress in a new session |
| Understanding goal progress | `goals/{slug}/01-definition.md` + `tree.sh view` on `goals/{slug}/04-results.tsv` |
| Understanding experiment status | `goals/{slug}/experiments/<NNN>/` (where NNN = `Active experiment:` ID) |
| Choosing where to grow next | `tree.sh view` / `branch` + `goals/{slug}/search-policy.md` (the navigating phase owns this) |
