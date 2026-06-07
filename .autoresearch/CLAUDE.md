# Autoresearch Agent Instructions

<!-- This file is the map for the autoresearch loop inside the target project.
     It is seeded as CLAUDE.md (auto-loaded by Claude Code) with AGENTS.md kept
     as a symlink to it (auto-loaded by Codex), so whichever agent is operating
     inside the target project root picks the file up automatically. Every
     phase skill reads it first to learn `Current phase:`, `Active goal:`, and
     `Active experiment:`. It must not be modified under any circumstances. -->

## Current Autoresearch State

Active goal: maximize-cifar10-test-accuracy  <!-- goal slug -->
Active experiment: 072  <!-- 3-digit ID e.g. 000, or "none" -->
Current phase: `executing`
Integration branch: autoresearch/dev  <!-- agent syncs with main each loop -->

## Routing for agents

**What is autoresearch?** A structured research loop with phases (goal → brainstorming → planning → executing → analyzing), each owned by one skill.

**If continuing autoresearch from a previous session:** Invoke `/using-autoresearch` first — it sets the session mode (copilot/autopilot, see [Autonomy](#autonomy)) and routes to the correct phase. Do not invoke phase skills directly in a new session without going through this entry point first.

| Phase | Skill | Meaning |
|-------|-------|---------|
| `goal` | `/research-goal` | Sync integration branch, set or confirm the research goal (metric, direction, constraints, verification) |
| `brainstorming` | `/research-brainstorm` | Literature review, history review, idea generation, evaluation, selection |
| `planning` | `/research-plan` | Creating concrete execution plan |
| `executing` | `/research-execute` | Implementing changes and running the experiment |
| `analyzing` | `/research-analyze` | Analyzing results, rendering verdict, routing to next step |

After analysis is complete, the loop always routes back to the goal phase: `Active experiment:` resets to `none`, `Current phase:` to `goal`. `Active goal:` is preserved across loops — the goal persists until the user explicitly changes it. Every experiment is recorded in the TSV index regardless of verdict. On `improvement`, code is committed, merged to the integration branch, and a PR to `main` is created. On `no-improvement`, `invalid`, or `crash`, code changes are discarded and the agent returns to the integration branch.

## Goal Switching

To switch autoresearch goals at any time call:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/skills/shared/scripts/update-loop-state.sh" "<PROJECT_ROOT>" --switch <slug>
```

The current goal's state is always preserved so you can always switch back later to continue it.

## Loop Discipline

These rules apply at ALL times. They are non-negotiable. A common failure mode is forgetting to re-invoke phase skills and missing key steps in the loop.

- ALWAYS invoke phase skills via the Skill tool — do not execute phases from memory. Re-read the skill every time, even if you have seen it before in the previous loop.
- ALWAYS write the required phase artifact before advancing: brainstorm-{NNN}.md, plan-{NNN}.md, exp-log-{NNN}.md, exp-report-{NNN}.md. The script rejects advancement if missing.
- ALWAYS call `update-loop-state.sh` (at `${CLAUDE_PLUGIN_ROOT}/skills/shared/scripts/update-loop-state.sh`) to advance the loop. It auto-advances, checks artifacts, and tells you the next skill. Run with no args to see usage.
- NEVER skip phases. The script enforces the order: goal → brainstorming → planning → executing → analyzing → goal. You cannot override it.

## Directory Structure

```
<project-root>/.autoresearch/
├── CLAUDE.md                          # Agent entry point / map
├── AGENTS.md                          # Symlink → CLAUDE.md (for Codex auto-load)
├── protected-files.json               # Hook-enforced file protection patterns
├── .session/
│   └── session-<session-id>.yaml      # Per-session sentinel — carries `mode: copilot|autopilot` for the current agent session
├── .goal-state/                       # Per-goal loop state (ground truth, managed by update-loop-state.sh)
│   └── {slug}.state                   # phase + experiment for one goal
├── goals/                             # Goal definitions (one per goal, slug-named)
│   └── {slug}.md                      # Goal: metric, direction, constraints, verification
├── experiment-indices/                # Per-goal experiment tracking (slug-named, matching goals/)
│   └── {slug}.tsv                     # Per-goal TSV for centralized metric tracking (managed by exp-index.sh)
├── goal-learnings/                    # Per-goal qualitative distillation (slug-named, matching goals/)
│   └── {slug}.md                      # Protocol Findings + Failed Approaches + Patterns for the slug-named goal
├── project-insights.md                # Cross-goal STRATEGIC wisdom
├── infra-errors.md                    # Cross-goal infrastructure error memory
├── brainstorm/
│   └── brainstorm-{NNN}.md            # Literature review, history review, candidate ideas, chosen idea
├── plans/
│   └── plan-{NNN}.md                  # Execution plan: milestones, code changes, environment, criteria
├── logs/
│   └── exp-log-{NNN}.md               # Execution log: decisions, job details, errors & dead ends
├── knowledge/
│   ├── README.md                      # Index of external knowledge entries
│   ├── papers/                        # Paper distillations (agent-readable)
│   └── references/                    # Reference implementation notes or miscellaneous materials (web articles etc.)
└── reports/
    └── exp-report-{NNN}.md            # Final analysis: results, verdict, key learning, next steps
```

## Key Rules

- **State updates**: Use `update-loop-state.sh` to change the 4 state fields above. Do NOT Edit or Write this file directly — it is protected by a PreToolUse hook.
- **Project context**: The project-root `CLAUDE.md` provides project orientation and is auto-loaded by Claude Code.
- **Local-only**: `.autoresearch/` is gitignored — never committed. Code changes on experiment branches are committed only on loop success.
- **Session restore**: Read this file → `Current phase:` determines which skill to invoke → `Active goal:` locates goal file and experiment index → `Active experiment:` locates experiment artifacts
- **Integration branch**: All experiment branches are cut from and merged back to the integration branch (default `autoresearch/dev`). The agent rebases it onto `main` at the start of every loop (in the goal phase). Successful experiments produce PRs to `main` as records; `gh` CLI must be available.
- **Scope enforcement**: `.autoresearch/protected-files.json` defines hook-enforced protection for out of scope files from modifications.

## Autonomy

| Mode | Behavior |
|------|----------|
| `copilot` | Agent pauses at every phase boundary via `AskUserQuestion` and uses the researcher as a resource on ambiguous calls. In the goal phase the agent presents the current goal and lets the user continue, switch to a different goal, or create a new one. |
| `autopilot` | Agent runs the loop autonomously, never stopping mid-loop. Narrow pause exception: blockers only the user can resolve (credentials, permission errors, admin-only infrastructure). In the goal phase with an active goal, the agent silently passes through goal setting to brainstorm — set-and-forget. With no active goal, autopilot prompts the user — the agent never picks a goal on its own. This is a deliberate safeguard to keep the human in the loop on direction decisions. |

Mode is scoped to the current agent session and mode reminders are injected continuously via hooks (Claude Code only) to steer the agent. The mode is immutable once set — to change it, start a new session.

## What to read, and when

| Situation | Read |
|-----------|------|
| Session start / resume | This file → `Current phase:` tells you which phase to resume, make sure session mode is set when continuing progress in a new session |
| Setting or confirming goal (phase = `goal`) | `goals/` directory + `experiment-indices/` + `goal-learnings/` + `project-insights.md` |
| Brainstorming ideas | Active goal file + `experiment-indices/{slug}.tsv` + `goal-learnings/{slug}.md` + `project-insights.md` + `knowledge/README.md` |
| Understanding experiment spec | `brainstorm/brainstorm-<NNN>.md` (where NNN = `Active experiment:` ID) |
| Reviewing experiment plan | `plans/plan-<NNN>.md` |
| Executing an experiment | `plans/plan-<NNN>.md` milestone checkboxes for progress tracking; active goal file for verification criteria; `experiment-indices/{slug}.tsv` for current baseline; `goal-learnings/{slug}.md § Failed Approaches` as a pre-execution sanity check |
| Hitting an infra error | `infra-errors.md` |
| Analyzing results | `logs/exp-log-<NNN>.md` `## Run Log` entries (metrics, WandB URLs) + `## Execution` outcome + active goal file verification + `experiment-indices/{slug}.tsv` for baseline |
