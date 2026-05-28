# Knowledge Base Index

This directory contains persistent external knowledge that informs brainstorming and planning across all research loops. Unlike experiment logs (which are loop-specific), entries here represent standing background context pulled in throughout the research process: relevant paper distillations, reference implementation notes, and domain knowledge.

## How to use
- `research-brainstorm` loads this README as a lightweight index, then loads specific entries on demand
- Entries are added by researchers (manually), by `/lit-search` (automated paper distillations), or proposed by the agent after consulting external sources
- Add entries for any paper, reference implementation, or domain insight that is likely to inform future experiments

## Venues
- `venues.md` (if it exists in this directory) defines which academic proceedings to search via `/lit-search`
- Copy the template from `${CLAUDE_PLUGIN_ROOT}/skills/lit-search/templates/venues-template.md` and edit to match your research domain
- Without a venues file, `/lit-search` defaults to top CS/ML conferences (NeurIPS, ICML, ICLR)

## Papers

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add paper summaries here)_ | | |

## References

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add reference notes here)_ | | |

## Domain Notes

| File | Topic | Key Takeaway |
|------|-------|-------------|
| _(empty — add domain-specific notes here)_ | | |
