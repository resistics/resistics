# Resistics Agent Guide

This directory is the canonical working context for coding agents contributing to
resistics. Read this file first, then follow the linked guidance for project
rules, coding standards, naming, and active migration plans.

## Current Direction

- MTH5 is the only public input format.
- New and migrated code must use Pydantic v2 APIs.
- Resistics remains a standalone Python package for scripts, notebooks, and
  batch processing.
- Core resistics also provides app-safe backend contracts for `resistics-app`,
  but must not depend on `resistics-app`, GUI frameworks, web frameworks, IPC,
  or app-specific state.
- Flow, parameter set, and processing job are separate concepts.
- Numerical complex-domain regression belongs in `regressioninc`; resistics owns
  MT-specific preparation, adapters, and solution metadata.

## Read Before Editing

- `project-rules.md`: product, migration, dependency, documentation, and test
  rules.
- `coding-standards.md`: formatting, Pydantic v2, public API, processing, MTH5,
  regression, and git hygiene standards.
- `naming.md`: canonical names for MTH5, flow/parameters/job, time windows,
  channels, regression, and filesystem values.
- `tui-design-guidelines.md`: colour, focus, keyboard-navigation, and testing
  guidance for the terminal UI.
- `plans/modernization.md`: current modernization plan for Pydantic v2,
  MTH5-only input, flows/parameters/jobs, and regression boundaries.
- `plans/codebase-hardening.md`: phased codebase cleanup, tooling,
  performance, CI, publishing, dependency, and type-checking plan.
- `plans/project-structure-and-flows.md`: design notes for project structure,
  processing flows, parameters, jobs, batching, and execution.

## Examples

Files under `examples/` are conceptual agent references for the target
flow/parameters/job model. They are not guaranteed to be executable against
the current code until the relevant migration phase is complete.
