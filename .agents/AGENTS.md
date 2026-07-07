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
- Flow, processing configuration, and processing run are separate concepts.
- Numerical complex-domain regression belongs in `regressioninc`; resistics owns
  MT-specific preparation, adapters, and solution metadata.

## Read Before Editing

- `project-rules.md`: product, migration, dependency, documentation, and test
  rules.
- `coding-standards.md`: formatting, Pydantic v2, public API, processing, MTH5,
  regression, and git hygiene standards.
- `naming.md`: canonical names for MTH5, flow/configuration/run, time windows,
  channels, regression, and filesystem values.
- `plans/modernization.md`: current modernization plan for Pydantic v2,
  MTH5-only input, flows/configurations/runs, and regression boundaries.
- `plans/project-structure-and-flows.md`: design notes for project structure,
  processing flows, configurations, runs, batching, and execution.

## Examples

Files under `examples/` are conceptual agent references for the target
flow/configuration/run model. They are not guaranteed to be executable against
the current code until the relevant migration phase is complete.
