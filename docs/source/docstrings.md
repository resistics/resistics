(docstring-authoring)=
# Docstring authoring

Docstrings are part of the supported Resistics interface. They should explain
the contract at the point where users and maintainers encounter the code, and
Sphinx renders them into the API reference.

(docstring-coverage)=
## Required coverage

Every production module, public class, public function, and public method must
have a useful docstring. A private helper also needs a docstring when its
behaviour is not evident from its name and signature, particularly when it has
non-obvious units, array shapes, mutation, I/O, ordering, error handling, or
framework constraints.

Pydantic validators and Textual callbacks are framework entry points rather
than public library APIs. They may omit a presence-only docstring when the
declaration and implementation already make their purpose clear. Ordinary
public methods in those modules are still documented.

During the Phase 7 migration, unconverted modules retain NumPy-style docstrings
and converted modules use MyST field lists. The opening summary should describe
the useful outcome rather than repeat the object's name. For a non-trivial
callable, document its parameters, returns or yields, and raised exceptions.
Keep types in Python annotations rather than duplicating them in field lists.
Constructors are documented on their class; do not duplicate the class contract
on `__init__`. Conventional dunder methods need prose only when their behaviour
is surprising.

(docstring-examples)=
## Examples and plots

Examples belong in docstrings when they help a user compose or understand the
API. They are expected for primary workflows such as project creation, flow,
parameter and job construction, processing, result access, and plotting. Add an
example to a lower-level operation when its units, shapes, edge cases, or
composition are otherwise difficult to infer.

Use executable `>>>` examples wherever practical. Production docstrings are
run as doctests, so examples must be deterministic and small. Examples that
require external data or interactive rendering should still show the essential
calls and clearly identify omitted setup.

Plots and related Sphinx directives stay with the function or class they
explain. Existing plot, note, warning, math, and cross-reference content is a
protected documentation asset, not text to move into a separate page. Convert
it in place to fenced `{plot}`, `{note}`, `{warning}`, `{math}`, and role syntax
while preserving its rendered behaviour.

At the Checkpoint 2.4 boundary, production docstrings contained at least 86
example sections and 14 plot directives. `tests/test_docstring_contract.py`
protects those counts and the locations of every existing plot directive;
normal doctest collection protects executable examples.

(docstring-checks)=
## Checking changes

Run the documentation checks after changing a production API:

```console
uv run --locked --no-sync ruff check resistics
uv run --locked --no-sync pydoclint --config=pyproject.toml resistics
```

Ruff enforces public docstring presence. Pydoclint enforces agreement between
the signature, annotations, and the package's MyST docstring style. The check
is baseline-free: correct a violation or use a narrow, reviewed suppression
with an explanation.
