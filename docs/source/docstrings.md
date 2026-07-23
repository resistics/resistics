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

Production docstrings use MyST field lists. The opening summary should describe
the useful outcome rather than repeat the object's name. For a non-trivial
callable, document its parameters, returns or yields, and raised exceptions.
Keep types in Python annotations rather than duplicating them in field lists.
Constructors are documented on their class; do not duplicate the class contract
on `__init__`. Conventional dunder methods need prose only when their behaviour
is surprising.

(docstring-content-checklist)=
## Content checklist

For each supported object, include the parts that help a reader use it
correctly:

- a one-sentence outcome and the object's domain meaning;
- parameters and their semantic constraints;
- returns or yields, including important units, array shapes, and ordering;
- mutations, file or network I/O, resource ownership, and other side effects;
- invariants, default behaviour, and meaningful edge cases;
- exceptions a caller is expected to handle;
- links to related Resistics or external Python objects; and
- a small executable example when composition is not obvious.

Do not add placeholder sections or restate annotations in prose. A short
contract is sufficient for a simple object; a processing operation should say
enough to distinguish scientific meaning from implementation detail.

(private-helper-docstrings)=
## Private helpers

An underscore makes an object private, but does not make its behaviour
self-explanatory. Document a private helper when it represents a substantial
operation or has non-obvious units, shapes, mutation, I/O, ordering,
concurrency, error recovery, or framework constraints. A small local
transformation whose name, signature, and immediate context provide the full
contract does not need a presence-only docstring.

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

Rich examples and plots belong with their documented objects when that is where
users will find them most useful. A guide may connect several APIs into a
workflow, but it should link to rather than replace the co-located contract.

(fenced-docstring-directives)=
## Fenced directives in docstrings

Use MyST fenced directives rather than reStructuredText `.. name::` syntax.
The common forms are:

````markdown
```{doctest}
>>> 2 + 2
4
```

```{plot}
:include-source: false
:filename-prefix: stable-semantic-name

from matplotlib import pyplot as plt
plt.plot([0, 1], [0, 1])
```

```{note}
Explain a useful qualification.
```

```{warning}
Explain a real hazard and how to avoid it.
```

```{math}
y = mx + c
```
````

Use semantic, stable plot filename prefixes because the documentation gate
checks the generated artifacts. Use MyST roles such as `{class}` and `{meth}`
for Python objects and `{ref}` for labelled documentation sections. Every
retained `{doctest}` and `{plot}` block must execute in the documentation gate.

At the Checkpoint 2.4 boundary, production docstrings contained at least 86
example sections and 14 plot directives. `tests/test_docstring_contract.py`
protects those counts and the locations of every existing plot directive. The
Sphinx doctest builder executes fenced examples, and the strict HTML build
executes and verifies the plot artifacts.

(docstring-checks)=
## Checking changes

Run the documentation checks after changing a production API:

```console
uv run --locked --no-sync ruff check resistics
uv run --locked --no-sync pydoclint --config=pyproject.toml resistics
uv run --locked --no-sync python scripts/check_documentation.py
```

Ruff enforces public docstring presence. Pydoclint enforces agreement between
the signature, annotations, and the package's MyST docstring style. The check
is baseline-free: correct a violation or use a narrow, reviewed suppression
with an explanation. Follow the {ref}`justified suppression process
<justified-suppressions>`; public API findings must never be hidden in a
baseline.
