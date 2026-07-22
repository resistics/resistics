(myst-foundation)=
# MyST foundation

This bounded page proves the mixed-parser migration path before the narrative
documentation and production docstrings move to MyST. The maintained site and
API entry pages now use MyST, while production docstrings retain explicit
per-object parser routing until their bounded conversion.
Python intersphinx resolves {py:class}`pathlib.Path`, and this explicit label
provides a stable {ref}`myst-foundation` reference.

```{eval-rst}
.. testsetup::

   from resistics.gather import GatherCriteria
```

## Production API parity

The following objects exercise a Pydantic model, its signature and annotations,
a public alias, and facade objects controlled by ``__all__``. Standard autodoc
runs inside the isolated documentation environment and retains the current API
rendering while MyST pages and converted docstrings are introduced.

```{eval-rst}
.. autoclass:: resistics.common.ProcessingProgressEvent
   :members:
   :no-index:
```

```{eval-rst}
.. autodata:: resistics.sampling.DateTimeLike
   :no-index:
```

```{eval-rst}
.. autoclass:: resistics.gather.Gather
   :members:
   :no-index:
```

```{eval-rst}
.. autoclass:: resistics.gather.GatherCriteria
   :members:
   :no-index:
```

## Representative migration object

The documentation-only target fills gaps not currently present together in one
production object. Its module-level ``__all__`` must omit
``ExcludedContract`` while retaining the public alias, inheritance hierarchy,
overloads, positional-only and keyword-only signature components, Pydantic
fields, cross-references, source links, and co-located executable directives.

```{eval-rst}
.. autodata:: prototype_api.contracts.Identifier

.. automodule:: prototype_api.contracts
   :members:
   :show-inheritance:
```

The focused view below makes inherited-member rendering explicit without
changing the global public-API policy.

```{eval-rst}
.. autoclass:: prototype_api.contracts.DerivedContract
   :members:
   :inherited-members:
   :no-index:
```
