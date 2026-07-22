(resistics-project)=
# `resistics.project`

(mth5-handle-ownership)=
## MTH5 handle ownership

`load()` and `open_mth5()` return objects that exclusively own one read-only
MTH5 handle. Prefer `with` for deterministic release, or call the idempotent
`close()` method explicitly. Cached summary tables remain readable after
closure; group, channel, and sample-data operations raise `RuntimeError`.

```{eval-rst}
.. automodule:: resistics.project
   :members:
   :undoc-members:
   :show-inheritance:
```
