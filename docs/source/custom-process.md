(custom-processes)=
# Custom processes

Custom processing is based on normal importable Python classes. A process
inherits from `ResisticsProcess`, declares its input and output data contracts,
stores configuration as Pydantic fields, and implements either `run()` or
`execute()`.

(custom-process-contract)=
## Process contract

The flow system uses four class-level declarations:

- `input_types` maps named input ports to their logical data types.
- `output_type` names the value produced by the process.
- `runtime_requirements` lists context supplied by the project or job runner.
- `include_in_default_parameters` controls whether templates include the
  process automatically.

For a numerical transform whose inputs map directly to `run()` arguments, the
base `execute()` implementation forwards those inputs. Readers, writers, and
project-aware operations can override `execute(inputs, context)` instead.

```python
from typing import Any, ClassVar

from resistics.common import ResisticsProcess


class AddOffset(ResisticsProcess):
    input_types: ClassVar[dict[str, str]] = {"value": "number"}
    output_type: ClassVar[str] = "number"

    offset: float = 0.0

    def execute(self, inputs: dict[str, Any], context: Any) -> float:
        return float(inputs["value"]) + self.offset
```

(custom-process-flow)=
## Use a process in a flow

Flow nodes identify process classes by qualified import path. Their input map
connects a named port to an earlier node identifier; topological validation
rejects missing producers, cycles, incompatible data types, and unsatisfied
runtime requirements before execution.

```python
from resistics.flow import FlowNode, ParameterSet

process = "my_project.processes.AddOffset"
node = FlowNode(
    id="offset",
    process=process,
    inputs={"value": "source"},
)
parameters = ParameterSet(
    name="custom",
    processes={process: {"offset": 1.5}},
)
```

Here `source` is the identifier of an earlier node that produces the required
`number` value. Add both nodes to a stage before validating the complete flow.

(custom-process-plugins)=
## Project plugins

Keep custom modules below the canonical `project/plugins/` package and use
qualified `plugins.<module>.<class>` paths that remain valid when YAML is loaded
later. Treat plugin code as executable project code rather than untrusted
configuration. The same descriptor and flow validation used for built-in
processes applies to plugin classes. External plugin directories are not part
of the supported project contract.
