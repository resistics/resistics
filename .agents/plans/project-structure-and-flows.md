# Design: Project Structure and Processing Flows

## Objective
To decouple processing flows (sequences of steps) from their parameters, enabling users to reuse flows with different parameter sets and ensuring high reproducibility by storing these within the project structure.

## Core Concepts

### 1. Processing Flow
A **Flow** defines the sequence and connectivity of processing steps. It specifies *what* steps are run (e.g., Read -> Calibrate -> FFT -> TF) but does not contain the specific settings for those steps.
- **Format**: YAML or JSON.
- **Logic**: A Directed Acyclic Graph (DAG) of process identifiers.

### 2. Parameter Set
A **Parameter Set** defines the specific values and settings for one or more processing steps. These are often "static" across multiple jobs.
- **Examples**: Window sizes, window types, decimation factors, frequency bands.
- **Format**: YAML or JSON.

### 3. Processing Job
A **Processing Job** binds a Flow with a Parameter Set and specific **Runtime Arguments** to create an executable processing environment.

### 4. Runtime Arguments
These are parameters that change with almost every job and define the *scope* of the processing.
- **Examples**: Primary station, Remote reference station, Time range (Start/End), input data sampling frequency.

---

## Processing Jobs: Benefits and Downsides

| Feature | Description |
| :--- | :--- |
| **Benefits** | |
| **Reproducibility** | Each processing job is a self-contained "recipe". Storing jobs in the project ensures anyone can recreate the results with 100% fidelity. |
| **Batch Processing** | High-level jobs allow users to queue the same flow and parameters on multiple stations efficiently. |
| **Traceability** | Results are linked to a specific job, making it clear which parameters produced which transfer function. |
| **Downsides** | |
| **Abstraction Overhead** | Introducing Flows, Parameter Sets, and Processing Jobs adds learning complexity for new users. |
| **Management** | Over time, projects can accumulate many configuration files, requiring good organizational tools (e.g., the standard folder structure). |

---

## Incorporating Runtime Arguments

To handle the distinction between static parameters and runtime arguments, the **Processing Job** will be the central point of integration:

```yaml
# Example Processing Job (processing/jobs/siteA_robust.yaml)
name: Site A Robust Processing
flow: standard_mt.yaml
parameters: robust_low_freq.yaml
runtime_args:
  station: StationA
  remote: StationB
  start_time: "2024-01-01 00:00:00"
  end_time: "2024-01-01 12:00:00"
```

### Execution Logic:
1. **Selection**: User selects a Flow and a Parameter Set in the App.
2. **Dynamic Inputs**: The App inspects the Flow and asks for the required Runtime Arguments (Stations, Timeframes).
3. **Binding**: These are combined into a temporary or saved **Processing Job**.
4. **Archival**: When the job executes, its definition is written into the `results/[output_label]/` folder as `job_info.json`.

---

## Batch Processing and Multi-site Jobs

To handle multiple stations efficiently, the system supports **Batch Jobs**. This can be achieved in two ways:

### 1. List-based Runtime Arguments
A single Processing Job can specify a list of values for a runtime argument.
```yaml
# processing/jobs/multi_site_mt.yaml
name: Robust Multi-site
flow: standard_mt.yaml
parameters: default.yaml
runtime_args:
  station: ["Station01", "Station02", "Station03"]
  remote: "Remote01"
```
The **Execution Engine** will automatically expand this into N individual job executions.

### 2. Batch Groups
For more complex scenarios where sites have different remotes or time ranges, a **Batch Configuration** can be used:
```yaml
# processing/jobs/campaign_2024.yaml
name: Full Campaign
flow: standard_mt.yaml
parameters: robust.yaml
batch:
  - { station: StationA, remote: StationB, start_time: "..." }
  - { station: StationC, remote: StationD, start_time: "..." }
```

### Result Organization:
Results for batch jobs will be grouped by the output label to keep the `results/` folder clean:
`results/[output_label]/[station_name]/...`

---

## Proposed Project Structure

Based on `resistics/resp.py`, the project structure will be organized as follows:

```text
project/
├── resistics.json          # Project metadata (MTH5 path, reference time, plugin paths, etc.)
├── processing/
│   ├── flows/              # Flow definitions
│   │   ├── standard_mt.yaml
│   │   └── quick_check.yaml
│   ├── parameters/         # Processing parameter sets
│   │   ├── default_parameters.yaml
│   │   └── high_sampling_parameters.yaml
│   └── jobs/               # Pre-defined bindings of Flow + Parameters
│       └── final_processing.yaml
├── data/
│   └── [survey]/[station]/
│       ├── [run]/          # MTH5 run-specific derived data
│       └── results/        # Processing outputs (spectra, transfer functions)
│           └── [output_label]/
│               ├── solution.json
│               └── job_info.json # Copy of the Processing Job used
└── logs/
    └── [job_name].log
```

---

## Integration Details

### resistics library
- **`Flow` class**: A new class to handle the DAG logic, likely utilizing `networkx` (as seen in `resistics-app`).
- **`ParameterSet` class**: A Pydantic model that holds step-specific settings.
- **`ProcessingJob` class**: A model that links a Flow and a ParameterSet with Runtime Arguments.
- **`ExecutionEngine`**: A class to take a `ProcessingJob` and execute it on a `Project`.

---

## GUI Strategy: Simplifying Configuration

To make resistics-app "commercial grade" and user-friendly, the GUI should emphasize visual assembly over manual YAML editing.

### 1. The Processing Job Wizard
Instead of a single complex page, the App will use a guided workflow:
- **Card-based Flow Selection**: Users pick from a library of visual "Recipes" (Standard MT, Quick Check, QC only).
- **Properties Pane for Parameters**: Selecting a "Recipe" opens a sidebar with categorized settings (Windowing, Solver, FFT). Changes here create a local `ParameterSet`.
- **Intelligent Site Selection**: Integrating the `Project Navigator` allows users to drag-and-drop sites/stations directly into the `Runtime Arguments` slot.

### 2. Batch Execution Interface
When multiple stations are selected in the Project Tree, the job becomes a batch job.
- **Validation**: The App automatically checks if all selected stations have the required sampling frequency for the chosen Parameter Set.
- **Queueing**: A background task manager (using Python's `multiprocessing` or `concurrent.futures`) processes the batch, updating a progress bar for each site.

---

## The Experimentation Workflow

Experimentation is at the heart of MT data processing. The decoupling of Flows and Parameters creates several "First-Class" experimentation features:

### 1. Parameter Sweeping (A/B Testing)
In the **Experimentation Page**, users can:
- Fix a **Flow** (e.g., Standard MT).
- Define a **Parameter Range** (e.g., Try window sizes: 512, 1024, 2048).
- **Execute & Compare**: The App executes all three jobs and overlays the resulting Transfer Functions (Apparent Resistivity/Phase) on a single plot.

### 2. Flow Comparison
Users can compare the impact of different algorithms (e.g., Comparing OLS with a new Robust Solver) by simply swapping the solver node in the flow designer while keeping parameters and data constant.

### 3. Iterative Refinement
1. **Quick Job**: Execute a standard flow with default parameters.
2. **Inspect**: Use the `Spectra Viewer` to identify noise.
3. **Adjust**: Create a new `ParameterSet` (e.g., add a Notch Filter).
4. **Repeat**: Execute the *same* Processing Job with the updated parameters. The system automatically creates a new labeled folder in `results/`, allowing side-by-side comparison of the improvement.

---

## Recommended DAG Libraries

To avoid reinventing the wheel, the following libraries are recommended for the `ExecutionEngine`:

### 1. Ploomber (Strongest for YAML-native)
- **Why**: Ploomber uses a `pipeline.yaml` as its primary definition of the DAG. It is lightweight, supports incremental builds (only re-run what changed), and is very well-suited for scientific Python.
- **Integration**: High. The `resistics-app` can generate `pipeline.yaml` files that Ploomber then executes.

### 2. Hamilton (Best logic, needs bridge)
- **Why**: You already liked Hamilton for its separation of static and runtime parameters. While it doesn't "boot" from YAML out of the box, it is trivial to write a small loader that takes a YAML list of process names and imports the corresponding Hamilton-annotated functions from a `resistics` registry.
- **Integration**: Medium. Excellent for "Experimentation" pages due to its declarative nature.

### 3. Kedro (Commercial Grade)
- **Why**: If this becomes a commercial app, Kedro's "Data Catalog" (driven by YAML) and strict modularity are industry standards for reproducible pipelines.
- **Integration**: Medium. Heavier than Ploomber/Hamilton but provides more "enterprise" features out of the box.

### 4. NetworkX + Custom Wrapper (Lightweight/Internal)
- **Why**: Since `resistics-app` already uses NetworkX, a simple custom wrapper using Pydantic for validation might be enough for the library's needs without adding a heavy third-party dependency.

---

## Processing Step Architecture

To balance robustness with ease of use, a **Hybrid Architecture** is recommended:

### 1. The Core: Class-based Processes
`resistics` will continue to use Pydantic-based classes for its internal processing steps.
- **Why**: Provides automated validation, easy serialization, and a clear distinction between **Static Parameters** (class fields) and **Runtime Data** (passed to `.run()`).

### 2. The User Interface: Functional Wrapper
To simplify experimentation for scientists, a `@resistics_process` decorator will be provided. This allows a standard Python function to behave like a `ResisticsProcess`.

```python
@resistics_process(name="MyCustomFilter")
def my_filter(data: TimeData, cutoff: float = 1.0) -> TimeData:
    # 'cutoff' is recognized as a static parameter
    # 'data' is recognized as runtime data
    return data.apply_filter(cutoff)
```

---

## Extensibility & Plugins

A key requirement is allowing users to easily add their own processing logic.

### 1. External Plugin Paths
Instead of a fixed internal folder, the `resistics.json` file will contain a `plugin_paths` key.
```json
{
  "mth5_path": "data.h5",
  "plugin_paths": [
    "/home/user/repos/my-custom-solvers",
    "C:\\Users\\scientist\\Documents\\resistics-addons"
  ]
}
```
- **Why**: This allows users to keep their plugins in their own Git repositories, shared across multiple resistics projects, and version-controlled independently.

### 2. Auto-Discovery Logic
The `ExecutionEngine` will include a `PluginLoader` that:
- Reads the `plugin_paths` from `resistics.json`.
- Scans these directories for `.py` files.
- Checks any paths in the `RESISTICS_PLUGINS` environment variable (for global/system-wide plugins).
- Imports the modules, triggering registration.
- Once registered, these custom steps appear in the **Flow Designer** and can be used in YAML flows.

### 3. Sharing and Testing
Since custom processes are just Python files (optionally with an associated YAML for default parameters), they can be easily shared between researchers or committed to a project's version control.

---

## Standalone vs. App
By keeping the logic and definitions in standard YAML files within the project, the system stays dual-use:
- **CLI/Notebook**: Power users can write Python scripts that iterate over these YAML files.
- **App**: Scientists can perform the same work visually, with the GUI acting as a "YAML Generator" and "Results Vizualiser".

---

## Future-Proofing: Language and Performance

The choice of Python for `resistics` is a balance between **Development Speed/Ecosystem** and **Execution Performance**.

### 1. Why Python Now?
- **Ecosystem**: Integration with `MTH5`, `Obspy`, `Pandas`, and `NumPy` is instantaneous.
- **User Base**: Most magnetotelluric scientists are familiar with Python, making the **Plugin System** viable.
- **Optimization**: Tools like `Numba` (JIT) and `Cython` can often bring Python code within 2x-5x of C/Rust performance for numerical kernels.

### 2. When to Consider Other Languages?
A sensible time to rewrite or port parts of the system is when:
- **Profiling Proves Bottlenecks**: If 90% of execution time is spent in a specific solver node that cannot be further optimized in Python.
- **Parallelization Complexity**: When Python's GIL (Global Interpreter Lock) prevents efficient scaling on multi-core HPC systems for your specific DAG.

### 3. Language Alternatives & Strategy
If performance becomes a critical barrier, a **Hybrid Approach** is usually superior to a full rewrite:

| Language | Role in Resistics | Pros/Cons |
| :--- | :--- | :--- |
| **Rust** | **High-Performance Kernels** | **Pros**: Memory safety, C-level speed, incredible Python integration via `PyO3`. **Cons**: Steep learning curve. |
| **Julia** | **Scientific Alternative** | **Pros**: Built for math, dynamic like Python. **Cons**: Smaller ecosystem for MT-specific formats (like MTH5). |
| **C++** | **Legacy Performance** | **Pros**: Industry standard. **Cons**: High maintenance overhead, memory safety risks. |
| **Mojo** | **The Wildcard** | **Pros**: Python compatibility with C speed. **Cons**: Too immature for production use today. |

**Recommended Path Forward**:
Stay in **Python** for the orchestration (DAG, Project Management, UI, Plugin Discovery). If performance lags, port the "Heaviest Nodes" (e.g., the Robust Solver kernel or Decimation filters) to **Rust** while keeping the app-facing interface in Python. This preserves the ease of use for the scientist while delivering "Commercial Grade" speed.
