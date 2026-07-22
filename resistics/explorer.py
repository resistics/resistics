"""Cached, UI-neutral project discovery for interactive explorers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import ClassVar, Literal, TypeVar

from pydantic import BaseModel, ConfigDict

from resistics.flow import FlowDefinition, ParameterSet, model_from_yaml_file
from resistics.gather import GatherCriteria
from resistics.job import (
    JobDefinition,
    JobSummary,
    JobValidation,
    ProjectJobs,
)
from resistics.project import MTH5FileSummary, Project, ProjectDataItem, RunSummary

type ResourceKind = Literal["flows", "parameters", "criteria", "jobs"]
type IndexSection = Literal["project", "flows", "parameters", "criteria", "jobs"]
type ResourceModel = FlowDefinition | ParameterSet | GatherCriteria | JobDefinition
_ProjectValue = TypeVar("_ProjectValue")


class ExplorerFileIdentity(BaseModel):
    """Filesystem identity used to reuse one parsed resource.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **path** — Project resource path.
    - **modified_ns** — Nanosecond modification timestamp reported by the filesystem.
    - **size** — File size in bytes.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    path: Path
    modified_ns: int
    size: int


class ExplorerIssue(BaseModel):
    """Non-fatal project discovery failure retained for presentation layers.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **section** — Explorer section that could not be read.
    - **message** — User-facing failure detail.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    section: str
    message: str


class IndexedResource(BaseModel):
    """Parsed YAML resource or its stable validation failure.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **kind** — Project resource namespace.
    - **identity** — File identity used as the parsed-value cache key.
    - **model** — Parsed model, or ``None`` when validation failed.
    - **error** — Validation error for malformed content.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    kind: ResourceKind
    identity: ExplorerFileIdentity
    model: ResourceModel | None
    error: str | None = None

    @property
    def path(self) -> Path:
        """Return the resource path represented by the identity."""
        return self.identity.path

    @property
    def is_valid(self) -> bool:
        """Return whether the resource parsed into its expected model."""
        return self.model is not None


class IndexedJob(BaseModel):
    """One cached job summary and its complete validation result.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **resource** — Parsed job file record.
    - **summary** — Display summary derived from validation.
    - **validation** — Cached resolved validation used for selection and execution.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    resource: IndexedResource
    summary: JobSummary
    validation: JobValidation


class ProjectExplorerState(BaseModel):
    """Cached project and MTH5 catalogue state without live file handles.

    **Attributes**

    - **model_config** — Pydantic frozen-model configuration.
    - **project_path** — Root of the indexed project.
    - **mth5_identity** — MTH5 path identity when the file can be statted.
    - **summary** — Cached MTH5 metadata summary.
    - **project_data_items** — Cached derived project artifact hierarchy.
    - **mth5_data_items** — Cached MTH5 group and dataset hierarchy.
    - **has_project_data_to_delete** — Whether the cached project catalogue contains removable derived data.
    - **issues** — Non-fatal discovery failures encountered while building the state.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    project_path: Path
    mth5_identity: ExplorerFileIdentity | None
    summary: MTH5FileSummary
    project_data_items: tuple[ProjectDataItem, ...]
    mth5_data_items: tuple[ProjectDataItem, ...]
    has_project_data_to_delete: bool
    issues: tuple[ExplorerIssue, ...] = ()


@dataclass(frozen=True)
class _ResourceCacheKey:
    """Private hash key for one parsed resource file version.

    **Attributes**

    - **kind** — Project resource namespace.
    - **path** — Project resource path.
    - **modified_ns** — Nanosecond modification timestamp reported by the filesystem.
    - **size** — File size in bytes.
    """

    kind: ResourceKind
    path: Path
    modified_ns: int
    size: int


class ProjectExplorerIndex:
    """Cache project discovery DTOs and parsed processing resources.

    The index owns no MTH5 or HDF5 handle. Cache hits perform no project or
    filesystem reads. Callers explicitly invalidate the affected section after
    internal mutations or invalidate every section before an external refresh.

    :param project: Open project that owns discovery operations and job validation.
    """

    def __init__(self, project: Project):
        self.project = project
        self.project_jobs = ProjectJobs(project)
        self._project_state: ProjectExplorerState | None = None
        self._runs: tuple[RunSummary, ...] | None = None
        self._resources: dict[ResourceKind, tuple[IndexedResource, ...]] = {}
        self._parsed_files: dict[_ResourceCacheKey, IndexedResource] = {}
        self._jobs: tuple[IndexedJob, ...] | None = None
        self._section_epochs: dict[IndexSection, int] = {
            "project": 0,
            "flows": 0,
            "parameters": 0,
            "criteria": 0,
            "jobs": 0,
        }
        self._jobs_epoch = 0
        self._lock = Lock()

    def project_state(self) -> ProjectExplorerState:
        """Return cached project discovery state, building it on a miss.

        :return: Handle-free project and MTH5 catalogue DTOs.
        """
        with self._lock:
            if self._project_state is not None:
                return self._project_state
            epoch = self._section_epochs["project"]

        issues: list[ExplorerIssue] = []
        summary = self.project.file_summary()
        project_items = self._project_values(
            "project data", self.project.list_project_data_items, issues
        )
        mth5_items = self._project_values(
            "MTH5 data", self.project.list_mth5_data_items, issues
        )
        try:
            has_project_data = bool(self.project.preview_project_data_deletion().paths)
        except Exception as exc:
            has_project_data = False
            issues.append(
                ExplorerIssue(section="project data deletion", message=str(exc))
            )
        state = ProjectExplorerState(
            project_path=self.project.project_path,
            mth5_identity=self._optional_identity(summary.mth5_path),
            summary=summary,
            project_data_items=tuple(project_items),
            mth5_data_items=tuple(mth5_items),
            has_project_data_to_delete=has_project_data,
            issues=tuple(issues),
        )
        with self._lock:
            if epoch == self._section_epochs["project"]:
                if self._project_state is None:
                    self._project_state = state
                return self._project_state
        return state

    def runs(self) -> tuple[RunSummary, ...]:
        """Return cached MTH5 run summaries, building them on first selection.

        :return: Stable run summaries used to resolve MTH5 data selections.
        """
        with self._lock:
            if self._runs is not None:
                return self._runs
            epoch = self._section_epochs["project"]
        runs = tuple(self.project.list_runs())
        with self._lock:
            if epoch == self._section_epochs["project"]:
                if self._runs is None:
                    self._runs = runs
                return self._runs
        return runs

    def resources(self, kind: ResourceKind) -> tuple[IndexedResource, ...]:
        """Return cached parsed resources for one project namespace.

        :param kind: Flow, parameter, criteria, or job namespace.

        :return: Stable path-ordered resource records, including malformed files.
        """
        with self._lock:
            cached = self._resources.get(kind)
            if cached is not None:
                return cached
            epoch = self._section_epochs[kind]
        directory = self.project.project_path / "processing" / kind
        records = tuple(
            self._resource(kind, identity)
            for identity in self._yaml_identities(directory)
        )
        with self._lock:
            if epoch == self._section_epochs[kind]:
                cached = self._resources.setdefault(kind, records)
                return cached
        return records

    def jobs(self) -> tuple[IndexedJob, ...]:
        """Return cached job summaries and loaded-resource validation results.

        :return: Path-ordered jobs, including malformed or unresolved definitions.
        """
        with self._lock:
            if self._jobs is not None:
                return self._jobs
            epoch = self._jobs_epoch
        jobs = tuple(self._indexed_job(resource) for resource in self.resources("jobs"))
        with self._lock:
            if epoch == self._jobs_epoch:
                if self._jobs is None:
                    self._jobs = jobs
                return self._jobs
        return jobs

    def job_validation(self, path: Path) -> JobValidation | None:
        """Return cached validation for one exact project job path.

        :param path: Job path selected by a caller.

        :return: Cached result, or ``None`` when the path is not indexed.
        """
        return next(
            (job.validation for job in self.jobs() if job.resource.path == path), None
        )

    def resource_for_path(
        self, kind: ResourceKind, path: Path
    ) -> IndexedResource | None:
        """Return one cached resource by exact path.

        :param kind: Resource namespace containing the path.
        :param path: Exact project resource path.

        :return: Matching record, if it exists in the cached namespace.
        """
        return next(
            (resource for resource in self.resources(kind) if resource.path == path),
            None,
        )

    def invalidate(self, *sections: IndexSection) -> None:
        """Invalidate selected sections and every dependent job result.

        :param *sections: Project or resource sections changed by an owning operation.
        """
        with self._lock:
            for section in sections:
                self._section_epochs[section] += 1
                if section == "project":
                    self._project_state = None
                    self._runs = None
                else:
                    self._resources.pop(section, None)
            self._jobs_epoch += 1
            self._jobs = None

    def invalidate_all(self) -> None:
        """Invalidate all discovery sections before an external refresh."""
        with self._lock:
            for section in self._section_epochs:
                self._section_epochs[section] += 1
            self._jobs_epoch += 1
            self._project_state = None
            self._runs = None
            self._resources.clear()
            self._jobs = None

    @staticmethod
    def _project_values(
        name: str,
        operation: Callable[[], list[_ProjectValue]],
        issues: list[ExplorerIssue],
    ) -> list[_ProjectValue]:
        """Run one non-critical project query and retain failures as issues.

        :param name: Section name recorded if the operation fails.
        :param operation: Project query returning explorer DTOs.
        :param issues: Mutable issue collection for non-fatal failures.

        :return: Query values, or an empty list after a failure.
        """
        try:
            return operation()
        except Exception as exc:
            issues.append(ExplorerIssue(section=name, message=str(exc)))
            return []

    @staticmethod
    def _optional_identity(path: Path) -> ExplorerFileIdentity | None:
        """Return a file identity, or ``None`` for an unavailable path.

        :param path: File whose identity may be unavailable.

        :return: Available identity, otherwise ``None``.
        """
        try:
            return ProjectExplorerIndex._identity(path)
        except OSError:
            return None

    @staticmethod
    def _identity(path: Path) -> ExplorerFileIdentity:
        """Return the modification-time and size identity for one file.

        :param path: Existing file to identify.

        :return: Stable identity for the current file version.
        """
        status = path.stat()
        return ExplorerFileIdentity(
            path=path, modified_ns=status.st_mtime_ns, size=status.st_size
        )

    @staticmethod
    def _yaml_identities(directory: Path) -> tuple[ExplorerFileIdentity, ...]:
        """Scan one resource directory into stable path-ordered identities.

        :param directory: Project resource directory to scan.

        :return: Identities for YAML files in path order.
        """
        paths = sorted(
            path
            for pattern in ("*.yaml", "*.yml")
            for path in directory.glob(pattern)
            if path.is_file()
        )
        return tuple(ProjectExplorerIndex._identity(path) for path in paths)

    def _resource(
        self, kind: ResourceKind, identity: ExplorerFileIdentity
    ) -> IndexedResource:
        """Return one identity-cached parsed resource or validation failure.

        :param kind: Resource namespace assigning the expected model.
        :param identity: Current file identity.

        :return: Parsed record or cached validation failure.
        """
        key = _ResourceCacheKey(
            kind=kind,
            path=identity.path,
            modified_ns=identity.modified_ns,
            size=identity.size,
        )
        with self._lock:
            cached = self._parsed_files.get(key)
            if cached is not None:
                return cached
        try:
            model = self._parse(kind, identity.path)
            resource = IndexedResource(kind=kind, identity=identity, model=model)
        except Exception as exc:
            resource = IndexedResource(
                kind=kind, identity=identity, model=None, error=str(exc)
            )
        with self._lock:
            return self._parsed_files.setdefault(key, resource)

    @staticmethod
    def _parse(kind: ResourceKind, path: Path) -> ResourceModel:
        """Parse one resource with the model assigned to its namespace.

        :param kind: Resource namespace assigning the expected model.
        :param path: YAML file to parse.

        :return: Validated resource model.
        """
        if kind == "flows":
            return model_from_yaml_file(FlowDefinition, path)
        if kind == "parameters":
            return model_from_yaml_file(ParameterSet, path)
        if kind == "criteria":
            return model_from_yaml_file(GatherCriteria, path)
        return model_from_yaml_file(JobDefinition, path)

    def _indexed_job(self, resource: IndexedResource) -> IndexedJob:
        """Resolve one parsed job through the cached resource namespaces.

        :param resource: Parsed or malformed job resource.

        :return: Display summary and complete validation result.
        """
        validation = self._validate_job_resource(resource)
        resolved = validation.resolved_job
        if resolved is None:
            summary = JobSummary(
                name=resource.path.stem,
                path=resource.path,
                errors=validation.errors,
                warnings=validation.warnings,
            )
        else:
            definition = resolved.definition
            summary = JobSummary(
                name=definition.name,
                path=resource.path,
                flow=definition.flow,
                parameters=definition.parameters,
                output_label=definition.output_label,
                is_valid=validation.ok,
                errors=validation.errors,
                warnings=validation.warnings,
            )
        return IndexedJob(
            resource=resource,
            summary=summary,
            validation=validation,
        )

    def _validate_job_resource(self, resource: IndexedResource) -> JobValidation:
        """Validate one job using only models cached by this index.

        :param resource: Parsed or malformed job resource.

        :return: Loaded-resource validation result.
        """
        if not isinstance(resource.model, JobDefinition):
            return JobValidation(
                ok=False, errors=[resource.error or "Job YAML is invalid"]
            )
        try:
            flow_resource = self._reference("flows", resource.model.flow)
            parameters_resource = self._reference(
                "parameters", resource.model.parameters
            )
            if not isinstance(flow_resource.model, FlowDefinition):
                return JobValidation(
                    ok=False,
                    errors=[flow_resource.error or "Flow YAML is invalid"],
                )
            if not isinstance(parameters_resource.model, ParameterSet):
                return JobValidation(
                    ok=False,
                    errors=[parameters_resource.error or "Parameter YAML is invalid"],
                )
            criteria_resource = None
            criteria = None
            if resource.model.criteria:
                criteria_resource = self._reference("criteria", resource.model.criteria)
                if not isinstance(criteria_resource.model, GatherCriteria):
                    return JobValidation(
                        ok=False,
                        errors=[criteria_resource.error or "Criteria YAML is invalid"],
                    )
                criteria = criteria_resource.model
        except Exception as exc:
            return JobValidation(ok=False, errors=[str(exc)])
        return self.project_jobs.validate_loaded(
            job_path=resource.path,
            definition=resource.model,
            flow_path=flow_resource.path,
            flow=flow_resource.model,
            parameters_path=parameters_resource.path,
            parameters=parameters_resource.model,
            criteria_path=(
                None if criteria_resource is None else criteria_resource.path
            ),
            criteria=criteria,
        )

    def _reference(self, kind: ResourceKind, name: str) -> IndexedResource:
        """Resolve one job reference within a cached resource namespace.

        :param kind: Referenced resource namespace.
        :param name: Exact filename or extension-free resource name.

        :return: Unique cached resource matching the reference.

        :raises ValueError: If the name is unsafe, missing, or ambiguous.
        """
        value = Path(name)
        if value.is_absolute() or value.parent != Path("."):
            raise ValueError(f"{kind[:-1].title()} must be a project file name")
        resources = self.resources(kind)
        candidates = (
            [resource for resource in resources if resource.path.name == value.name]
            if value.suffix in {".yaml", ".yml"}
            else [
                resource for resource in resources if resource.path.stem == value.name
            ]
        )
        if not candidates:
            raise ValueError(f"{kind[:-1].title()} not found: {name}")
        if len(candidates) > 1:
            names = ", ".join(resource.path.name for resource in candidates)
            raise ValueError(f"Ambiguous {kind[:-1]} {name!r}: {names}")
        return candidates[0]
