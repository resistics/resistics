"""Project overview, data catalogue, and plotting presentation."""

from __future__ import annotations

import json
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Any

from textual import on, work
from textual.widgets import Static, TabbedContent, TextArea, Tree
from textual.widgets._tree import TreeNode

from resistics.tui.logging import _exception_entry
from resistics.tui.screens.project_base import _ProjectExplorerBase
from resistics.tui.services import _feature_error
from resistics.tui.state import PlotTarget

if TYPE_CHECKING:
    from resistics.explorer import ProjectExplorerState
    from resistics.project import ProjectDataItem


type _DataTreeNode = TreeNode[Any]


class _ProjectDataMixin(_ProjectExplorerBase):
    """Present project overview, data hierarchy, and plot actions."""

    def _populate_overview(self, state: ProjectExplorerState) -> None:
        """Render one worker-loaded project summary.

        :param state: Handle-free project discovery result.
        """
        summary = state.summary
        self.action_state.plot_targets["project"] = (
            ("project", None) if summary.n_runs > 0 else None
        )
        content = (
            f"[b]{self.project.project_path.name}[/b]\n\n"
            f"Project: {self.project.project_path}\n"
            f"MTH5: {summary.mth5_path}\n"
            f"MTH5 version: {summary.file_version}\n"
            f"Reference time: {self.project.ref_time!s}\n"
            f"Time span: {summary.start_time or '-'} → {summary.end_time or '-'}\n"
            "Sample rates: "
            f"{', '.join(str(value) for value in summary.sample_rates) or '-'}\n\n"
            f"Surveys: {summary.n_surveys}\n"
            f"Stations: {summary.n_stations}\n"
            f"Runs: {summary.n_runs}\n"
            f"Channels: {summary.n_channels}"
        )
        self.query_one("#project-content", Static).update(content)

    def _populate_data_tree(self, state: ProjectExplorerState) -> None:
        """Populate the filtered Project and MTH5 data hierarchy.

        :param state: Handle-free project and MTH5 catalogue returned by a worker.
        """
        tree = self.query_one("#data-tree", Tree)
        tree.clear()
        tree.root.label = "Data"
        tree.root.data = None
        project_node = tree.root.add("Project", data=("project", "."))
        mth5_node = tree.root.add("MTH5", data=("mth5", "/"))
        self.data_items.clear()
        self.action_state.plot_targets["data"] = None
        self.action_state.has_project_data_to_delete = state.has_project_data_to_delete
        for issue in state.issues:
            self.notify(
                f"Unable to inspect {issue.section}: {issue.message}",
                severity="warning",
            )
        self._add_data_catalog(project_node, list(state.project_data_items))
        self._add_data_catalog(mth5_node, list(state.mth5_data_items))
        tree.root.expand()
        project_node.expand()
        mth5_node.expand()

    def _add_data_catalog(
        self, root: _DataTreeNode, items: list[ProjectDataItem]
    ) -> None:
        """Add each persistent data-type category below one source root.

        :param root: Source node receiving the category branches.
        :param items: Persistent items belonging to the source.
        """
        for label, data_type in self.DATA_CATEGORIES:
            matching = [item for item in items if item.data_type == data_type]
            category = root.add(
                f"{label} "
                f"({self.service.data_category_count(root.data, matching, data_type)})",
                data=("category", label),
            )
            self._add_data_items(category, items, data_type)

    def _add_data_items(
        self,
        root: _DataTreeNode,
        items: list[ProjectDataItem],
        data_type: str,
    ) -> None:
        """Add one category's items, retaining their path ancestors.

        :param root: Category node receiving visible items.
        :param items: Persistent items available below the source.
        :param data_type: Data type selected for this category.
        """
        visible = self.service.visible_data_paths(items, data_type)
        nodes: dict[str | None, _DataTreeNode] = {None: root}
        for item in sorted(
            (item for item in items if item.path in visible),
            key=lambda value: (value.path.count("/"), value.path),
        ):
            parent = nodes.get(item.parent_path, root)
            key = f"{item.source}:{item.path}"
            self.data_items[key] = item
            if item.kind in {"directory", "group"}:
                nodes[item.path] = parent.add(item.name, data=(item.source, item.path))
            else:
                parent.add_leaf(item.name, data=(item.source, item.path))

    def _data_item_for_node(
        self, node: _DataTreeNode | None = None
    ) -> ProjectDataItem | None:
        """Return the browsed data item for a tree node, excluding tree chrome.

        :param node: Flow node within the stage.
        :return: The browsed data item for a tree node, excluding tree chrome.
        """
        if node is None:
            node = self.query_one("#data-tree", Tree).cursor_node
        data = None if node is None else node.data
        if not isinstance(data, tuple) or len(data) != 2:
            return None
        source, path = data
        if source not in {"project", "mth5"}:
            return None
        return self.data_items.get(f"{source}:{path}")

    def _data_tree_cursor(self) -> _DataTreeNode | None:
        """Return the focused Data-tree node, if there is one.

        :return: Focused data-tree node, or ``None`` outside the active tree.
        """
        if self.query_one(TabbedContent).active != "data":
            return None
        tree = self.query_one("#data-tree", Tree)
        if self.focused is not tree:
            return None
        return tree.cursor_node

    def _data_plot_target(self, node: _DataTreeNode | None = None) -> PlotTarget | None:
        """Return a plot target for the highlighted item, if it is supported.

        :param node: Flow node within the stage.
        :return: A plot target for the highlighted item, if it is supported.
        """
        return self.service.data_plot_target(self._data_item_for_node(node))

    def _flow_plot_target(self) -> PlotTarget | None:
        """Return the focused or opened cached-valid flow plot target.

        :return: The focused or opened cached-valid flow plot target.
        """
        highlighted = self._highlighted_yaml_file()
        path: Path | None
        if highlighted is not None and highlighted[1] == "#flow-content":
            path = highlighted[0]
        else:
            path = self.selected_flow_path
        if path is None or path not in self.action_state.valid_flow_paths:
            return None
        return ("flow", path)

    def _job_plot_target(self) -> PlotTarget | None:
        """Return the focused or opened cached-valid job plot target.

        :return: The focused or opened cached-valid job plot target.
        """
        highlighted_path = self._highlighted_job_path()
        if highlighted_path is not None:
            summary = next(
                (
                    value
                    for value in self.job_summaries.values()
                    if value.path == highlighted_path
                ),
                None,
            )
            return (
                ("job", highlighted_path)
                if summary is not None and summary.is_valid
                else None
            )
        if (
            self.selected_job_path is not None
            and self.selected_validation is not None
            and self.selected_validation.ok
            and self.selected_validation.resolved_job is not None
        ):
            return ("job", self.selected_job_path)
        return None

    def _has_project_timeline(self) -> bool:
        """Return the cached project-timeline eligibility.

        :return: The cached project-timeline eligibility.
        """
        return self.action_state.plot_targets["project"] is not None

    def _start_project_plot(self) -> None:
        if not self._has_project_timeline():
            return
        self.notify("Opening project timeline")
        self._open_plot(("project", None))

    def _start_selected_data_plot(self) -> None:
        target = self.action_state.plot_targets["data"]
        if target is None:
            return
        self.notify("Opening plot")
        self._open_plot(target)

    def _start_selected_flow_plot(self) -> None:
        target = self._flow_plot_target()
        if target is None:
            return
        self.notify("Opening flow plot")
        self._open_plot(target)

    def _start_selected_job_plot(self) -> None:
        target = self._job_plot_target()
        if target is None:
            return
        self.notify("Opening job plot")
        self._open_plot(target)

    @work(thread=True, exclusive=True, group="plotting")
    def _open_plot(self, target: PlotTarget) -> None:
        """Open a selected Plotly figure in the browser.

        :param target: Validated plot kind and its typed payload.
        """
        plot_project = None
        target_type = target[0]
        plot_name = {
            "flow": "flow plot",
            "job": "job plot",
            "project": "project timeline",
        }.get(target_type, "plot")
        started = monotonic()
        try:
            import plotly.io as pio

            from resistics.project import load

            self.app.call_from_thread(self.notify, f"Building {plot_name}")
            plot_project = load(self.project.project_path)
            figure = self.service.build_plot_figure(plot_project, target)
            build_seconds = monotonic() - started
            self.app.call_from_thread(
                self.notify,
                f"{plot_name.capitalize()} built in {build_seconds:.1f}s; "
                "opening browser",
            )
            pio.show(figure)
            self.app.call_from_thread(self.notify, f"{plot_name.capitalize()} opened")
        except Exception as exc:
            message = f"Unable to open {plot_name}: {_feature_error('Plotting', exc)}"
            self.log_buffer.append(_exception_entry("Plotting", message, exc))
            self.app.call_from_thread(
                self.notify,
                f"{message}\nSee Session logs for the full traceback.",
                severity="error",
                markup=False,
            )
        finally:
            if plot_project is not None and plot_project is not self.project:
                plot_project.close()

    @on(Tree.NodeSelected, "#data-tree")
    def show_data_metadata(self, event: Tree.NodeSelected) -> None:
        item = event.node.data
        details = self.query_one("#data-metadata", TextArea)
        if item is None:
            details.text = json.dumps(
                {"message": "Select Project or MTH5 data"}, indent=2
            )
            return
        try:
            source, path = item
            if source == "category":
                details.text = json.dumps(
                    {"message": f"Expand {path} to inspect its data."}, indent=2
                )
                return
            if source == "project" and Path(path).suffix.lower() == ".json":
                details.text = json.dumps(
                    self.project.get_project_data_json(path), indent=2
                )
            else:
                metadata = (
                    self.project.get_project_data_metadata(path)
                    if source == "project"
                    else self.project.get_mth5_data_metadata(path)
                )
                details.text = metadata.model_dump_json(indent=2)
        except Exception as exc:
            details.text = json.dumps({"error": str(exc)}, indent=2)

    @on(Tree.NodeHighlighted, "#data-tree")
    def update_data_plot_selection(self, event: Tree.NodeHighlighted) -> None:
        """Cache plot eligibility when the highlighted data item changes.

        :param event: Event used by this operation.
        """
        self.action_state.plot_targets["data"] = self._data_plot_target(event.node)
        self.refresh_bindings()

    def _check_data_tree_action(self, action: str) -> bool:
        """Check expansion eligibility from the current in-memory tree node.

        :param action: Expansion or collapse action name.

        :return: Whether the tree action is currently available.
        """
        node = self._data_tree_cursor()
        if node is None or not node.allow_expand:
            return False
        return (
            not node.is_expanded if action == "expand_data_node" else node.is_expanded
        )

    def _check_plot_action(self, active: str) -> bool:
        """Check plot eligibility using cached project and selection state.

        :param active: Identifier of the active tab.

        :return: Whether the active tab has a valid cached plot target.
        """
        if active in {"project", "data"}:
            return self.action_state.plot_targets[active] is not None
        if active == "flows":
            return not self.editing_yaml and self._flow_plot_target() is not None
        if active == "jobs":
            return not self.editing_yaml and self._job_plot_target() is not None
        return False

    def action_plot(self) -> None:
        """Plot the project, highlighted data, selected flow, or selected job."""
        active = self.query_one(TabbedContent).active
        if active == "project":
            self._start_project_plot()
        elif active == "data":
            self._start_selected_data_plot()
        elif active == "flows":
            self._start_selected_flow_plot()
        elif active == "jobs":
            self._start_selected_job_plot()

    def action_expand_data_node(self) -> None:
        """Expand the highlighted Data-tree branch and all of its descendants."""
        node = self._data_tree_cursor()
        if node is not None and node.allow_expand:
            node.expand_all()
            self.refresh_bindings()

    def action_collapse_data_node(self) -> None:
        """Collapse the highlighted Data-tree branch and all of its descendants."""
        node = self._data_tree_cursor()
        if node is not None and node.allow_expand:
            node.collapse_all()
            self.refresh_bindings()
